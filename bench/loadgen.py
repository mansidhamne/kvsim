"""Phase 1: replay a trace against an OpenAI-compatible endpoint and record every request.

Timing model
* Open-loop ACROSS conversations: every conversation starts at its scheduled time, no matter
  how slow the server is. (A closed loop would let a slow policy receive less load, which hides
  its slowness.)
* Within a conversation, turn k+1 is sent at max(its scheduled time, when turn k finished):
  a user can't ask a follow-up before seeing the answer. The delay this adds is recorded as
  `late_s` so you can tell when the system was too slow for the trace.
* `time_scale` > 1 compresses the schedule (for tests against fast mock workers).

Targets
* One URL (normally the router, Phase 3), or several workers with a simple client-side
  choice (`round_robin` | `random` | `session`) so Phase 1 can run without a router.

Output: one JSON object per request (JSONL), see `Record`.

    python -m bench.loadgen --targets http://127.0.0.1:8001 --active-convs 10 --duration 60 \
        --out results/phase1/run.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import numpy as np

from .workload import (ImagePool, RequestOptions, Trace, TraceConfig, build_request,
                       fingerprint, generate_trace, summarize_trace)


@dataclass
class Record:
    conv_id: int
    turn: int
    target: str
    worker: str | None          # x-kvroute-worker response header, if any
    scheduled_s: float          # from the trace (after time_scale)
    sent_s: float               # relative to run start
    late_s: float               # sent - scheduled (waiting for the previous turn)
    ttft_s: float | None        # first non-empty content token
    e2e_s: float | None
    itl_s: list[float] = field(default_factory=list)   # inter-token latency, one value PER TOKEN
    output_chunks: int = 0
    multi_token_chunks: int = 0  # chunks that carried >1 token (their gap is split across them)
    conv_start_s: float = 0.0    # when this request's conversation started (warm-up filter)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None   # from the server's usage report (exact token count)
    cached_tokens: int | None = None
    mm_tokens: dict | None = None     # prompt_tokens_details.multimodal_tokens, e.g. {"image": 512}
    output_text: str = ""        # first ~300 chars of the reply (sanity check, e.g. fp16 garbage)
    status: int | None = None
    error: str | None = None


class TargetPicker:
    def __init__(self, targets: list[str], mode: str, seed: int = 0):
        if not targets:
            raise ValueError("need at least one target")
        self.targets, self.mode = targets, mode
        self._i = 0
        self._rng = random.Random(seed)
        self._home: dict[int, str] = {}

    def pick(self, conv_id: int) -> str:
        if len(self.targets) == 1:
            return self.targets[0]
        if self.mode == "round_robin":
            t = self.targets[self._i % len(self.targets)]
            self._i += 1
            return t
        if self.mode == "random":
            return self._rng.choice(self.targets)
        if self.mode == "session":
            if conv_id not in self._home:
                self._home[conv_id] = self.targets[len(self._home) % len(self.targets)]
            return self._home[conv_id]
        raise ValueError(f"unknown pick mode {self.mode!r}")


async def _send(session: aiohttp.ClientSession, url: str, body: dict, headers: dict,
                rec: Record, t_run0: float, timeout_s: float) -> None:
    t0 = time.perf_counter()
    rec.sent_s = t0 - t_run0
    last = None
    tokens_seen = None           # completion tokens reported up to the previous content chunk
    text: list[str] = []
    try:
        async with session.post(url.rstrip("/") + "/v1/chat/completions", json=body,
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            rec.status = r.status
            rec.worker = r.headers.get("x-kvroute-worker")
            if r.status != 200:
                rec.error = (await r.text())[:500]
                return
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                chunk_tokens = None
                if obj.get("usage"):
                    u = obj["usage"]
                    # with continuous_usage_stats every chunk carries usage; later chunks may omit
                    # prompt details, so only overwrite fields that are present
                    if u.get("prompt_tokens") is not None:
                        rec.prompt_tokens = u["prompt_tokens"]
                    if u.get("completion_tokens") is not None:
                        chunk_tokens = u["completion_tokens"]
                        rec.completion_tokens = chunk_tokens
                    det = u.get("prompt_tokens_details") or {}
                    if det.get("cached_tokens") is not None:
                        rec.cached_tokens = det["cached_tokens"]
                    if det.get("multimodal_tokens") is not None:
                        rec.mm_tokens = det["multimodal_tokens"]
                for ch in obj.get("choices") or []:
                    content = (ch.get("delta") or {}).get("content")
                    now = time.perf_counter()
                    if content:
                        if last is None:
                            rec.ttft_s = now - t0
                        else:
                            # tokens this chunk advanced (per-chunk usage), so a chunk carrying k
                            # tokens contributes k gaps of (gap / k), not one long gap
                            k = 1
                            if chunk_tokens is not None and tokens_seen is not None:
                                k = max(1, chunk_tokens - tokens_seen)
                            if k > 1:
                                rec.multi_token_chunks += 1
                            g = (now - last) / k
                            rec.itl_s.extend([round(g, 5)] * k)
                        last = now
                        if chunk_tokens is not None:
                            tokens_seen = chunk_tokens
                        rec.output_chunks += 1
                        if len(text) < 300:
                            text.append(content)
                    elif ch.get("finish_reason") and rec.ttft_s is None:
                        # A token was generated but decoded to "" (e.g. a partial UTF-8 byte or
                        # a special token, common with max_tokens=1): the finish chunk is the
                        # first sign of it.
                        rec.ttft_s = now - t0
        rec.output_text = "".join(text)[:300]
        rec.e2e_s = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - every failure is recorded, never raised
        rec.error = f"{type(e).__name__}: {e}"[:500]


async def run(trace: Trace, pool: ImagePool, targets: list[str], *, pick: str = "round_robin",
              time_scale: float = 1.0, opts: RequestOptions = RequestOptions(),
              timeout_s: float = 600.0, max_connections: int = 512,
              progress=None, progress_every_s: float = 60.0) -> list[Record]:
    """`progress(line)` is called every `progress_every_s` with a one-line status (requests
    done / total, errors, TTFT p99 over the last interval), so a long replay never looks hung."""
    seed = trace.config.get("seed", 0) if isinstance(trace.config, dict) else trace.config.seed
    picker = TargetPicker(targets, pick, seed)
    records: list[Record] = []
    conn = aiohttp.TCPConnector(limit=max_connections)
    async with aiohttp.ClientSession(connector=conn) as session:
        t_run0 = time.perf_counter()

        async def conversation(conv) -> None:
            conv_start = conv.turns[0].at / time_scale
            prev_end = None
            for k, turn in enumerate(conv.turns):
                if k and turn.think_s is not None and prev_end is not None:
                    # the user reads the reply, then thinks: send think_s after it finished
                    sched = prev_end + turn.think_s / time_scale
                else:
                    sched = turn.at / time_scale
                delay = sched - (time.perf_counter() - t_run0)
                if delay > 0:
                    await asyncio.sleep(delay)
                body, headers = build_request(trace, conv, k, pool, opts)
                target = picker.pick(conv.conv_id)
                rec = Record(conv.conv_id, k, target, None, round(sched, 4), 0.0, 0.0, None, None)
                rec.conv_start_s = round(conv_start, 4)
                await _send(session, target, body, headers, rec, t_run0, timeout_s)
                rec.late_s = max(0.0, rec.sent_s - sched)
                records.append(rec)
                prev_end = rec.sent_s + (rec.e2e_s or 0.0)
                if rec.error:            # a failed turn ends the conversation
                    return

        total = trace.n_requests()
        last_end = max((t.at for c in trace.conversations for t in c.turns), default=0.0) / time_scale

        async def heartbeat():
            seen = 0
            while True:
                await asyncio.sleep(progress_every_s)
                recent = records[seen:]
                seen = len(records)
                ttft = sorted(r.ttft_s for r in recent if r.ttft_s is not None)
                p99 = f"{ttft[min(len(ttft) - 1, int(0.99 * len(ttft)))]:.2f}s" if ttft else "-"
                el = time.perf_counter() - t_run0
                progress(f"    {el:5.0f}s / ~{last_end:.0f}s  done {len(records)}/{total}  "
                         f"errors {sum(1 for r in records if r.error)}  TTFT p99 (last {progress_every_s:.0f}s) {p99}")

        hb = asyncio.create_task(heartbeat()) if progress else None
        try:
            await asyncio.gather(*(conversation(c) for c in trace.conversations))
        finally:
            if hb:
                hb.cancel()
    records.sort(key=lambda r: (r.sent_s, r.conv_id, r.turn))
    return records


def summarize(records: list[Record]) -> dict:
    ok = [r for r in records if r.error is None and r.ttft_s is not None]
    ttft = np.array([r.ttft_s for r in ok]) if ok else np.array([0.0])
    itl = np.array([g for r in ok for g in r.itl_s]) if any(r.itl_s for r in ok) else np.array([0.0])
    pt = sum(r.prompt_tokens or 0 for r in ok)
    ct = sum(r.cached_tokens or 0 for r in ok)
    per_target: dict[str, int] = {}
    for r in ok:
        key = r.worker or r.target
        per_target[key] = per_target.get(key, 0) + 1
    return {
        "requests": len(records), "ok": len(ok), "errors": len(records) - len(ok),
        "ttft_p50_s": round(float(np.percentile(ttft, 50)), 4),
        "ttft_p99_s": round(float(np.percentile(ttft, 99)), 4),
        "itl_p50_s": round(float(np.percentile(itl, 50)), 4),
        "itl_p99_s": round(float(np.percentile(itl, 99)), 4),
        "hit_rate": round(ct / pt, 4) if pt else None,
        "late_turns": sum(r.late_s > 0.05 for r in ok),
        "per_worker": dict(sorted(per_target.items())),
    }


def write_jsonl(records: list[Record], path: str | Path, meta: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        if meta is not None:
            f.write(json.dumps({"meta": meta}) + "\n")
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a multi-turn VLM trace.")
    ap.add_argument("--targets", required=True, help="comma-separated base URLs")
    ap.add_argument("--pick", default="round_robin", choices=["round_robin", "random", "session"])
    ap.add_argument("--trace", help="trace JSON from `python -m bench.workload`; else generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--active-convs", type=int, default=10)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--images", help="directory of real photos; default synthetic images")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--no-image-uuids", action="store_true")
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trace = Trace.load(a.trace) if a.trace else generate_trace(
        TraceConfig(seed=a.seed, active_convs=a.active_convs, duration_s=a.duration))
    pool = ImagePool(a.images)
    opts = RequestOptions(model=a.model, send_image_uuids=not a.no_image_uuids)
    print("trace:", json.dumps(summarize_trace(trace)), flush=True)
    recs = asyncio.run(run(trace, pool, a.targets.split(","), pick=a.pick,
                           time_scale=a.time_scale, opts=opts))
    meta = {"trace": summarize_trace(trace), "targets": a.targets, "pick": a.pick,
            "time_scale": a.time_scale, "model": a.model, "images": a.images or "synthetic",
            "trace_fingerprint": fingerprint(trace)}
    write_jsonl(recs, a.out, meta)
    print("summary:", json.dumps(summarize(recs)))


if __name__ == "__main__":
    main()
