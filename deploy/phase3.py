"""Phase 3 experiment: every routing policy on the same trace, against the same workers.

For each load level and each policy ("cell"):
  1. reset every worker's prefix cache (vLLM: POST /reset_prefix_cache, dev mode) so each
     policy starts cold; verify the reset succeeded
  2. start a fresh router with that policy (in this process)
  3. replay the trace through the router (open loop, real time)
  4. save per-request records, the router's decision log, worker counter deltas, a summary

Requests scheduled during the first `warmup_s` are excluded from the summaries.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

from bench.analyze import plot_cells, summarize_cell
from bench.loadgen import run as replay
from bench.loadgen import write_jsonl
from bench.workload import ImagePool, RequestOptions, TraceConfig, fingerprint, generate_trace, summarize_trace
from kvsim.config import CostModel
from router.load import parse_vllm_metrics
from router.policies import POLICIES
from router.server import RouterConfig, RouterRunner


@dataclass
class Phase3Config:
    policies: tuple[str, ...] = POLICIES
    active_convs: tuple[int, ...] = (24,)
    duration_s: float = 300.0
    warmup_s: float = 60.0
    seed: int = 0
    alpha: float = 1.0
    slo_ttft_s: float = 2.0
    slo_itl_s: float = 0.1
    time_scale: float = 1.0
    trace_overrides: dict = field(default_factory=dict)   # e.g. {"think_mean_s": 4} for quick runs


async def _post_json(session, url, timeout=30):
    async with session.post(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        try:
            return r.status, await r.json(content_type=None)
        except Exception:  # noqa: BLE001
            return r.status, None


async def reset_workers(urls: list[str], tries: int = 10) -> list[bool]:
    """Clear each worker's prefix cache. vLLM refuses while requests are running, so retry."""
    ok = []
    async with aiohttp.ClientSession() as s:
        for u in urls:
            success = False
            for _ in range(tries):
                status, body = await _post_json(s, u.rstrip("/") + "/reset_prefix_cache")
                success = status == 200 and (body is None or body.get("success", True) is not False)
                if success:
                    break
                await asyncio.sleep(3)
            ok.append(success)
    return ok


async def worker_counters(urls: list[str]) -> list[dict]:
    out = []
    async with aiohttp.ClientSession() as s:
        for u in urls:
            try:
                async with s.get(u.rstrip("/") + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    out.append(parse_vllm_metrics(await r.text()))
            except Exception as e:  # noqa: BLE001
                out.append({"error": str(e)})
    return out


def _delta(before: list[dict], after: list[dict], key: str) -> float | None:
    vals = [a.get(key) - b.get(key) for a, b in zip(after, before)
            if a.get(key) is not None and b.get(key) is not None]
    return sum(vals) if vals else None


async def run_cell(urls: list[str], policy: str, trace, pool: ImagePool, cost: CostModel,
                   cfg: Phase3Config, out_dir: Path, opts: RequestOptions, progress=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_ok = await reset_workers(urls)
    before = await worker_counters(urls)
    rcfg = RouterConfig(workers=urls, policy=policy, alpha=cfg.alpha, cost=cost, seed=cfg.seed)
    t0 = time.perf_counter()
    async with RouterRunner(rcfg) as rr:
        await asyncio.sleep(1.0)                 # first /metrics scrape (KV capacity)
        capacity = [c.capacity for c in rr.router.index.caches]
        progress(f"  {policy}: replaying {trace.n_requests()} requests (prints every 60 s)")
        records = await replay(trace, pool, [rr.url], time_scale=cfg.time_scale, opts=opts,
                               progress=progress, progress_every_s=60.0 / cfg.time_scale)
        decisions = list(rr.router.log_rows)
    wall = time.perf_counter() - t0
    after = await worker_counters(urls)

    recs = [asdict(r) for r in records]
    write_jsonl(records, out_dir / "requests.jsonl", meta={"policy": policy,
                "trace_fingerprint": fingerprint(trace), "trace": summarize_trace(trace)})
    with (out_dir / "decisions.jsonl").open("w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    max_tok = {(c.conv_id, k): t.max_tokens for c in trace.conversations for k, t in enumerate(c.turns)}
    summary = summarize_cell(recs, decisions, cfg.warmup_s / cfg.time_scale, cfg.slo_ttft_s,
                             cfg.slo_itl_s, cost.block_size, max_tok)
    all_ok = [r for r in recs if r.get("error") is None]
    cell = {
        "policy": policy, "active_convs": trace.config.active_convs,
        "trace_fingerprint": fingerprint(trace), "wall_s": round(wall, 1),
        "reset_ok": reset_ok, "router_capacity_blocks": capacity,
        "server": {"prefix_hits": _delta(before, after, "prefix_hits"),
                   "prefix_queries": _delta(before, after, "prefix_queries")},
        "client_all": {"cached_tokens": sum(r.get("cached_tokens") or 0 for r in all_ok),
                       "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in all_ok),
                       "requests": len(recs)},
        "summary": summary,
    }
    (out_dir / "cell.json").write_text(json.dumps(cell, indent=2))
    s = summary
    progress(f"  {policy:17s} hit {s['hit_rate'] or 0:.2f}  TTFT p50 {s['ttft_p50_s'] or 0:.2f}s "
             f"p99 {s['ttft_p99_s'] or 0:.2f}s  ITL p99 {s['itl_p99_s'] or 0:.3f}s  "
             f"goodput {s['goodput_frac'] or 0:.2f}  errors {s['errors']}  workers {s['per_worker']}")
    return cell


async def run_phase3(urls: list[str], cfg: Phase3Config, out: Path, cost: CostModel,
                     pool: ImagePool, opts: RequestOptions, progress=print,
                     title: str = "Routing policies") -> list[dict]:
    cells = []
    for n in cfg.active_convs:
        trace = generate_trace(TraceConfig(seed=cfg.seed, active_convs=n,
                                           duration_s=cfg.duration_s + cfg.warmup_s,
                                           **cfg.trace_overrides))
        progress(f"\nload: {n} active conversations, trace {json.dumps(summarize_trace(trace))}")
        for pol in cfg.policies:
            cells.append(await run_cell(urls, pol, trace, pool, cost, cfg,
                                        out / f"convs{n}" / pol, opts, progress))
    (out / "cells.json").write_text(json.dumps(cells, indent=2))
    plot_cells(cells, out / "phase3.png", title)
    return cells
