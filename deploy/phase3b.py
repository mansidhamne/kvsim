"""Phase 3b experiment runner: the Phase 3 audit fixes plus real-traffic workloads.

Differences from deploy/phase3.py (kept unchanged so runs 1-2 stay reproducible):
  * workloads: synthetic | visionarena | mooncake (bench/traces/)
  * between cells, ALL worker caches are reset: prefix, multimodal processor, encoder
  * a discarded warm-up replay at the start of every session (kernels compile, caches warm)
  * several seeds per load; the policy ORDER is rotated with the seed, so no policy always
    runs first or last
  * think time starts when the previous reply finished; warm-up filter by conversation start
  * per-token inter-token latency (continuous usage stats), preemption counts per cell
  * workload statistics (reuse ceiling, live working set) saved next to every trace

The KV cache size is fixed per worker process, so cache-size sweeps restart the workers; the
caller (deploy/kaggle_run.py phase3b) loops over cache sizes and calls `run_session` for each.

Results layout: OUT/<workload>/cache-<label>/load-<value>/seed-<n>/<policy>/{cell.json,
requests.jsonl, decisions.jsonl}; OUT/<workload>/cache-<label>/load-<value>/seed-<n>/workload.json.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

from bench.analyze import summarize_cell
from bench.loadgen import run as replay
from bench.loadgen import write_jsonl
from bench.traces import mooncake as mc
from bench.traces import visionarena as va
from bench.traces.stats import workload_stats
from bench.workload import ImagePool, RequestOptions, TraceConfig, fingerprint, generate_trace
from kvsim.config import CostModel
from router.server import RouterConfig, RouterRunner

from .phase3 import _delta, _post_json, worker_counters

PHASE3B_POLICIES = ("round_robin", "least_requests", "session_affinity", "cache_aware")


@dataclass
class Phase3bConfig:
    workload: str = "visionarena"              # synthetic | visionarena | mooncake
    # Load knob per workload: synthetic = active conversations, visionarena = conversation
    # starts per second, mooncake = time dilation (1 = real speed; 2 = half the request rate).
    loads: tuple[float, ...] = (0.3,)
    seeds: tuple[int, ...] = (0, 1, 2)
    policies: tuple[str, ...] = PHASE3B_POLICIES
    duration_s: float = 300.0                  # measured span of conversation starts
    warmup_s: float = 60.0                     # conversations starting earlier are not measured
    session_warmup_s: float = 90.0             # discarded replay before the first cell
    alpha: float = 1.0
    slo_ttft_s: float = 2.0
    slo_itl_s: float = 0.1
    time_scale: float = 1.0                    # >1 only for mock tests
    va_paths: tuple[str, ...] = ()             # VisionArena parquet files (local paths)
    va_max_rows: int = 5000
    va_overrides: dict = field(default_factory=dict)
    mooncake_path: str = ""
    mooncake_overrides: dict = field(default_factory=dict)
    synthetic_overrides: dict = field(default_factory=dict)
    history: str = "own"                       # "own" replies in follow-ups (see RequestOptions)


class TraceFactory:
    """Builds the trace for (load, seed); loads source data once."""

    def __init__(self, cfg: Phase3bConfig):
        self.cfg = cfg
        self._va_rows = None
        self._mc_rows = None

    def build(self, load: float, seed: int, duration_s: float | None = None):
        c = self.cfg
        dur = (duration_s if duration_s is not None else c.duration_s + c.warmup_s)
        if c.workload == "synthetic":
            tr = generate_trace(TraceConfig(seed=seed, active_convs=int(load), duration_s=dur,
                                            **c.synthetic_overrides))
            return tr, ImagePool(), {}
        if c.workload == "visionarena":
            if self._va_rows is None:
                self._va_rows = va.load_rows(list(c.va_paths), c.va_max_rows)
            vcfg = va.VAConfig(**{**c.va_overrides, "seed": seed, "conv_rate": float(load),
                                  "duration_s": dur})
            tr, pool, stats = va.build_trace(self._va_rows, vcfg)
            return tr, pool, stats
        if c.workload == "mooncake":
            if self._mc_rows is None:
                self._mc_rows = mc.load_rows(c.mooncake_path)
            # seeds pick different windows of the hour (the trace itself is fixed)
            base = mc.MooncakeConfig(**c.mooncake_overrides)
            mcfg = mc.MooncakeConfig(**{**asdict(base), "time_dilation": float(load),
                                        "duration_s": dur, "start_s": base.start_s + 900.0 * seed})
            tr, stats = mc.build_trace(self._mc_rows, mcfg)
            return tr, ImagePool(), stats
        raise ValueError(f"unknown workload {c.workload!r}")


async def reset_all(urls: list[str], tries: int = 10) -> list[dict]:
    """Reset prefix, multimodal-processor and encoder caches on every worker. vLLM refuses the
    prefix reset while requests run, so retry. mm/encoder resets return 200 without a body."""
    out = []
    async with aiohttp.ClientSession() as s:
        for u in urls:
            base = u.rstrip("/")
            res = {}
            for _ in range(tries):
                status, body = await _post_json(s, base + "/reset_prefix_cache")
                res["prefix"] = status == 200 and (body is None or body.get("success", True) is not False)
                if res["prefix"]:
                    break
                await asyncio.sleep(3)
            for name in ("mm", "encoder"):
                status, _ = await _post_json(s, f"{base}/reset_{name}_cache")
                res[name] = status == 200
            out.append(res)
    return out


def rotated(policies: tuple[str, ...], k: int) -> list[str]:
    k %= len(policies)
    return list(policies[k:] + policies[:k])


async def run_cell(urls, policy, trace, pool, cost: CostModel, cfg: Phase3bConfig, out_dir: Path,
                   opts: RequestOptions, labels: dict, progress=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    resets = await reset_all(urls)
    before = await worker_counters(urls)
    rcfg = RouterConfig(workers=urls, policy=policy, alpha=cfg.alpha, cost=cost, seed=labels["seed"])
    t0 = time.perf_counter()
    async with RouterRunner(rcfg) as rr:
        await asyncio.sleep(1.0)                  # first /metrics scrape (KV capacity)
        capacity = [c.capacity for c in rr.router.index.caches]
        progress(f"    {policy}: {trace.n_requests()} requests")
        records = await replay(trace, pool, [rr.url], time_scale=cfg.time_scale, opts=opts,
                               progress=progress, progress_every_s=60.0 / cfg.time_scale)
        decisions = list(rr.router.log_rows)
        retries = rr.router.counters.get("retries", 0)
    wall = time.perf_counter() - t0
    after = await worker_counters(urls)

    recs = [asdict(r) for r in records]
    write_jsonl(records, out_dir / "requests.jsonl",
                meta={"policy": policy, **labels, "trace_fingerprint": fingerprint(trace)})
    with (out_dir / "decisions.jsonl").open("w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    max_tok = {(c.conv_id, k): t.max_tokens for c in trace.conversations for k, t in enumerate(c.turns)}
    summary = summarize_cell(recs, decisions, cfg.warmup_s / cfg.time_scale, cfg.slo_ttft_s,
                             cfg.slo_itl_s, cost.block_size, max_tok)
    ok = [r for r in recs if r.get("error") is None]
    summary["multi_token_chunk_share"] = (sum(r.get("multi_token_chunks", 0) for r in ok)
                                          / max(1, sum(r.get("output_chunks", 0) for r in ok)))
    cell = {
        "policy": policy, **labels, "trace_fingerprint": fingerprint(trace),
        "wall_s": round(wall, 1), "resets": resets, "router_capacity_blocks": capacity,
        "router_retries": retries, "history": opts.history,
        # per request: hash of the generated reply. With history="own" a follow-up's prompt
        # contains the previous reply, so policies see identical prompts only while replies match
        "reply_shas": {f"{r['conv_id']}:{r['turn']}": r.get("reply_sha") for r in ok},
        "server": {k: _delta(before, after, k) for k in ("prefix_hits", "prefix_queries", "preemptions")},
        "client_all": {"cached_tokens": sum(r.get("cached_tokens") or 0 for r in ok),
                       "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in ok),
                       "requests": len(recs)},
        "summary": summary,
    }
    (out_dir / "cell.json").write_text(json.dumps(cell, indent=2))
    s = summary
    progress(f"    {policy:17s} hit {s['hit_rate'] or 0:.2f}  TTFT p50 {s['ttft_p50_s'] or 0:.2f}s "
             f"p99 {s['ttft_p99_s'] or 0:.2f}s  ITL p99 {s['itl_p99_s'] or 0:.3f}s  goodput "
             f"{s['goodput_frac'] or 0:.2f}  errors {s['errors']}  preempt {cell['server']['preemptions']}")
    return cell


async def session_warmup(urls, factory: TraceFactory, cfg: Phase3bConfig, cost: CostModel,
                         opts: RequestOptions, progress=print) -> None:
    """Discarded replay so the first measured cell doesn't pay for kernel compilation."""
    if cfg.session_warmup_s <= 0:
        return
    tr, pool, _ = factory.build(cfg.loads[0], seed=10_000, duration_s=cfg.session_warmup_s)
    progress(f"  session warm-up: {tr.n_requests()} requests (not measured)")
    async with RouterRunner(RouterConfig(workers=urls, policy="round_robin", cost=cost)) as rr:
        await asyncio.sleep(1.0)
        await replay(tr, pool, [rr.url], time_scale=cfg.time_scale, opts=opts)


async def run_session(urls: list[str], cfg: Phase3bConfig, out: Path, cost: CostModel,
                      cache_label: str, opts: RequestOptions, progress=print) -> list[dict]:
    factory = TraceFactory(cfg)
    await session_warmup(urls, factory, cfg, cost, opts, progress)
    cells = []
    for load in cfg.loads:
        for i, seed in enumerate(cfg.seeds):
            trace, pool, build_stats = factory.build(load, seed)
            ws = workload_stats(trace, pool)
            d = out / cfg.workload / f"cache-{cache_label}" / f"load-{load:g}" / f"seed-{seed}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "workload.json").write_text(json.dumps(
                {"build": build_stats, "stats": ws, "fingerprint": fingerprint(trace),
                 "source": trace.source}, indent=2))
            order = rotated(cfg.policies, i)
            progress(f"\n  {cfg.workload} cache={cache_label} load={load:g} seed={seed}: "
                     f"{trace.n_requests()} requests, reuse ceiling {ws['reuse_ceiling']:.2f}, "
                     f"live set ~{ws['peak_live_tokens']} tokens; order {order}")
            labels = {"workload": cfg.workload, "cache": cache_label, "load": load, "seed": seed,
                      "order_index": None, "reuse_ceiling": ws["reuse_ceiling"]}
            for j, pol in enumerate(order):
                labels["order_index"] = j
                cells.append(await run_cell(urls, pol, trace, pool, cost, cfg, d / pol, opts,
                                            dict(labels), progress))
    return cells
