"""Block 6: turn raw per-request records into the numbers we compare policies on.

Everything latency-related is a MODELED proxy (queue wait + modeled prefill
time). It is not a GPU measurement and must be described that way until it is
calibrated against a running inference server.
"""
from __future__ import annotations

import numpy as np

from .simulator import SimResult


def summarize(res: SimResult, warmup_s: float, horizon_s: float) -> dict[str, float]:
    # Measure only requests that arrive in steady state; later ones still load the system.
    rs = [r for r in res.records if warmup_s <= r.arrival <= horizon_s]
    if not rs:
        raise ValueError("no measured requests; check warmup/horizon")

    ttft = np.array([r.ttft for r in rs])
    prompt = sum(r.prompt_tokens for r in rs)
    hit = sum(r.hit_tokens for r in rs)
    routing = sum(r.routing_tokens for r in rs)
    evicted = sum(r.evicted_tokens for r in rs)
    cold = sum(r.cold_tokens for r in rs)
    n = len(rs)
    service = np.array([r.service_s for r in rs])
    n_workers = len(res.workers)
    busy = np.array([w.busy_s for w in res.workers])
    window_end = max(r.finish for r in rs)

    return {
        "n_requests": n,
        # share of all prompt tokens served from cache
        "hit_rate": hit / prompt,
        # of tokens that COULD have been reused (cached somewhere / seen before), how many were
        "reusable_hit_rate": hit / max(1, hit + routing + evicted),
        "recomputed_prefix_tokens_per_req": (routing + evicted) / n,
        "recomputed_routing_tokens_per_req": routing / n,
        "recomputed_evicted_tokens_per_req": evicted / n,
        "cold_tokens_per_req": cold / n,
        "ttft_mean_s": float(ttft.mean()),
        "ttft_p50_s": float(np.percentile(ttft, 50)),
        "ttft_p95_s": float(np.percentile(ttft, 95)),
        "ttft_p99_s": float(np.percentile(ttft, 99)),
        # completed requests / wall time from steady-state start to last measured completion
        "throughput_rps": n / max(1e-9, window_end - warmup_s),
        # what the cluster could sustain at saturation given this policy's hit pattern
        "capacity_rps_est": n_workers / float(service.mean()),
        # load imbalance across workers: stdev/mean of busy time (0 = perfectly even)
        "busy_cv": float(busy.std() / max(1e-9, busy.mean())),
    }
