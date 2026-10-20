"""Calibration 5 (simulator only): is "session-affinity beats cache-aware" a fluke of one
trace, or systematic under the engine's whole-prompt cache?

Real runs are ~10 minutes each, so we cannot repeat them for many seeds. But the simulator
reproduced the real hit rates within ~0.03 (see 04), so we use it to ask the follow-up:
sweep per-worker cache budget and seeds, same workload as 04, `sequence_lru` cache.

    python calibration/05_sim_regimes.py            # ~1 min, no GPU / Ollama needed
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from harness import RESULTS  # noqa: E402
from _common import plt  # noqa: E402

from kvsim import (ClusterConfig, CostModel, WorkloadConfig, generate_requests,  # noqa: E402
                   make_router, simulate, summarize)

BS = 32
POL = {"round_robin": {}, "least_loaded": {}, "session_affinity": {}, "cache_aware": {"pending_aware": False}}
COLORS = {"round_robin": "#9aa0a6", "least_loaded": "#4285f4", "session_affinity": "#2e7d32",
          "cache_aware": "#e60023"}
MIBS = [20, 40, 80, 160, 320, 640]
SEEDS = list(range(8))


def job(args):
    mib, seed = args
    cal = json.loads((RESULTS / "calibration.json").read_text())
    real = json.loads((RESULTS / "routing_real.json").read_text())
    base = cal["cost_model"]
    cost = CostModel(block_size=BS, prefill_tokens_per_s=base["prefill_tokens_per_s"] / 3,   # 3-way GPU sharing, as in 04
                     fixed_overhead_s=base["fixed_overhead_s"], image_cost_multiplier=1.0)
    wl = WorkloadConfig(seed=seed, horizon_s=90.0, warmup_s=0.0, active_convs=12, think_mean_s=10.0,
                        think_min_s=0.5, turns_min=3, turns_max=5, system_tokens=192, images_min=1,
                        images_max=3, tokens_per_image=96, user_tokens=(20, 60),
                        response_tokens=(40, 120), n_popular_boards=6, popular_prob=0.3)
    reqs = generate_requests(wl, cost)
    cap_tokens = mib * 1024 / cal["cache"]["kib_per_token"]
    cl = ClusterConfig(n_workers=3, cache_capacity_tokens=int(cap_tokens), cache_kind="sequence_lru")
    out = {}
    for name, kw in POL.items():
        res = simulate(reqs, make_router(name, np.random.default_rng(seed + 12345), **kw), cl, cost)
        s = summarize(res, 0.0, 1e12)
        out[name] = (s["hit_rate"], s["ttft_p95_s"], s["ttft_mean_s"])
    return mib, seed, out


def main():
    jobs = [(m, s) for m in MIBS for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(job, jobs))
    data = {n: {m: [] for m in MIBS} for n in POL}
    for mib, _s, out in rows:
        for n, v in out.items():
            data[n][mib].append(v)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for n, c in COLORS.items():
        for a, k, lab in ((ax[0], 0, "token hit rate"), (ax[1], 1, "p95 latency (s)")):
            arr = np.array([[v[k] for v in data[n][m]] for m in MIBS])
            a.plot(range(len(MIBS)), arr.mean(1), "-o", ms=4, color=c, label=n.replace("_", "-"))
            a.fill_between(range(len(MIBS)), arr.min(1), arr.max(1), color=c, alpha=0.12, lw=0)
            a.set_xticks(range(len(MIBS)), [str(m) for m in MIBS])
            a.set(xlabel="per-worker prompt-cache budget (MiB)", ylabel=lab)
    ax[1].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Simulator, engine-style whole-prompt cache (validated against real hit rates in 04): "
                 f"{len(SEEDS)} seeds, band = min..max", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "routing_sim_regimes.png")

    print(f"{'MiB':>5} " + " ".join(f"{n:>17}" for n in POL))
    summary = {}
    for m in MIBS:
        cells = []
        for n in POL:
            h = np.mean([v[0] for v in data[n][m]])
            p = np.mean([v[1] for v in data[n][m]])
            cells.append(f"{h:5.2f} / p95 {p:6.2f}s")
            summary.setdefault(str(m), {})[n] = {"hit_rate": float(h), "p95_s": float(p)}
        print(f"{m:5d} " + " ".join(f"{c:>17}" for c in cells))
    # how often is cache-aware's hit rate above session-affinity's, per budget?
    print("\nfraction of seeds where cache-aware hit rate > session-affinity:")
    for m in MIBS:
        w = np.mean([ca[0] > sa[0] for ca, sa in zip(data["cache_aware"][m], data["session_affinity"][m])])
        print(f"  {m:4d} MiB: {w:4.0%}")
    (RESULTS / "routing_sim_regimes.json").write_text(json.dumps(summary, indent=1))
    print("wrote", RESULTS / "routing_sim_regimes.png")


if __name__ == "__main__":
    main()
