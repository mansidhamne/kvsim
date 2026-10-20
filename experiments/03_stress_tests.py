"""Experiment 3: where does cache-aware routing stop helping (or start hurting)?

Four probes, each at a load where the baselines are stressed but not collapsed:

  (a) affinity weight alpha under a normal vs a HOT-PREFIX workload  -> herding
  (b) per-worker cache capacity                                      -> thrashing / saturation
  (c) conversation length (reuse available)                          -> nothing to reuse
  (d) is the router's view of *pending* (queued) prefixes worth it?

    python experiments/03_stress_tests.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORS, MODELED, RESULTS, plt  # noqa: E402

from kvsim import ClusterConfig, WorkloadConfig  # noqa: E402
from kvsim.runner import Job, PolicySpec, run_jobs  # noqa: E402

SEEDS = [0, 1, 2]
N = 125                                    # concurrency for (a), (b), (d)
RR = PolicySpec("round-robin", "round_robin")
LL = PolicySpec("least-loaded", "least_loaded")


def CA(label="cache-aware", **kw):
    return PolicySpec(label, "cache_aware", tuple(sorted(kw.items())))


def collect(jobs):
    """-> {tag: {policy: {metric: mean over seeds}}} plus per-seed p99 for bands."""
    by_tag: dict = {}
    for (tag, seed), per_policy in run_jobs(jobs):
        for pol, summ in per_policy.items():
            for k, v in summ.items():
                by_tag.setdefault(tag, {}).setdefault(pol, {}).setdefault(k, []).append(v)
    return by_tag


def line(ax, xs, ys_per_seed, color, label):
    arr = np.array(ys_per_seed)           # (n_x, n_seeds)
    ax.plot(xs, arr.mean(1), "-o", ms=4, color=color, label=label)
    ax.fill_between(xs, arr.min(1), arr.max(1), color=color, alpha=0.13, lw=0)


def main():
    out: dict = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

    # ---- (a) alpha sweep, normal vs hot-prefix ------------------------------------
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    normal = WorkloadConfig(active_convs=N)
    hot = replace(normal, popular_prob=0.9, n_popular_boards=2, zipf_s=1.5,
                  images_min=12, images_max=16)
    pols = (LL,) + tuple(CA(f"a={a}", alpha=a) for a in alphas)
    jobs = [Job(replace(w, seed=s), policies=pols, tag=(name, s))
            for name, w in (("normal", normal), ("hot-prefix", hot)) for s in SEEDS]
    res_a = {}
    for (tag, _s), per_policy in run_jobs(jobs):
        res_a.setdefault(tag, []).append(per_policy)
    ax = axes[0, 0]
    for name, color in (("normal", "#e60023"), ("hot-prefix", "#7b1fa2")):
        runs = res_a[name]
        ys = [[r[f"a={a}"]["ttft_p99_s"] for r in runs] for a in alphas]
        line(ax, range(len(alphas)), ys, color, f"cache-aware, {name} workload")
        ll = np.mean([r["least-loaded"]["ttft_p99_s"] for r in runs])
        ax.axhline(ll, color=color, ls="--", lw=1.3, label=f"least-loaded, {name} workload")
        out[f"alpha_{name}"] = {str(a): float(np.mean(y)) for a, y in zip(alphas, ys)}
        out[f"alpha_{name}_least_loaded"] = float(ll)
    ax.set_xticks(range(len(alphas)), [str(a) for a in alphas])
    ax.set_xlabel("alpha: weight on cache affinity (1 = estimated-TTFT routing, large = sticky)")
    ax.set_ylabel("p99 TTFT proxy (s)")
    ax.set_yscale("log")
    ax.set_title("(a) too much stickiness herds requests onto hot GPUs", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # ---- (b) cache capacity ----------------------------------------------------------
    caps = [20_000, 50_000, 100_000, 150_000, 300_000, 600_000, 1_200_000]
    jobs = [Job(WorkloadConfig(active_convs=N, seed=s), ClusterConfig(cache_capacity_tokens=c),
                policies=(LL, CA()), tag=(c, s)) for c in caps for s in SEEDS]
    by = collect(jobs)
    ax = axes[0, 1]
    for pol, color in (("least-loaded", COLORS["least-loaded"]), ("cache-aware", COLORS["cache-aware"])):
        line(ax, range(len(caps)), [by[c][pol]["ttft_p99_s"] for c in caps], color, pol)
    ax.set_xticks(range(len(caps)), [f"{c // 1000}k" for c in caps])
    ax.set_xlabel("KV cache budget per worker (tokens)")
    ax.set_ylabel("p99 TTFT proxy (s)")
    ax.set_yscale("log")
    ax.set_title("(b) the gain narrows at both extremes of cache size", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    out["capacity"] = {str(c): {p: float(np.mean(by[c][p]["ttft_p99_s"])) for p in by[c]}
                       for c in caps}
    out["capacity_hit"] = {str(c): {p: float(np.mean(by[c][p]["hit_rate"])) for p in by[c]}
                           for c in caps}

    # ---- (c) conversation length at fixed request rate -------------------------------
    R = 13.0                                   # requests/s, ~ N=125 at the default length
    turns = [1, 2, 3, 5, 8, 12]
    jobs = [Job(WorkloadConfig(seed=s, turns_min=t, turns_max=t, conv_start_rate_override=R / t),
                policies=(LL, CA()), tag=(t, s)) for t in turns for s in SEEDS]
    by = collect(jobs)
    ax = axes[1, 0]
    for pol, color in (("least-loaded", COLORS["least-loaded"]), ("cache-aware", COLORS["cache-aware"])):
        line(ax, range(len(turns)), [by[t][pol]["ttft_mean_s"] for t in turns], color, pol)
    ax.set_xticks(range(len(turns)), [str(t) for t in turns])
    ax.set_xlabel(f"turns per conversation (fixed {R:.0f} req/s offered)")
    ax.set_ylabel("mean TTFT proxy (s)")
    ax.set_yscale("log")
    ax.set_title("(c) single-turn traffic leaves little to reuse", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    out["turns"] = {str(t): {p: float(np.mean(by[t][p]["ttft_mean_s"])) for p in by[t]}
                    for t in turns}
    out["turns_hit"] = {str(t): {p: float(np.mean(by[t][p]["hit_rate"])) for p in by[t]}
                        for t in turns}

    # ---- (d) pending-prefix awareness ------------------------------------------------
    concs = [75, 100, 125, 150, 175, 200]
    pols = (LL, CA("cache-aware (sees pending)"),
            CA("cache-aware (resident only)", pending_aware=False))
    jobs = [Job(WorkloadConfig(active_convs=c, seed=s), policies=pols, tag=(c, s))
            for c in concs for s in SEEDS]
    by = collect(jobs)
    ax = axes[1, 1]
    for pol, color in (("least-loaded", COLORS["least-loaded"]),
                       ("cache-aware (resident only)", "#f9a825"),
                       ("cache-aware (sees pending)", COLORS["cache-aware"])):
        line(ax, concs, [by[c][pol]["ttft_p99_s"] for c in concs], color, pol)
    ax.set_xlabel("concurrent conversations")
    ax.set_ylabel("p99 TTFT proxy (s)")
    ax.set_yscale("log")
    ax.set_title("(d) seeing queued prefixes only matters near saturation", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    out["pending"] = {str(c): {p: float(np.mean(by[c][p]["ttft_p99_s"])) for p in by[c]}
                      for c in concs}

    fig.suptitle("Where cache-aware routing helps, and where it stops helping\n" + MODELED,
                 fontsize=10.5)
    fig.tight_layout()
    path = RESULTS / "stress_tests.png"
    fig.savefig(path)
    (RESULTS / "stress_tests.json").write_text(json.dumps(out, indent=1))
    print("wrote", path)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
