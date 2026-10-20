"""Experiment 1: how do the three policies behave as concurrency grows?

Same trace for all policies at each (concurrency, seed). 3 seeds -> line = mean,
band = min..max across seeds.

    python experiments/01_concurrency_sweep.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORS, MODELED, RESULTS, plt  # noqa: E402

from kvsim import WorkloadConfig  # noqa: E402
from kvsim.runner import DEFAULT_POLICIES, Job, PolicySpec, run_jobs  # noqa: E402

CONCURRENCY = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250]
SEEDS = [0, 1, 2]
# the three required policies + one extra baseline (drawn dashed) for the "why not sticky sessions?" question
POLICIES = DEFAULT_POLICIES + (PolicySpec("session-affinity", "session_affinity"),)
EXTRA = {"session-affinity": "#2e7d32"}


def main():
    jobs = [Job(WorkloadConfig(active_convs=n, seed=s), policies=POLICIES, tag=(n, s))
            for n in CONCURRENCY for s in SEEDS]
    results = dict(run_jobs(jobs))

    # data[policy][metric][concurrency] -> list over seeds
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (n, _s), per_policy in results.items():
        for pol, summ in per_policy.items():
            for k, v in summ.items():
                data[pol][k][n].append(v)

    (RESULTS / "sweep.json").write_text(json.dumps(
        {pol: {m: {str(n): v for n, v in d.items()} for m, d in ms.items()}
         for pol, ms in data.items()}, indent=1))

    panels = [
        ("ttft_mean_s", "mean TTFT proxy (s)", True),
        ("ttft_p99_s", "p99 TTFT proxy (s)", True),
        ("hit_rate", "cache hit rate (share of prompt tokens)", False),
        ("capacity_rps_est", "est. sustainable throughput (req/s)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (metric, label, logy) in zip(axes.ravel(), panels):
        for pol, color in {**COLORS, **EXTRA}.items():
            xs = np.array(CONCURRENCY)
            arr = np.array([data[pol][metric][n] for n in CONCURRENCY])   # (n_conc, n_seeds)
            extra = pol in EXTRA
            ax.plot(xs, arr.mean(1), "--s" if extra else "-o", ms=3.5, lw=1.4, color=color,
                    label=pol + (" (extra baseline)" if extra else ""))
            ax.fill_between(xs, arr.min(1), arr.max(1), color=color, alpha=0.15, lw=0)
        ax.set_ylabel(label)
        if logy:
            ax.set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("concurrent conversations")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Routing policies vs. concurrency  (3 simulated workers, same trace per point)\n"
                 + MODELED, fontsize=10.5)
    fig.tight_layout()
    out = RESULTS / "concurrency_sweep.png"
    fig.savefig(out)
    print("wrote", out)

    # human-readable table at a few points
    print(f"\n{'conc':>5} {'policy':13s} {'hit':>5} {'recomp/req':>10} {'mean':>8} {'p95':>8} "
          f"{'p99':>8} {'cap rps':>8}")
    for n in CONCURRENCY:
        for pol in {**COLORS, **EXTRA}:
            g = lambda m: float(np.mean(data[pol][m][n]))  # noqa: E731
            print(f"{n:5d} {pol:13s} {g('hit_rate'):5.2f} "
                  f"{g('recomputed_prefix_tokens_per_req'):10.0f} {g('ttft_mean_s'):8.3f} "
                  f"{g('ttft_p95_s'):8.3f} {g('ttft_p99_s'):8.3f} {g('capacity_rps_est'):8.1f}")


if __name__ == "__main__":
    main()
