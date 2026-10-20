"""Experiment 2: GPU A holds the prefix but has a queue; GPU B is idle and cold.
Wait for A, or recompute everything on B?

    wait on A:        TTFT = q_A + cost(new tokens)
    recompute on B:   TTFT =       cost(prefix + new tokens)
    break-even:       q* = cost(prefix)            (the time it would take to re-prefill the prefix)

The router (alpha = 1) must flip exactly at q*. We check that by calling the real
router object, not by re-deriving the formula.

    python experiments/02_breakeven.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import MODELED, RESULTS, plt  # noqa: E402

from kvsim import CacheAware, CostModel  # noqa: E402
from kvsim.scenarios import breakeven_queue_s, wait_or_recompute  # noqa: E402

COST = CostModel()
NEW_BLOCKS = 6                               # the new user message + last reply, ~400 tokens
PREFIXES_TOKENS = [2_000, 8_000, 32_000]     # a short chat, a typical image-heavy chat, a huge one
COLORS = ["#fbbc04", "#e60023", "#7b1fa2"]


def main():
    router = CacheAware(np.random.default_rng(0), alpha=1.0)
    qs = np.linspace(0, 2.0, 201)
    fig, ax = plt.subplots(figsize=(8, 5))
    # Waiting on A costs q + cost(new tokens): independent of how long the cached prefix is.
    ax.plot(qs, qs + COST.prefill_time(NEW_BLOCKS), color="black", lw=2.2,
            label="wait for GPU A (cached prefix)")
    for tokens, color in zip(PREFIXES_TOKENS, COLORS):
        pb = tokens // COST.block_size
        recompute = COST.prefill_time(pb + NEW_BLOCKS)
        q_star = breakeven_queue_s(COST, pb, NEW_BLOCKS)
        picks = np.array([wait_or_recompute(router, COST, pb, NEW_BLOCKS, q) for q in qs])

        # the router must wait exactly while q < q*
        assert all(picks[qs < q_star - 1e-6] == 0) and all(picks[qs > q_star + 1e-6] == 1)

        ax.axhline(recompute, color=color, ls="--", lw=1.8,
                   label=f"recompute on idle GPU B ({tokens // 1000}k-token prefix)")
        ax.plot([q_star], [recompute], "o", color=color, ms=8)
        ax.annotate(f"break-even q* = {q_star:.2f}s", (q_star, recompute),
                    textcoords="offset points", xytext=(8, 6 if tokens < 4_000 else -14),
                    color=color, fontsize=9)
        print(f"prefix {tokens:>6} tok: break-even queue wait q* = {q_star:.3f}s  "
              f"(router flips there; recompute-on-B TTFT = {recompute:.3f}s)")

    ax.set_xlabel("queue wait on GPU A, the one that holds the prefix (s)")
    ax.set_ylabel("modeled TTFT (s)")
    ax.set_ylim(0, 2.2)
    ax.set_title("Wait for the cached GPU, or recompute on the idle one?\n" + MODELED,
                 fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    out = RESULTS / "breakeven.png"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
