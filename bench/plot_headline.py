"""Headline figure: change vs round-robin per workload, pooled over all seeds and sessions.

    python -m bench.plot_headline   # reads results/phase3b/pooled_paired.json -> results/phase3b/headline.png
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/phase3b/pooled_paired.json"
OUT = ROOT / "results/phase3b/headline.png"

GROUPS = [("VisionArena ample cache (A full+25%, B 50%)", "Real VLM chats\nample KV cache"),
          ("VisionArena scarce cache (B2+V1)", "Real VLM chats\nscarce KV cache"),
          ("Mooncake (D+V2)", "Production prefix trace\n(Mooncake)")]
METRICS = [("ttft_p99_s", "TTFT p99"), ("ttft_p50_s", "TTFT p50"), ("goodput_frac", "goodput"),
           ("hit_rate", "prompt tokens\nfrom cache")]
POLICIES = [("session_affinity vs round_robin", "session affinity", "#9aa5b1"),
            ("cache_aware vs round_robin", "cache-aware", "#2a78d6")]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(SRC.read_text())
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(13, 4.2), sharey=True)
    width = 0.38
    for ax, (key, title) in zip(axes, GROUPS):
        g = data[key]
        for j, (pk, label, color) in enumerate(POLICIES):
            xs, med, lo, hi, notes = [], [], [], [], []
            for i, (m, _) in enumerate(METRICS):
                d = g[pk][m]
                xs.append(i + (j - 0.5) * width)
                med.append(100 * d["median"])
                lo.append(100 * (d["median"] - d["min"]))
                hi.append(100 * (d["max"] - d["median"]))
                notes.append(f"{d['better']}/{d['n']}")
            ax.bar(xs, med, width, yerr=[lo, hi], capsize=2, color=color, label=label)
            for x, y, n in zip(xs, med, notes):
                ax.text(x, (y + 4) if y >= 0 else (y - 4), n, ha="center",
                        va="bottom" if y >= 0 else "top", fontsize=6.5, color="#333")
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(range(len(METRICS)), [m[1] for m in METRICS], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-75, 160)
    axes[0].set_ylabel("change vs round-robin (%)\nmedian over seeds, bars = min-max")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("2x T4, vLLM 0.32, Qwen2-VL-2B. Lower TTFT is better; higher goodput / reuse is better. "
                 "Labels: seeds where the policy beat round-robin.", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
