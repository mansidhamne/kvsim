"""Think time between turns and conversation burstiness, from Alibaba ServeGen (Apache-2.0).

Source: github.com/alibaba/ServeGen, data/conversations/conversations_hashed.json: 1,616 real
multi-turn conversations with a timestamp per turn and input/output token counts (content
replaced by hashes).

Think time = gap between two turns' send times minus the time it took to generate the previous
reply. ServeGen doesn't record generation time, so it is estimated as output tokens / `gen_tps`
(an ASSUMPTION, default 30 tokens/s); gaps shorter than that are clipped at `min_think_s`.

The derived distribution (quantiles only, no conversation data) is committed in
bench/traces/data/servegen_think.json so experiments don't need the 50 MB file.

    python -m bench.traces.servegen /path/to/conversations_hashed.json   # regenerate the file
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "servegen_think.json"
SOURCE = "alibaba/ServeGen data/conversations/conversations_hashed.json (Apache-2.0)"


def derive(path: str | Path, gen_tps: float = 30.0, min_think_s: float = 1.0) -> dict:
    conv = json.loads(Path(path).read_text())
    think, gaps, turns, starts = [], [], [], []
    for c in conv:
        c = sorted(c, key=lambda t: t["turn"])
        turns.append(len(c))
        starts.append(c[0]["timestamp"])
        for a, b in zip(c, c[1:]):
            g = b["timestamp"] - a["timestamp"]
            gaps.append(g)
            think.append(max(min_think_s, g - a["output_token_count"] / gen_tps))
    starts = np.sort(np.array(starts))
    ia = np.diff(starts)
    q = np.linspace(0, 1, 101)
    return {
        "source": SOURCE, "gen_tps_assumed": gen_tps, "min_think_s": min_think_s,
        "conversations": len(conv), "follow_ups": len(think),
        "think_s_quantiles": [round(float(x), 3) for x in np.quantile(think, q)],
        "send_gap_s_quantiles": [round(float(x), 3) for x in np.quantile(gaps, q)],
        "turns_per_conversation": {"mean": float(np.mean(turns)), "p50": float(np.median(turns)),
                                   "p90": float(np.percentile(turns, 90))},
        # burstiness of conversation starts: coefficient of variation of inter-arrival gaps
        # (1.0 = Poisson). ServeGen selected these conversations, so this is indicative only.
        "start_interarrival_cv": float(np.std(ia) / np.mean(ia)) if len(ia) > 1 else None,
    }


def load() -> dict:
    return json.loads(DATA.read_text())


class ThinkTime:
    """Sample think times from the committed ServeGen quantiles (inverse CDF, linear between
    quantiles), multiplied by `scale` (< 1 compresses time so a run fits a Kaggle session;
    the cache-size sweep then covers the eviction pressure that compression removes)."""

    def __init__(self, scale: float = 1.0, data: dict | None = None):
        d = data or load()
        self.q = np.array(d["think_s_quantiles"], dtype=float)
        self.scale = scale

    def sample(self, rng: np.random.Generator, n: int | None = None):
        u = rng.random(n)
        x = np.interp(u * (len(self.q) - 1), np.arange(len(self.q)), self.q) * self.scale
        return x


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gen-tps", type=float, default=30.0)
    a = ap.parse_args()
    d = derive(a.path, a.gen_tps)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(d, indent=1))
    qs = d["think_s_quantiles"]
    print(f"{d['follow_ups']} follow-ups from {d['conversations']} conversations; think time "
          f"p10 {qs[10]:.0f}s p50 {qs[50]:.0f}s p90 {qs[90]:.0f}s; start CV {d['start_interarrival_cv']:.2f}")


if __name__ == "__main__":
    main()
