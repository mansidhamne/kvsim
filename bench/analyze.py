"""Per-request JSONL (load generator) + router decision log -> per-policy summary and figure.

Requests that were SCHEDULED during the warm-up window are excluded from every metric (caches
start empty after each reset). All token counts are the engine's own (usage report), never the
router's estimates, except where a field says "predicted".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def bootstrap_ci(values, stat, n: int = 1000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of `stat(values)` (resampling requests). Tells you whether two
    policies' numbers differ by more than run-to-run noise would."""
    v = np.asarray(values, dtype=float)
    if len(v) < 5:
        return None
    rng = np.random.default_rng(seed)
    sims = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    lo, hi = np.percentile(sims, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [float(lo), float(hi)]


def summarize_cell(records: list[dict], decisions: list[dict], warmup_s: float,
                   slo_ttft_s: float, slo_itl_s: float, block_size: int = 16,
                   max_tokens: dict | None = None) -> dict:
    """records: loadgen Record dicts; decisions: router log rows; max_tokens: (conv, turn) -> n."""
    # Warm-up filter by CONVERSATION start (same conversations measured for every policy; with
    # think time after the reply, follow-up send times depend on the policy's speed).
    # (records from runs 1-2 have no conv_start_s; they used send time)
    meas = [r for r in records
            if (r["conv_start_s"] if "conv_start_s" in r else r["scheduled_s"]) >= warmup_s]
    ok = [r for r in meas if r.get("error") is None and r.get("ttft_s") is not None]
    ttft = [r["ttft_s"] for r in ok]
    itl_all = [g for r in ok for g in r.get("itl_s") or []]
    itl_mean = {id(r): (float(np.mean(r["itl_s"])) if r.get("itl_s") else 0.0) for r in ok}
    good = [r for r in ok if r["ttft_s"] <= slo_ttft_s and itl_mean[id(r)] <= slo_itl_s]
    prompt = sum(r.get("prompt_tokens") or 0 for r in ok)
    cached = sum(r.get("cached_tokens") or 0 for r in ok)
    span = (max(r["sent_s"] + (r.get("e2e_s") or 0) for r in ok) - min(r["sent_s"] for r in ok)) if ok else 0
    per_worker: dict[str, int] = {}
    for r in ok:
        per_worker[r.get("worker") or r["target"]] = per_worker.get(r.get("worker") or r["target"], 0) + 1
    shares = list(per_worker.values())

    # router-side: decisions joined to measured requests by (conv, turn)
    keyset = {(str(r["conv_id"]), str(r["turn"])) for r in meas}
    dec = [d for d in decisions if (str(d.get("conv")), str(d.get("turn"))) in keyset]
    dms = [d["decision_ms"] for d in dec if "decision_ms" in d]
    pred = [(d.get("predicted_resident_tokens", 0), d.get("actual_cached_tokens"))
            for d in dec if d.get("actual_cached_tokens") is not None]
    # share of the prompt predicted to be cached vs the engine's answer (robust to the router
    # counting words while the engine counts model tokens, ~2% apart)
    # Both over the ENGINE's prompt length: dividing the prediction by the router's own estimate
    # made very short prompts look wrong (the chat template adds ~17 tokens the router doesn't
    # count: Mooncake block D showed 87% vs 99% within 5 points).
    share_err = [abs(d.get("predicted_resident_tokens", 0) - d["actual_cached_tokens"])
                 / max(1, d.get("prompt_tokens") or 1)
                 for d in dec if d.get("actual_cached_tokens") is not None and d.get("prompt_tokens")]
    # Is the router's prefill estimate realistic? Compare it to the measured TTFT on requests
    # that found the chosen worker with no prefill backlog (TTFT is then ~ prefill time) and
    # were mostly NOT cached: that's where the expensive terms (images, attention) dominate.
    # Cache hits are cheap and easy to estimate, so including them would hide a bad model.
    est_ratio, est_ratio_hits = [], []
    for d in dec:
        sc = d.get("scores") or []
        i = int(d["worker"][1:]) if str(d.get("worker", "")).startswith("w") else None
        if (i is None or i >= len(sc) or sc[i]["backlog_s"] >= 0.05 or not d.get("router_ttft_s")
                or d["router_ttft_s"] <= 0.05 or d.get("actual_cached_tokens") is None):
            continue
        ratio = d["est_prefill_s"] / d["router_ttft_s"]
        cached_share = d["actual_cached_tokens"] / max(1, d.get("prompt_tokens") or 1)
        (est_ratio if cached_share < 0.5 else est_ratio_hits).append(ratio)
    routing_miss = []
    for d in dec:
        hits = [s["hit_blocks"] for s in d.get("scores") or []]
        if hits:
            chosen = next((s["hit_blocks"] for i, s in enumerate(d["scores"]) if f"w{i}" == d["worker"]), 0)
            routing_miss.append((max(hits) - chosen) * block_size)

    wrong_len = 0
    if max_tokens:
        wrong_len = sum(1 for r in ok if r.get("completion_tokens") is not None
                        and r["completion_tokens"] != max_tokens.get((r["conv_id"], r["turn"])))
    return {
        "requests": len(meas), "ok": len(ok), "errors": len(meas) - len(ok),
        "missing_ttft": sum(1 for r in meas if r.get("error") is None and r.get("ttft_s") is None),
        "hit_rate": cached / prompt if prompt else None,
        "prompt_tokens_mean": prompt / len(ok) if ok else None,
        "recomputed_tokens_per_req": (prompt - cached) / len(ok) if ok else None,
        "ttft_p50_s": _pct(ttft, 50), "ttft_p90_s": _pct(ttft, 90), "ttft_p99_s": _pct(ttft, 99),
        "ttft_mean_s": float(np.mean(ttft)) if ttft else None,
        "itl_p50_s": _pct(itl_all, 50), "itl_p99_s": _pct(itl_all, 99),
        "e2e_p50_s": _pct([r["e2e_s"] for r in ok if r.get("e2e_s")], 50),
        "goodput_frac": len(good) / len(meas) if meas else None,
        "goodput_rps": len(good) / span if span else None,
        "throughput_rps": len(ok) / span if span else None,
        "late_frac": sum(1 for r in ok if r.get("late_s", 0) > 1.0) / len(ok) if ok else None,
        "per_worker": dict(sorted(per_worker.items())),
        "imbalance": (max(shares) / (sum(shares) / len(shares))) if shares else None,
        "decision_ms_p50": _pct(dms, 50), "decision_ms_p99": _pct(dms, 99),
        "prediction_exact_frac": (sum(abs(p - a) <= block_size for p, a in pred) / len(pred)) if pred else None,
        "prediction_mae_tokens": float(np.mean([abs(p - a) for p, a in pred])) if pred else None,
        "prediction_bias_tokens": float(np.mean([p - a for p, a in pred])) if pred else None,
        "prediction_share_within5pp": (float(np.mean([e <= 0.05 for e in share_err])) if share_err else None),
        "estimate_ratio_median": float(np.median(est_ratio)) if est_ratio else None,
        "estimate_ratio_n": len(est_ratio),
        "estimate_ratio_hits_median": float(np.median(est_ratio_hits)) if est_ratio_hits else None,
        "ttft_p99_ci": bootstrap_ci(ttft, lambda x: np.percentile(x, 99)),
        "ttft_p50_ci": bootstrap_ci(ttft, lambda x: np.percentile(x, 50)),
        "goodput_ci": bootstrap_ci([1.0 if r in good else 0.0 for r in ok] +
                                   [0.0] * (len(meas) - len(ok)), np.mean),
        "routing_miss_tokens_per_req": float(np.mean(routing_miss)) if routing_miss else None,
        "wrong_output_len": wrong_len,
        # sanitized records (deploy/sanitize_results.py) have no output_text: skip, don't fake ""
        "sample_outputs": [r["output_text"][:120] for r in ok[:3] if "output_text" in r],
    }


def load_jsonl(path: str | Path) -> tuple[dict | None, list[dict]]:
    """Reads `x.jsonl` or `x.jsonl.gz` (archived runs are gzipped)."""
    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    if path.suffix == ".gz":
        import gzip
        text = gzip.open(path, "rt").read()
    else:
        text = path.read_text()
    meta, rows = None, []
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "meta" in obj and len(obj) == 1:
            meta = obj["meta"]
        else:
            rows.append(obj)
    return meta, rows


LABEL = {"round_robin": "round-robin", "least_requests": "least-requests", "least_loaded": "least-loaded",
         "session_affinity": "session-affinity", "cache_aware": "cache-aware"}


def plot_cells(cells: list[dict], out_png: Path, title: str = "") -> None:
    """cells: [{"policy", "active_convs", "summary"}]. One row of bar charts per load level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loads = sorted({c["active_convs"] for c in cells})
    metrics = [("hit_rate", "prompt tokens from cache", 1), ("ttft_p50_s", "TTFT p50 (s)", 1),
               ("ttft_p99_s", "TTFT p99 (s)", 1), ("itl_p99_s", "ITL p99 (s)", 1),
               ("goodput_frac", "goodput (share within SLO)", 1)]
    fig, axes = plt.subplots(len(loads), len(metrics), figsize=(3.4 * len(metrics), 3.0 * len(loads)),
                             squeeze=False)
    for i, n in enumerate(loads):
        row = [c for c in cells if c["active_convs"] == n]
        names = [LABEL.get(c["policy"], c["policy"]) for c in row]
        for j, (key, lab, _) in enumerate(metrics):
            ax = axes[i][j]
            vals = [c["summary"].get(key) or 0 for c in row]
            ax.bar(range(len(row)), vals, color=["#9aa5b1"] * (len(row) - 1) + ["#2a78d6"]
                   if row and row[-1]["policy"] == "cache_aware" else "#9aa5b1")
            ax.set_xticks(range(len(row)), names, rotation=35, ha="right", fontsize=7)
            ax.set_title(f"{lab}\n{n} active conversations", fontsize=8)
            for k, v in enumerate(vals):
                ax.text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)


# ------------------------------------------------------------------------------- phase 3b

METRICS = ("hit_rate", "ttft_p50_s", "ttft_p99_s", "itl_p99_s", "goodput_frac",
           "recomputed_tokens_per_req")


def aggregate(cells: list[dict]) -> list[dict]:
    """Per (workload, cache, load, policy): median and min-max over seeds of each metric, and
    hit rate as a share of the workload's reuse ceiling."""
    from collections import defaultdict
    g = defaultdict(list)
    for c in cells:
        g[(c["workload"], c["cache"], c["load"], c["policy"])].append(c)
    out = []
    for (w, cache, load, pol), cs in sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = {"workload": w, "cache": cache, "load": load, "policy": pol, "seeds": len(cs)}
        for m in METRICS:
            v = [c["summary"].get(m) for c in cs if c["summary"].get(m) is not None]
            row[m] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))} if v else None
        ceil = [c.get("reuse_ceiling") for c in cs if c.get("reuse_ceiling")]
        hits = [c["summary"].get("hit_rate") for c in cs if c["summary"].get("hit_rate") is not None]
        if ceil and hits:
            row["share_of_ceiling"] = float(np.median([h / x for h, x in zip(hits, ceil)]))
        out.append(row)
    return out


def paired(cells: list[dict], a: str, b: str, metric: str) -> list[dict]:
    """Relative change of `metric` for policy `a` vs policy `b`, computed WITHIN each seed (same
    trace, same session), then summarised across seeds: the comparison that cancels run-to-run
    noise. Negative = a is lower than b."""
    from collections import defaultdict
    by = defaultdict(dict)
    for c in cells:
        by[(c["workload"], c["cache"], c["load"], c["seed"])][c["policy"]] = c["summary"].get(metric)
    groups = defaultdict(list)
    for (w, cache, load, seed), pols in by.items():
        if pols.get(a) is not None and pols.get(b):
            groups[(w, cache, load)].append(pols[a] / pols[b] - 1.0)
    return [{"workload": w, "cache": cache, "load": load, "a": a, "b": b, "metric": metric,
             "seeds": len(v), "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)),
             "same_sign_all_seeds": bool(all(x < 0 for x in v) or all(x > 0 for x in v))}
            for (w, cache, load), v in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]])]


def plot_aggregate(rows: list[dict], out_png: Path, title: str = "") -> None:
    """One row of panels per (workload, cache, load): median over seeds with min-max bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted({(r["workload"], r["cache"], r["load"]) for r in rows}, key=lambda k: [str(x) for x in k])
    metrics = [("hit_rate", "prompt tokens from cache"), ("ttft_p99_s", "TTFT p99 (s)"),
               ("itl_p99_s", "ITL p99 (s)"), ("goodput_frac", "goodput")]
    fig, axes = plt.subplots(len(keys), len(metrics), figsize=(3.4 * len(metrics), 2.8 * len(keys)),
                             squeeze=False)
    for i, k in enumerate(keys):
        rs = [r for r in rows if (r["workload"], r["cache"], r["load"]) == k]
        names = [LABEL.get(r["policy"], r["policy"]) for r in rs]
        for j, (m, lab) in enumerate(metrics):
            ax = axes[i][j]
            med = [(r[m] or {}).get("median", 0) for r in rs]
            lo = [med[n] - (rs[n][m] or {}).get("min", 0) for n in range(len(rs))]
            hi = [(rs[n][m] or {}).get("max", 0) - med[n] for n in range(len(rs))]
            ax.bar(range(len(rs)), med, yerr=[lo, hi], capsize=3,
                   color=["#2a78d6" if r["policy"] == "cache_aware" else "#9aa5b1" for r in rs])
            ax.set_xticks(range(len(rs)), names, rotation=35, ha="right", fontsize=7)
            ax.set_title(f"{lab}\n{k[0]} cache={k[1]} load={k[2]:g} (n={rs[0]['seeds']})", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
