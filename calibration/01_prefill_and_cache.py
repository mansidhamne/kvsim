"""Calibration 1: fit the cost model and measure the cache on a real engine.

  A. cold prompts of increasing length      -> prefill_tokens_per_s, fixed_overhead_s
  B. warm prefix + short new suffix         -> is a cache hit really ~free?  (attention term)
  C. cache capacity probe                   -> how many prompts does the engine really keep?

Writes calibration/results/calibration.json (the `cost_model` section is what
CostModel.from_calibration() loads) and calibration/results/prefill_fit.png.

    python calibration/01_prefill_and_cache.py            # ~4 min on an M1
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from harness import RESULTS, TEXT_MODEL, OllamaServer, system_info, words  # noqa: E402
from _common import plt  # noqa: E402

NUM_CTX = 8192
CAP_MIB = 48                        # prompt-cache budget used for the capacity probe


def sample(res):
    t = res.task
    if t is None or not t.complete:
        return None
    return {"wall": res.wall_s, "n_total": t.n_tokens, "cached": t.cached, "evaluated": t.evaluated,
            "prompt_ms": t.prompt_ms}


def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return beta, se, r2


def run_a_and_b():
    rng = random.Random(7)
    with OllamaServer(11520, num_parallel=1, num_ctx=NUM_CTX, tag="calib1") as srv:
        info = system_info(srv)
        print("loading model + warm-up ...")
        srv.generate("hello", read_log=False)
        srv.warmup(n=6)

        # ---- A: cold prompts of increasing length ----
        lengths = [40, 100, 200, 400, 800, 1200, 1600]
        plan = [(n, r) for n in lengths for r in range(4)]
        rng.shuffle(plan)
        A = []
        for k, (n, r) in enumerate(plan):
            s = sample(srv.generate(words(n, 100_000 + n * 10 + r)))
            if s:
                A.append(s)
        print(f"A: {len(A)} cold samples, evaluated tokens {min(a['evaluated'] for a in A)}"
              f"..{max(a['evaluated'] for a in A)}")

        # ---- B: warm prefix + new suffix (and identical resend) ----
        B, floor = [], []
        for L in (140, 570, 1100):
            for rep in range(3):
                seed = 500_000 + L * 10 + rep
                P = words(L, seed)
                cold = sample(srv.generate(P))
                if cold:
                    A.append(cold)
                    B.append({**cold, "L": L, "S": 0})
                suffixes = [3, 15, 60, 250]
                rng.shuffle(suffixes)
                for S in suffixes:
                    s = sample(srv.generate(P + " " + words(S, seed + 7 * S + 1)))
                    if s:
                        B.append({**s, "L": L, "S": S})
                s = sample(srv.generate(P))        # identical resend: (almost) everything cached
                if s:
                    floor.append({**s, "L": L})
        print(f"B: {len(B)} samples; identical-resend samples: {len(floor)}")
    return info, A, B, floor


def run_c():
    """Capacity probe. For each m: insert m fresh prompts, then revisit the first one."""
    out = []
    with OllamaServer(11521, num_parallel=1, cache_ram_mib=CAP_MIB, num_ctx=NUM_CTX,
                      tag="calib1_cap") as srv:
        srv.generate("hello", read_log=False)
        srv.warmup(n=3)
        for m in (2, 4, 6, 8, 10, 12, 14, 16, 20, 24):
            first = None
            toks = []
            for i in range(m):
                P = words(150, 900_000 + m * 100 + i)
                if i == 0:
                    first = P
                s = sample(srv.generate(P))
                toks.append(s["n_total"] if s else 0)
            s = sample(srv.generate(first + " " + words(5, 31337 + m)))
            frac = (s["cached"] / s["n_total"]) if s else float("nan")
            out.append({"m": m, "mean_prompt_tokens": float(np.mean(toks)), "hit_fraction": frac})
            print(f"  C: after {m:2d} prompts, revisit first -> {frac:5.1%} of prompt from cache")
        log = srv.log_text()
    return out, log


def main():
    info, A, B, floor = run_a_and_b()

    # ---- fit A only (cold) ----
    xa = [a["evaluated"] for a in A if a["cached"] <= 8]
    ya = [a["wall"] for a in A if a["cached"] <= 8]
    (a0, b0), (se_a0, se_b0), r2_0 = ols(np.c_[np.ones(len(xa)), xa], ya)
    (_, _, q0), _, r2_q = ols(np.c_[np.ones(len(xa)), xa, np.square(xa)], ya)
    print(f"\nA (cold only): overhead = {a0 * 1e3:.0f} ms, rate = {1 / b0:.0f} tok/s, R2 = {r2_0:.4f}"
          f"  (adding a n^2 term raises R2 to {r2_q:.4f})")

    # ---- joint fit: t = a + b*n_new + c*n_new*n_cached ----
    S_all = A + [b for b in B if b["S"] > 0]
    n_new = np.array([s["evaluated"] for s in S_all], float)
    n_cached = np.array([s["cached"] for s in S_all], float)
    t = np.array([s["wall"] for s in S_all], float)
    beta, se, r2 = ols(np.c_[np.ones_like(t), n_new, n_new * n_cached], t)
    a, b, c = beta
    c_signif = c > 0 and c > 2 * se[2]
    print(f"joint fit ({len(t)} samples): overhead = {a * 1e3:.0f} ms, rate = {1 / b:.0f} tok/s, "
          f"attn c = {c:.2e} +/- {se[2]:.1e} s/(tok*tok), R2 = {r2:.4f}  "
          f"-> attention term {'USED' if c_signif else 'not significant, set to 0'}")

    # linear model (no attention) for the final cost model if c isn't significant
    if not c_signif:
        (a, b), _, _ = ols(np.c_[np.ones_like(t), n_new], t)
        c = 0.0

    hits = [x for x in B if x["S"] > 0]
    print("\nCache-hit cost (median wall ms):")
    for L in (140, 570, 1100):
        cold = [x["wall"] for x in B if x["L"] == L and x["S"] == 0]
        tok = np.median([x["n_total"] for x in B if x["L"] == L and x["S"] == 0])
        for S in (3, 15, 60, 250):
            h = [x["wall"] for x in hits if x["L"] == L and x["S"] == S]
            nn = np.median([x["evaluated"] for x in hits if x["L"] == L and x["S"] == S])
            print(f"  prefix ~{tok:5.0f} tok: cold {np.median(cold) * 1e3:7.0f} ms | "
                  f"+{S:3d} words (~{nn:4.0f} new tok): {np.median(h) * 1e3:6.0f} ms "
                  f"({np.median(cold) / np.median(h):5.1f}x faster)")
    fl = np.median([f["wall"] for f in floor]) if floor else float("nan")
    print(f"  identical re-send (everything cached): median {fl * 1e3:.0f} ms "
          f"(evaluated {np.median([f['evaluated'] for f in floor]):.0f} tok)")

    # ---- C: capacity ----
    print("\nC: cache-capacity probe (budget", CAP_MIB, "MiB):")
    cap, log = run_c()
    ok = [x for x in cap if x["hit_fraction"] > 0.9]
    bad = [x for x in cap if x["hit_fraction"] <= 0.9]
    m_max = max((x["m"] for x in ok), default=0)
    tok_per_prompt = float(np.mean([x["mean_prompt_tokens"] for x in cap]))
    cap_tokens = int(m_max * tok_per_prompt)
    import re
    states = re.findall(r"cache state: (\d+) prompts, ([\d.]+) MiB", log)
    kib_per_tok = None
    if states:
        n_p, mib = int(states[-1][0]), float(states[-1][1])
        kib_per_tok = mib * 1024 / max(1.0, n_p * tok_per_prompt)
    print(f"  largest m still hitting: {m_max} prompts ~ {cap_tokens} tokens"
          f"  (first miss at m={min((x['m'] for x in bad), default=None)})"
          f"; ~{kib_per_tok:.1f} KiB/token" if kib_per_tok else "")

    cost_model = {"prefill_tokens_per_s": round(1 / b, 1), "fixed_overhead_s": round(float(a), 4),
                  "attn_s_per_token_pair": float(c)}
    result = {"system": info, "cost_model": cost_model,
              "fit": {"cold_only": {"overhead_s": a0, "tokens_per_s": 1 / b0, "r2": r2_0,
                                    "r2_with_quadratic": r2_q},
                      "joint": {"overhead_s": float(beta[0]), "tokens_per_s": float(1 / beta[1]),
                                "attn_c": float(beta[2]), "attn_c_se": float(se[2]), "r2": r2}},
              "identical_resend_median_s": fl,
              "cache": {"budget_mib": CAP_MIB, "largest_m_hitting": m_max,
                        "mean_prompt_tokens": tok_per_prompt, "capacity_tokens": cap_tokens,
                        "kib_per_token": kib_per_tok, "probe": cap}}
    (RESULTS / "calibration.json").write_text(json.dumps(result, indent=1))
    (RESULTS / "calibration_raw_exp1.json").write_text(json.dumps({"A": A, "B": B, "floor": floor}))

    # ---- plot ----
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ax[0].scatter([a_["evaluated"] for a_ in A if a_["cached"] <= 8],
                  [a_["wall"] for a_ in A if a_["cached"] <= 8], s=18, color="#e60023")
    xs = np.linspace(0, max(xa), 50)
    ax[0].plot(xs, a0 + b0 * xs, color="black", lw=1.2,
               label=f"{1 / b0:.0f} tok/s + {a0 * 1e3:.0f} ms")
    ax[0].set(xlabel="prompt tokens evaluated (cold)", ylabel="wall time, num_predict=1 (s)",
              title="A. prefill cost vs length")
    ax[0].legend(frameon=False)
    for L, col in zip((140, 570, 1100), ("#fbbc04", "#e60023", "#7b1fa2")):
        h = [x for x in hits if x["L"] == L]
        ax[1].scatter([x["evaluated"] for x in h], [x["wall"] for x in h], s=18, color=col,
                      label=f"prefix ~{int(np.median([x['cached'] for x in h]))} tok cached")
    ax[1].set(xlabel="new tokens evaluated", ylabel="wall time (s)",
              title="B. warm-prefix requests (cache hit)")
    ax[1].legend(frameon=False)
    ax[2].plot([x["m"] for x in cap], [x["hit_fraction"] for x in cap], "-o", color="#4285f4")
    ax[2].set(xlabel="prompts inserted before revisiting the first", ylabel="share of prompt from cache",
              title=f"C. capacity probe ({CAP_MIB} MiB budget)", ylim=(-0.05, 1.05))
    fig.suptitle(f"Real measurements: {TEXT_MODEL} via Ollama {info['ollama_server_version']} on "
                 f"{info['machine']} (not data-center GPUs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "prefill_fit.png")
    print("\nwrote", RESULTS / "calibration.json", "and prefill_fit.png")
    print("cost_model =", json.dumps(cost_model))


if __name__ == "__main__":
    main()
