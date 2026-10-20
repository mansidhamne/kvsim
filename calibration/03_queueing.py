"""Calibration 3: does the simulator's worker model predict queueing on a real engine?

The simulator treats a worker as a single-server FIFO queue with the fitted prefill cost.
Here one Ollama server (parallelism 1 => serial) gets Poisson arrivals of cold prompts at
several utilisations rho; we replay the SAME arrival times and prompt lengths through the
simulator and compare mean / p95 latency.

Honest scope: one process, one laptop GPU, cold prompts only (no cache). It tests the
queueing model and the fitted cost model together, not routing.

    python calibration/03_queueing.py            # ~4 min
"""
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from harness import RESULTS, OllamaServer, parse_tasks, words  # noqa: E402
from _common import plt  # noqa: E402

from kvsim import ClusterConfig, CostModel, Request, RoundRobin, simulate  # noqa: E402

N_REQ = 30
RHOS = [0.3, 0.5, 0.7, 0.85, 1.0]
BS = 8                                    # fine blocks so simulated lengths match real ones


def speed_check(srv, cost, tag):
    """5 sequential cold requests: real vs fitted service time (detects drift / throttling)."""
    ratios = []
    for i in range(5):
        r = srv.generate(words(300, 91_000 + sum(map(ord, tag)) + i))
        pred = cost.fixed_overhead_s + r.task.n_tokens / cost.prefill_tokens_per_s
        ratios.append(r.wall_s / pred)
    m = float(np.mean(ratios))
    print(f"speed check [{tag}]: real/fitted service time = {m:.2f}")
    return m


def main():
    cal = json.loads((RESULTS / "calibration.json").read_text())
    cost = CostModel.from_calibration(RESULTS / "calibration.json", block_size=BS)
    print("cost model:", cost)
    rng = random.Random(3)
    out = []
    with OllamaServer(11540, num_parallel=1, num_ctx=8192, tag="calib3") as srv:
        srv.generate("hello", read_log=False)
        srv.warmup(n=5)

        # tokens/word and mean service time with a few sequential requests
        probe = [srv.generate(words(w, 60_000 + w)) for w in (100, 300, 600)]
        tpw = np.mean([p.task.n_tokens / w for p, w in zip(probe, (100, 300, 600))])
        pre = speed_check(srv, cost, "before load")
        mean_words = 350
        mean_service = cost.fixed_overhead_s + mean_words * tpw / cost.prefill_tokens_per_s
        print(f"tokens/word = {tpw:.2f}; modeled mean service = {mean_service:.2f} s")

        for rho in RHOS:
            lam = rho / mean_service
            gaps = [rng.expovariate(lam) for _ in range(N_REQ)]
            sched = np.cumsum(gaps)
            nwords = [rng.randint(100, 600) for _ in range(N_REQ)]
            lat, sent = [None] * N_REQ, [None] * N_REQ
            mark = srv._log_size()
            t0 = time.perf_counter()

            def fire(i):
                time.sleep(max(0.0, sched[i] - (time.perf_counter() - t0)))
                s = time.perf_counter()
                sent[i] = s - t0
                srv.generate(words(nwords[i], 70_000 + int(rho * 100) * 1000 + i), read_log=False)
                lat[i] = time.perf_counter() - s

            with ThreadPoolExecutor(max_workers=N_REQ) as ex:
                list(ex.map(fire, range(N_REQ)))

            reqs = []
            for i in range(N_REQ):
                toks = int(round(nwords[i] * tpw))
                nb = max(1, -(-toks // BS))
                blocks = tuple(range(1_000_000 * (i + 1), 1_000_000 * (i + 1) + nb))
                reqs.append(Request(i, i, 0, sent[i], blocks, tuple(float(k) for k in range(nb + 1))))
            reqs.sort(key=lambda r: r.arrival)
            cl1 = ClusterConfig(n_workers=1, cache_capacity_tokens=10**9)
            res = simulate(reqs, RoundRobin(), cl1, cost)
            sim = {r.rid: r.ttft for r in res.records}
            sim_lat = np.array([sim[i] for i in range(N_REQ)])
            lat = np.array(lat)

            # Oracle: same queue, but each request's service time is what the server really
            # spent (from its log, FIFO-matched). If this matches reality, the queueing model
            # is right and any remaining error is the cost model.
            tasks = [t for t in parse_tasks(srv._read_from(mark)) if t.complete]
            order = sorted(range(N_REQ), key=lambda i: sent[i])
            ok = len(tasks) == N_REQ
            corr = np.corrcoef([t.n_tokens for t in tasks], [nwords[i] for i in order])[0, 1] if ok else float("nan")
            oracle_mean = oracle_p95 = float("nan")
            infl = float("nan")
            if ok and corr > 0.95:
                svc = {i: tasks[k].prompt_ms / 1e3 + cost.fixed_overhead_s * 0.8 for k, i in enumerate(order)}
                infl = float(np.mean([svc[i] / (cost.fixed_overhead_s + nwords[i] * tpw / cost.prefill_tokens_per_s)
                                      for i in range(N_REQ)]))
                oreqs = [Request(i, i, 0, sent[i], (9_000_000 + i,),
                                 (0.0, max(1e-9, (svc[i] - cost.fixed_overhead_s) * cost.prefill_tokens_per_s / BS)))
                         for i in order]
                ores = simulate(oreqs, RoundRobin(), cl1, cost)
                olat = np.array([r.ttft for r in ores.records])
                oracle_mean, oracle_p95 = float(olat.mean()), float(np.percentile(olat, 95))
            row = {"rho": rho, "measured_mean": lat.mean(), "measured_p95": np.percentile(lat, 95),
                   "sim_mean": sim_lat.mean(), "sim_p95": np.percentile(sim_lat, 95),
                   "oracle_mean": oracle_mean, "oracle_p95": oracle_p95,
                   "service_inflation_vs_fit": infl,
                   "mean_abs_pct_err": float(np.mean(np.abs(sim_lat - lat) / lat) * 100)}
            out.append(row)
            print(f"rho={rho:4.2f}  mean: real {row['measured_mean']:6.2f}s  sim {row['sim_mean']:6.2f}s"
                  f" | p95: real {row['measured_p95']:6.2f}s  sim {row['sim_p95']:6.2f}s"
                  f" | per-request |err| {row['mean_abs_pct_err']:4.0f}%"
                  f"\n          sim w/ measured service times: mean {oracle_mean:6.2f}s p95 {oracle_p95:6.2f}s"
                  f"  (real service / fitted = {infl:.2f})")

        post = speed_check(srv, cost, "after load")
    (RESULTS / "queueing_validation.json").write_text(json.dumps(
        {"speed_check_before": pre, "speed_check_after": post, "rows": out}, indent=1, default=float))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, name in ((ax[0], "mean", "mean latency"), (ax[1], "p95", "p95 latency")):
        a.plot(RHOS, [r[f"measured_{key}"] for r in out], "-o", color="#e60023", label="real (Ollama, M1)")
        a.plot(RHOS, [r[f"sim_{key}"] for r in out], "--s", color="black", label="simulator, fitted cost model")
        a.plot(RHOS, [r[f"oracle_{key}"] for r in out], ":^", color="#4285f4",
               label="simulator, real per-request service times")
        a.set(xlabel="offered utilisation rho (modeled)", ylabel=f"{name} (s)", title=name)
        a.legend(frameon=False)
    fig.suptitle(f"Queueing validation, {N_REQ} cold requests per point, 1 server", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "queueing_validation.png")
    print("wrote", RESULTS / "queueing_validation.png")


if __name__ == "__main__":
    main()
