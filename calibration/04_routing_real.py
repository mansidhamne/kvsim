"""Calibration 4: run the routing policies on REAL engines and compare with the simulator.

* 3 `ollama serve` processes = 3 workers, each with its own prompt cache (budget set via
  LLAMA_ARG_CACHE_RAM) and serial execution (parallelism 1).
* A thin router in this process sends every request of a kvsim trace to one worker. Its view
  of each worker is the simulator's own `Worker` + `SequenceCache` objects (a "shadow index"
  updated as requests are routed and finish), so the SAME router classes run unchanged.
* Prompt text is derived from the trace's prefix-node ids, so identical prefixes are
  identical token prefixes and the engine's real prefix reuse can kick in.
* Cache accounting is exact, read from llama-server's log (tokens cached vs evaluated).
  Latency is the client wall-clock for num_predict=1.

CAVEAT: all workers share one laptop GPU, so they CONTEND. Token/hit counts are valid
regardless; latencies are only indicative. The simulator replay divides the per-worker
prefill rate by `--contention` (default = number of workers) to approximate sharing.

    python calibration/04_routing_real.py               # ~10 min, 4 policies
    python calibration/04_routing_real.py --smoke       # ~1.5 min sanity run
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from harness import RESULTS, OllamaServer, parse_tasks, words  # noqa: E402
from _common import plt  # noqa: E402

from kvsim import (ClusterConfig, CostModel, WorkloadConfig, generate_requests,  # noqa: E402
                   make_router, simulate, summarize)
from kvsim.cache import make_cache  # noqa: E402
from kvsim.worker import Queued, Worker  # noqa: E402

BS = 32                                   # simulated block size (tokens)
POLICIES = {
    "round_robin": {}, "least_loaded": {}, "session_affinity": {},
    "cache_aware": {"pending_aware": False},   # pending prefixes are not modeled for SequenceCache
}
LABEL = {"round_robin": "round-robin", "least_loaded": "least-loaded",
         "session_affinity": "session-affinity", "cache_aware": "cache-aware"}


class ShadowWorker(Worker):
    """The router's view of one real worker: resident cache (as it believes) + in-flight work."""

    def __init__(self, wid, cap_blocks):
        super().__init__(wid, cap_blocks, make_cache("sequence_lru", cap_blocks))
        self.inflight: dict[int, Queued] = {}

    def est_wait(self, now):
        return sum(q.est_service_s for q in self.inflight.values())

    def blind_wait(self, now):
        return sum(q.blind_service_s for q in self.inflight.values())

    def on_route(self, q: Queued):
        self.inflight[q.req.rid] = q
        for b in q.req.blocks:
            self._pending[b] = self._pending.get(b, 0) + 1

    def on_done(self, q: Queued):
        self.inflight.pop(q.req.rid, None)
        self.cache.insert(q.req.blocks)
        for b in q.req.blocks:
            c = self._pending[b] - 1
            if c:
                self._pending[b] = c
            else:
                del self._pending[b]


def measure_tokens_per_word(port) -> float:
    with OllamaServer(port, 1, num_ctx=8192, tag="route_tpw") as s:
        s.generate("hello", read_log=False)
        s.warmup(n=3)
        r = s.generate(words(300, 31_415))
        return r.task.n_tokens / 300


def pct(a, q):
    return float(np.percentile(a, q))


def run_real(name, trace, prompts, args, cost, cap_blocks):
    K = args.workers
    servers = [OllamaServer(args.port + i, 1, args.cache_mib, args.num_ctx, tag=f"route_{name}")
               for i in range(K)]
    for s in servers:
        s.start()
    try:
        for s in servers:
            s.generate("hello", read_log=False)
            s.warmup(n=3)
        marks = [s._log_size() for s in servers]
        shadow = [ShadowWorker(i, cap_blocks) for i in range(K)]
        router = make_router(name, np.random.default_rng(args.seed + 12345), **POLICIES[name])
        lock = threading.Lock()
        lat, assigned = {}, {}

        def work(req, wid, q, sent_at):
            servers[wid].generate(prompts[req.rid], read_log=False)
            done = time.perf_counter()
            with lock:
                shadow[wid].on_done(q)
            lat[req.rid] = done - sent_at

        pool = ThreadPoolExecutor(max_workers=len(trace))
        futs, t0 = [], time.perf_counter()
        for req in trace:
            delay = t0 + req.arrival * args.time_scale - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            with lock:
                wid, est = router.route(req, shadow, req.arrival, cost)
                q = Queued(req, est, cost.prefill_time(req.cum_weight[-1]))
                shadow[wid].on_route(q)
                assigned[req.rid] = wid
            futs.append(pool.submit(work, req, wid, q, time.perf_counter()))
        for f in futs:
            f.result()
        wall = time.perf_counter() - t0
        time.sleep(0.5)
        tasks = [[t for t in parse_tasks(s._read_from(m)) if t.complete]
                 for s, m in zip(servers, marks)]
    finally:
        for s in servers:
            s.stop()

    flat = [t for ts in tasks for t in ts]
    n = len(trace)
    if len(flat) != n:
        print(f"  WARNING: {len(flat)} server tasks parsed for {n} requests")
    tot, cached, evald = (sum(getattr(t, k) for t in flat) for k in ("n_tokens", "cached", "evaluated"))
    L = np.array([lat[r.rid] for r in trace])
    per_worker = [sum(1 for w in assigned.values() if w == i) for i in range(K)]
    return {"n": n, "tasks_parsed": len(flat), "hit_rate": cached / tot,
            "evaluated_tokens_per_req": evald / n, "prompt_tokens_per_req": tot / n,
            "lat_mean_s": float(L.mean()), "lat_p50_s": pct(L, 50), "lat_p95_s": pct(L, 95),
            "lat_p99_s": pct(L, 99), "per_worker_requests": per_worker, "wall_s": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--active-convs", type=int, default=12)
    ap.add_argument("--think", type=float, default=10.0)
    ap.add_argument("--horizon", type=float, default=90.0)
    ap.add_argument("--cache-mib", type=int, default=40)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=11600)
    ap.add_argument("--contention", type=float, default=None)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.horizon, args.policies = 25.0, "least_loaded,cache_aware"
    contention = args.contention or args.workers

    cal = json.loads((RESULTS / "calibration.json").read_text())
    kib_per_tok = cal["cache"]["kib_per_token"]
    base = cal["cost_model"]
    cost_live = CostModel(block_size=BS, prefill_tokens_per_s=base["prefill_tokens_per_s"],
                          fixed_overhead_s=base["fixed_overhead_s"], image_cost_multiplier=1.0)
    cost_sim = replace(cost_live, prefill_tokens_per_s=cost_live.prefill_tokens_per_s / contention)

    wl = WorkloadConfig(seed=args.seed, horizon_s=args.horizon, warmup_s=0.0,
                        active_convs=args.active_convs, think_mean_s=args.think, think_min_s=0.5,
                        turns_min=3, turns_max=5, system_tokens=192, images_min=1, images_max=3,
                        tokens_per_image=96, user_tokens=(20, 60), response_tokens=(40, 120),
                        n_popular_boards=6, popular_prob=0.3)
    trace = generate_requests(wl, cost_live)
    print(f"trace: {len(trace)} requests, {len({r.conv_id for r in trace})} conversations, "
          f"{trace[-1].arrival:.0f}s of arrivals")

    print("measuring tokens/word ...")
    tpw = measure_tokens_per_word(args.port + 50)
    wpb = max(1, round(BS / tpw))
    chunk_cache = {}

    def chunk(node):
        if node not in chunk_cache:
            chunk_cache[node] = words(wpb, 12_345_678 + node)
        return chunk_cache[node]

    prompts = {r.rid: " ".join(chunk(b) for b in r.blocks) for r in trace}
    tpb = tpw * wpb
    cap_tokens = args.cache_mib * 1024 / kib_per_tok
    cap_blocks = max(1, int(cap_tokens / tpb))
    mean_blocks = np.mean([r.n_blocks for r in trace])
    print(f"tokens/word {tpw:.2f} -> {wpb} words/block ~ {tpb:.1f} tok/block; mean prompt "
          f"~{mean_blocks * tpb:.0f} tok; worker cache budget {args.cache_mib} MiB ~ {cap_tokens:.0f} "
          f"tokens ~ {cap_blocks} blocks (~{cap_tokens / (mean_blocks * tpb):.1f} mean prompts)")

    cluster = ClusterConfig(n_workers=args.workers, cache_capacity_tokens=cap_blocks * BS,
                            cache_kind="sequence_lru")
    rows = {}
    for name in args.policies.split(","):
        print(f"\n=== {LABEL[name]}: real run ===")
        real = run_real(name, trace, prompts, args, cost_live, cap_blocks)
        res = simulate(trace, make_router(name, np.random.default_rng(args.seed + 12345),
                                          **POLICIES[name]), cluster, cost_sim)
        s = summarize(res, 0.0, 1e12)
        sim = {"hit_rate": s["hit_rate"],
               "evaluated_tokens_per_req": (1 - s["hit_rate"]) * mean_blocks * tpb,
               "lat_mean_s": s["ttft_mean_s"], "lat_p95_s": s["ttft_p95_s"],
               "lat_p99_s": s["ttft_p99_s"]}
        rows[name] = {"real": real, "sim": sim}
        print(f"  real: hit {real['hit_rate']:.2f}  evaluated {real['evaluated_tokens_per_req']:.0f} tok/req"
              f"  latency mean {real['lat_mean_s']:.2f}s p95 {real['lat_p95_s']:.2f}s"
              f"  per-worker {real['per_worker_requests']}")
        print(f"  sim:  hit {sim['hit_rate']:.2f}  evaluated {sim['evaluated_tokens_per_req']:.0f} tok/req"
              f"  latency mean {sim['lat_mean_s']:.2f}s p95 {sim['lat_p95_s']:.2f}s")

    out = {"args": vars(args), "tokens_per_word": tpw, "tokens_per_block": tpb,
           "cap_blocks": cap_blocks, "contention": contention, "rows": rows}
    (RESULTS / ("routing_real_smoke.json" if args.smoke else "routing_real.json")).write_text(
        json.dumps(out, indent=1, default=float))

    names = list(rows)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(names))
    for a, (key, title) in zip(ax, (("hit_rate", "token hit rate (exact, from server logs)"),
                                    ("evaluated_tokens_per_req", "tokens recomputed per request"),
                                    ("lat_p95_s", "p95 latency (s) - indicative only"))):
        a.bar(x - 0.2, [rows[n]["real"][key] for n in names], 0.4, color="#e60023", label="real engines")
        a.bar(x + 0.2, [rows[n]["sim"][key] for n in names], 0.4, color="#9aa0a6", label="simulator")
        a.set_xticks(x, [LABEL[n] for n in names], rotation=15)
        a.set_title(title, fontsize=10)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Routing on {args.workers} real Ollama workers (1 shared M1 GPU) vs simulator replay, "
                 f"{len(trace)} requests", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / ("routing_real_smoke.png" if args.smoke else "routing_real.png"))
    print("\nwrote results to", RESULTS)


if __name__ == "__main__":
    main()
