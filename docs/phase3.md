# Phase 3: the router on real workers

A KV-cache-aware router in front of vLLM workers, compared against the usual load-balancing
policies on the same multi-turn image trace.

## The router (`router/`)

An OpenAI-compatible proxy (`python -m router.server --workers URL1,URL2 --policy cache_aware`).
Any client written for vLLM or the OpenAI API can point at it unchanged.

| endpoint | behavior |
|---|---|
| `POST /v1/chat/completions` | routed per request; streaming and non-streaming; images |
| `POST /v1/completions` | routed the same way (the text prompt is hashed as one message) |
| `GET /v1/models` | forwarded to the first worker that answers |
| `GET /health`, `/metrics`, `/stats`; `POST /reset` | router's own status and controls |

Not covered: embeddings, audio, batch and other OpenAI endpoints (no use for them here).

```
client ──► router ──► vLLM worker 0
             │  ├───► vLLM worker 1
             │  1. hash the prompt into 16-token blocks (router/hashing.py)
             │  2. score workers with the policy (router/policies.py)
             │  3. stream the worker's reply back unchanged
             │  4. first token seen -> prompt is now cached on that worker (shadow index)
             │  5. final usage chunk -> log predicted vs actual cached tokens
             └─ every 0.5 s: scrape each worker's /metrics (queue, KV usage, KV capacity)
```

| piece | what it does |
|---|---|
| `hashing.py` | messages -> pseudo-tokens (words, role markers, image tokens keyed by the image `uuid`) -> chained block hashes, like vLLM's structure. Also the per-block cost weight (image tokens cost more). |
| `index/shadow.py` | per worker: an LRU of blocks the router believes are cached (sized from `vllm:cache_config_info`), plus blocks of requests still prefilling |
| `load.py` | requests in flight; prefill backlog = sum of estimated prefill times of requests that haven't produced a first token, minus time already spent |
| `policies.py` | the simulator's policies (`kvsim.routers`) run unchanged through an adapter, plus `least_requests` |
| `server.py` | the proxy, `/stats`, `/metrics`, `/reset`, decision log (JSONL) |

The cost model comes from Phase 2 (`calibration_vllm_final.json`): the router's estimates of
prefill time include the attention term and the image-token cost measured on the T4.

### Policies

| policy | rule |
|---|---|
| round_robin | next worker in turn |
| least_requests | fewest requests in flight (what a generic HTTP load balancer does) |
| least_loaded | smallest prefill backlog, counting every prompt as uncached (cache-blind) |
| session_affinity | a conversation's first turn goes to the least-loaded worker, later turns stay there |
| cache_aware | argmin over workers of `backlog + alpha * estimated prefill of the uncached part`; alpha = 1 means "lowest estimated time to first token" |

### What the shadow index can't see

It learns about a prompt being cached only from its own routing decisions, so it misses
evictions caused by the engine's memory pressure and is stale while requests queue. Its
accuracy is measured on every request (`prediction_exact_frac`: predicted cached tokens within
one block of what vLLM reported). On mock workers it is ~100% at light load and ~50% when
overloaded. Phase 4 replaces it with an index fed by vLLM's KV events.

## The experiment (`deploy/phase3.py`)

For each load level (active conversations) and each policy:

1. reset every worker's prefix cache (`POST /reset_prefix_cache`, needs `VLLM_SERVER_DEV_MODE=1`)
2. start a fresh router with that policy
3. replay the same trace (same fingerprint for every policy), open loop, in real time
4. save `requests.jsonl`, `decisions.jsonl`, `cell.json` per cell; `cells.json`, `phase3.png`,
   `validation.json` overall

Requests scheduled in the first 60 s are warm-up and not measured. Before the experiment, every
worker must pass the Phase 2 smoke check (including the "Paris" fp16 check).

Metrics per cell: hit rate (engine-reported cached / prompt tokens), recomputed tokens per
request, TTFT p50/p90/p99, ITL p50/p99, goodput (share of requests with TTFT <= 2 s and mean
ITL <= 100 ms), throughput, per-worker request counts, router decision time, prediction accuracy.

Validation FAILs on: different traces across policies, a cache reset that didn't succeed, any
failed request, missing TTFT, missing usage details, wrong output length, unreadable output.
WARNs on: server and client cache counts disagreeing, more than 20% of follow-up turns sent over
1 s late (the system was overloaded for the trace), router predictions under 70% accurate.

## Run it

On the Mac (mock workers, ~1 minute):

```bash
python -m bench.mock_worker --n 2 --base-port 8001 --time-scale 10 --cache-blocks 1200 &
python -m deploy.kaggle_run phase3 --external http://127.0.0.1:8001,http://127.0.0.1:8002 \
    --convs 12,24 --duration 120 --warmup 20 --time-scale 10 --quick --out results/phase3_mock
```

On Kaggle: import [`deploy/kaggle_phase3.ipynb`](../deploy/kaggle_phase3.ipynb), GPU T4 x2,
Internet on, Run All (~95 min: each policy replays ~8 min of real traffic per load level, with a status line every 60 s).

## Runs

### Run 1 (2026-11-14, 2x T4, Qwen2-VL-2B): NOT VALID for the policies that use estimates

Files: `results/phase3/run1/` (per-request logs gzipped). The run itself completed cleanly: 0
errors, caches reset before every policy, server and client cache counters identical, the same
trace (same total prompt tokens) for every policy. Re-analysis with the later checks
(`validation_reanalysis.json`) found a bug in the router's cost estimates:

**The bug.** The cost model weighted image tokens 5.9x (vision encoder) and then fed the
*weighted* token count into the quadratic attention term, so image-heavy cold prompts were
estimated ~9x too slow (median 9 s estimated vs 1 s measured; up to 30 s for a request that took
2.2 s). Fixed: attention uses raw tokens, only the linear part is weighted. On the same requests
the corrected estimates are 1.06-1.10x the measured TTFT (p10-p90: 0.8-1.3x). A new FAIL check
compares estimates with measured TTFT on mostly-uncached requests, and a regression test pins
the formula.

**What it affected.** least_loaded, session_affinity (its first-turn placement) and cache_aware
act on these estimates. cache_aware moved 2% (12 convs) and 6% (24 convs) of follow-up turns away
from the worker that held their cache, because it believed that worker had a 13-17 s backlog
(real waits were well under 1 s). Those moved turns hit only 35-56% of their prompt.
round_robin and least_requests don't use estimates, so their numbers stand.

**Results that stand** (cache-blind baselines; 95% bootstrap CIs):

| convs | policy | hit | TTFT p50 | TTFT p99 [CI] | ITL p99 | goodput [CI] |
|---|---|---|---|---|---|---|
| 12 | round_robin | 0.75 | 0.33 s | 2.41 s [2.11, 2.58] | 0.30 s | 0.96 [0.95, 0.98] |
| 12 | least_requests | 0.75 | 0.33 s | 2.46 s [2.11, 2.96] | 0.25 s | 0.96 [0.94, 0.98] |
| 24 | round_robin | 0.75 | 0.41 s | 4.41 s [3.92, 5.02] | 0.80 s | 0.78 [0.75, 0.81] |
| 24 | least_requests | 0.76 | 0.40 s | 3.13 s [2.80, 3.66] | 0.68 s | 0.87 [0.85, 0.89] |

**Indicative only** (estimates 7-11x too high): session_affinity 0.85-0.86 hit, p99 1.97 s /
2.09 s, goodput 0.99 / 0.97; cache_aware 0.84-0.85 hit, p99 1.73 s / 2.39 s, goodput 1.00 / 0.95.
Even with the bug, cache-aware routing roughly halved TTFT p99 vs round-robin at 24 convs.

**Other findings**

* The router's view of each cache is accurate: predicted cached share within 5 points of the
  engine's for 99-100% of requests (1 large miss in 990). Predictions run ~32 tokens low because
  the router counts words and the tokenizer produces ~2% more tokens; the old "within one block"
  check flagged this as a 32-37% match, so it was replaced by the share-based check.
* With 2 workers and ~200k tokens of KV cache each (12,636 blocks, read from
  `vllm:cache_config_info`), nothing is evicted, so even round-robin reuses 75%: a miss leaves a
  second copy behind. Follow-ups that stayed on their worker reused 92-93% of the prompt; moved
  ones 61-67%.
* At 24 convs the system is overloaded for round_robin (36% of follow-ups sent > 1 s late);
  least_requests' load awareness alone cuts TTFT p99 from 4.4 s to 3.1 s.

**Next run:** same notebook, all five policies again (the cache-blind ones double as a
run-to-run noise check). Then a scarce-cache run (`--num-gpu-blocks-override`, or 4 workers).

### Run 2 (2026-11-14, 2x T4, Qwen2-VL-2B, cost-estimate fix): VALID except one cell

Files: `results/phase3/run2/`. Validation: 43 pass, 4 warn, 1 fail. The router's prefill
estimates are now realistic for every policy (median 1.03-1.17x the measured TTFT on
mostly-uncached requests; run 1 was 7-11x). Predicted cached share within 5 points of the
engine's for 99-100% of requests.

**The one FAIL.** `24 convs / least_loaded`: 1 of 895 requests failed in 6 ms with
`ServerDisconnectedError`. The router reused a pooled keep-alive connection the worker had just
closed; the request never reached vLLM (no error in either vLLM log). Fixed: the router now
retries once or twice when the connection fails before any response byte (safe, nothing ran),
and counts retries. That cell's numbers are indicative; it's a cache-blind baseline.

**Results** (requests scheduled after the 60 s warm-up; TTFT p99 and goodput with 95% bootstrap CIs;
goodput = share of requests with TTFT <= 2 s and mean ITL <= 100 ms)

| convs | policy | hit | recomputed tok/req | TTFT p50 | TTFT p99 [CI] | ITL p99 | goodput [CI] |
|---|---|---|---|---|---|---|---|
| 12 | round_robin | 0.75 | 542 | 0.32 s | 2.30 s [2.02, 2.42] | 0.28 s | 0.977 [0.962, 0.989] |
| 12 | least_requests | 0.75 | 541 | 0.30 s | 3.14 s [2.28, 3.64] | 0.23 s | 0.956 [0.937, 0.973] |
| 12 | least_loaded | 0.75 | 552 | 0.30 s | 2.29 s [2.03, 2.48] | 0.30 s | 0.971 [0.956, 0.985] |
| 12 | session_affinity | 0.85 | 321 | 0.23 s | 1.93 s [1.69, 1.99] | 0.17 s | 0.996 [0.989, 1.000] |
| 12 | **cache_aware** | 0.86 | 313 | 0.24 s | **1.68 s** [1.54, 1.85] | 0.18 s | **0.998** [0.994, 1.000] |
| 24 | round_robin | 0.75 | 511 | 0.41 s | 3.11 s [2.88, 3.46] | 0.77 s | 0.828 [0.802, 0.854] |
| 24 | least_requests | 0.76 | 500 | 0.40 s | 2.76 s [2.60, 3.12] | 0.66 s | 0.864 [0.842, 0.885] |
| 24 | least_loaded (1 error) | 0.76 | 496 | 0.35 s | 2.78 s [2.51, 2.92] | 0.66 s | 0.884 [0.863, 0.904] |
| 24 | session_affinity | 0.86 | 298 | 0.26 s | 1.77 s [1.56, 1.87] | 0.33 s | 0.992 [0.987, 0.998] |
| 24 | **cache_aware** | 0.85 | 305 | 0.25 s | **1.69 s** [1.53, 1.96] | 0.32 s | **0.990** [0.982, 0.996] |

**What it shows**

* **Cache-aware vs the cache-blind balancers, at 24 conversations (busy):** TTFT p99 1.69 s vs
  3.11 s (round-robin, -46%) and 2.76 s (least-requests, -39%); goodput 0.99 vs 0.83 / 0.86, i.e.
  1.90 vs 1.59 requests/s meeting the SLO (+20%) at the same offered load; ITL p99 0.32 s vs
  0.77 s (-58%), because fewer long recomputes compete with decoding; 40% fewer prompt tokens
  recomputed per request (305 vs 511). The CIs don't overlap.
* **Cache-aware vs session affinity: no measurable difference** in this setup (every CI
  overlaps). Pinning a conversation already captures nearly all the reuse when caches are
  large (nothing is evicted with ~200k tokens of KV per worker) and traffic is spread evenly.
  The simulator predicted the same. Cache-aware moved only 1.3% / 3.0% of follow-ups (run 1,
  with the bug: 2% / 6%). Where it should differ (untested so far): scarce caches, hot prefixes,
  uneven load, worker restarts.
* **Router overhead:** 2.3 ms median, 4.3-5.0 ms p99 per decision, the same for every policy:
  it's prompt hashing (reading image headers, hashing ~2k tokens), not the policy. About 1% of
  a cache-hit TTFT.

**Run-to-run noise** (the cache-blind policies weren't affected by the bug, so run 1 vs run 2 is a
repeat): hit rate within 0.005, TTFT p50 within 0.03 s, goodput within 0.05. TTFT p99 is much
noisier than its bootstrap CI suggests when the system is overloaded: round-robin at 24 convs
measured 4.41 s [3.92, 5.02] in run 1 and 3.11 s [2.88, 3.46] in run 2. So compare policies
within one run, and claim tail improvements as ranges over repeated runs (Phase 6: 3 seeds).
