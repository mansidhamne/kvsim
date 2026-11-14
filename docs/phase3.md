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
Internet on, Run All (~75-90 min).
