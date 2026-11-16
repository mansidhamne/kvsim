# Phase 3b: audit fixes, real traffic, scarce caches

Plan and reasoning: [phase3b_plan.md](phase3b_plan.md). This page: what was built, what the
real data showed before any GPU run, and how to run it.

## What the public data showed (measured locally, before any GPU time)

| | synthetic (runs 1-2) | VisionArena-Chat (real VLM chats) | ServeGen (real timing) | Mooncake (production) |
|---|---|---|---|---|
| turns per conversation | 3-8 (assumed) | **mean 1.46; 78% single-turn** (4,629 conversations, shard 0) | mean 3.5 (multi-turn conversations only) | no sessions |
| think time between turns | 12 s mean (assumed) | not recorded | **median 274 s; p10 36 s; p90 53 min** (4,104 follow-ups; send gap minus output / 30 tok/s, assumed) | n/a |
| images | 2-6 per conversation (assumed) | **1 per conversation**, ~0.47 megapixels median | n/a | none |
| user question | 5-60 words | **median 6 words**, p90 47 | n/a | n/a |
| model reply | 60-160 words | **median 127 words**, p90 393 | n/a | n/a |
| reuse ceiling | **0.85** | 0.34 (100-conversation sample) | n/a | **0.21** (6 min window), 0.37 (full hour) |

So the synthetic workload was far more reuse-friendly than real traffic: real VLM chats are mostly
one question about one image, and real users take minutes between turns, so a conversation's
cache has to survive minutes of other traffic. The scarce-cache case is the realistic one.

## What changed

| area | change |
|---|---|
| `bench/traces/servegen.py` | think-time distribution from ServeGen, committed as quantiles (`bench/traces/data/servegen_think.json`, derived from Apache-2.0 data) |
| `bench/traces/visionarena.py` | replays real conversations: real image bytes (uuid = hash of bytes), real user turns, original replies as fixed history (output length = original, capped at 384 tokens); starts at a set rate (optional burstiness); think time = ServeGen x `think_scale` (default 0.1, median ~27 s, so a run fits a session; the cache sweep covers the eviction pressure this removes); follow-ups landing > 60 s after the last conversation start are dropped and counted |
| `bench/traces/mooncake.py` | replays a 6-minute steady-state window of the production trace at real speed; each hash id = 32 fixed words, so prefix sharing is exact; seeds pick windows 15 min apart; every request is its own session (the trace has none) |
| `bench/traces/stats.py` | reuse ceiling and peak live working set per trace (saved as `workload.json` next to every run) |
| `bench/workload.py` | images from datasets; optional system prompt (VisionArena has none); think time counted from the end of the previous reply; per-chunk usage requested |
| `bench/loadgen.py` | follow-ups sent think time after the previous reply finished; inter-token latency per token (a chunk carrying k tokens counts as k gaps of gap/k); conversation start recorded for the warm-up filter |
| `bench/mock_worker.py` | `/reset_mm_cache`, `/reset_encoder_cache`, per-chunk usage, multi-token chunks, preemption counter |
| `router/server.py` | keep-alive 3 s (vLLM closes idle connections at 5 s) + safe retry |
| `deploy/phase3b.py` | resets prefix + multimodal + encoder caches before every cell; discarded warm-up replay per session; seeds with the policy order rotated; preemptions and retries per cell |
| `deploy/kaggle_run.py phase3b` | downloads VisionArena shards / the Mooncake trace on the GPU machine; cache sizes as `full`, a block count, or `x<fraction>` of the trace's live working set (vLLM restarted per size); aggregates across seeds |
| `bench/analyze.py` | `aggregate` (median and min-max over seeds, share of the reuse ceiling), `paired` (policy A vs B within each seed, then across seeds) |
| `deploy/validate.py validate_phase3b` | Phase 3 checks + all three caches reset + policy order rotated across seeds; preemptions and retries reported |

Licensing: VisionArena data is downloaded on Kaggle into `va_data/` (gitignored) and never
written to results; results hold latencies, token counts and aggregate statistics only.

## Verified on mock workers

Both real-data paths run end to end and validate (2 mock workers, time compressed 10x):
VisionArena (100 real conversations, 2 seeds, rotated order, all caches reset) and Mooncake
(257 production requests). 74 tests pass, including per-token ITL with 3-token chunks, think
time after the reply, cache-reset coverage, rotation, and the cross-seed paired comparison.

## Run it

Kaggle: import [`deploy/kaggle_phase3b.ipynb`](../deploy/kaggle_phase3b.ipynb) (GPU T4 x2,
Internet on). Run block 0 (load probe) first and set `HIGH` from it, then blocks A-E across
sessions. Download `p3b_results.zip` after each block.

Mac (mocks):

```bash
python -m bench.mock_worker --n 2 --base-port 8001 --time-scale 10 --cache-blocks 3000 &
python -m deploy.kaggle_run phase3b --workload visionarena --va-paths <local parquet> \
    --external http://127.0.0.1:8001,http://127.0.0.1:8002 --loads 0.3 --seeds 0,1 \
    --duration 120 --warmup 20 --session-warmup 30 --time-scale 10 --out results/p3b_mock
```

## Block 0: load probe (2026-11-15, 2x T4, VisionArena shards 0-1, round-robin, 1 seed)

Files: `results/phase3b/probe/` (copied with `deploy/sanitize_results.py`: model replies removed).

| conv starts/s | requests | TTFT p50 | TTFT p99 | ITL p99 | goodput | reuse ceiling | round-robin hit |
|---|---|---|---|---|---|---|---|
| 0.3 | 69 | 0.35 s | 1.65 s | 0.03 s | 1.00 | 0.28 | 0.22 |
| 0.6 | 128 | 0.59 s | 2.43 s | 0.06 s | 0.98 | 0.28 | 0.16 |
| **1.2** | 234 | 0.74 s | **3.04 s** | 0.40 s | **0.93** | 0.21 | 0.11 |
| 2.4 | 500 | 1.46 s | 8.28 s | 1.57 s | **0.26** | 0.24 | 0.16 |

Chosen loads: **high 1.2, mid 0.8, low 0.4** conversations/s (2.4 is past the knee).

What the full shards showed:
* images almost never repeat across conversations (0-3 of 48-387 per trace), so reuse on real
  VLM chats comes from follow-ups within a conversation; ceilings 0.21-0.28;
* 10-20% of follow-ups fall after the window (real think-time tail) and are dropped;
* **2 requests failed (HTTP 400): prompts over the 8,192-token limit.** The length filter used
  words x 1.3, and the router's estimate split text on spaces only, so Chinese/Japanese text
  (no spaces) was undercounted (router estimate vs engine: median 1.03x, p99 2.2x, worst 42x).
  Fixed: the filter now counts with the model's own tokenizer on the GPU machine, and the router
  counts one piece per CJK/kana/hangul/thai character (3-character pieces for other non-ASCII
  words). Validation correctly marked the probe NOT VALID; its numbers are only used to pick loads.
* The `EngineDeadError` at the end of the vLLM log is the normal shutdown (SIGTERM) after the
  run, not a crash.

## Block A (2026-11-15): VALID, 134 pass / 0 warn / 0 fail

VisionArena, 1.2 conversation starts/s, 2x T4, full cache (12,636 blocks/GPU) and 25% (4,094),
3 seeds with rotated policy order. Files: `results/phase3b/A/` (sanitized). 0 errors, 0
preemptions, 0 retries, all caches reset before every cell, length filter with the model's tokenizer.

Median over seeds [min, max]:

| cache | policy | hit | TTFT p50 | TTFT p99 | goodput |
|---|---|---|---|---|---|
| full | round_robin | 0.15 [0.15, 0.17] | 0.79 s | 4.42 s [2.73, 6.34] | 0.93 [0.85, 0.95] |
| full | least_requests | 0.16 [0.14, 0.19] | 0.82 s | 3.33 s [3.04, 5.83] | 0.91 [0.84, 0.92] |
| full | session_affinity | 0.24 [0.22, 0.29] | 0.66 s | 2.28 s [2.07, 4.55] | 0.98 [0.88, 0.98] |
| full | cache_aware | 0.22 [0.22, 0.29] | 0.66 s | 2.13 s [2.07, 5.67] | 0.98 [0.90, 0.99] |
| 25% | round_robin | 0.12 [0.10, 0.16] | 0.77 s | 4.45 s [4.16, 7.46] | 0.91 [0.83, 0.92] |
| 25% | least_requests | 0.12 [0.12, 0.16] | 0.89 s | 4.65 s [3.21, 4.89] | 0.83 [0.82, 0.92] |
| 25% | session_affinity | 0.22 [0.20, 0.27] | 0.68 s | 2.07 s [2.07, 5.56] | 0.98 [0.88, 0.98] |
| 25% | cache_aware | 0.21 [0.19, 0.26] | 0.68 s | 2.68 s [2.32, 5.90] | 0.95 [0.86, 0.97] |

Paired within each seed (same trace, same session), median [min, max] over 3 seeds:

| comparison | full cache | 25% cache |
|---|---|---|
| session affinity vs round-robin, TTFT p99 | **-28%** [-53, -16], all seeds | **-50%** [-53, -26], all seeds |
| session affinity vs round-robin, goodput | +4.2% [+3.2, +6.1], all seeds | +6.2% [+5.4, +8.3], all seeds |
| session affinity vs least-requests, TTFT p50 | -25% [-26, -19], all seeds | -24% [-25, -24], all seeds |
| cache-aware vs round-robin, TTFT p99 | -22% [-53, -11], all seeds | -40% [-44, -21], all seeds |
| **cache-aware vs session affinity**, TTFT p99 | +0.2% [-7, +25], mixed | **+12% [+6, +29], worse in all seeds** |
| cache-aware vs session affinity, goodput | +0.5% [-0.2, +1.8], mixed | -2.2% [-3.1, -0.8], worse in all seeds |

What the data says (and what it rules out):

* **Keeping a conversation on its GPU is what helps**: session affinity and cache-aware reuse
  ~50-80% more prompt tokens than the cache-blind balancers and cut TTFT p99 by 22-50% in every
  seed. Both reach ~100% of the trace's estimated reuse ceiling with the full cache (the ceiling
  is estimated with the router's token counts, so it is approximate).
* **Cache-aware routing adds nothing over session affinity on real VLM chats, and is slightly
  worse when the cache is smaller.** Images almost never repeat across conversations (1-5 per
  trace of ~400-470), so there is no cross-conversation reuse to exploit; cache-aware only differs
  by occasionally moving a conversation (8-9% of follow-ups). Every move lost the cache (0 tokens
  reused after the move). The router moved because the conversation's GPU looked ~0.4-0.6 s busier;
  its TTFT predictions are ~25% too low (median predicted/measured 0.75, correlation 0.70-0.75)
  because the cost model counts prefill work but not the 7-8 requests decoding on each GPU.
* **The 25% cache was only mildly scarce.** Real per-GPU demand (prompts of conversations waiting
  for their next turn + running requests) peaked at 23-52k tokens, median 10-16k; the 25% cap was
  65k. Evictions came from cache turnover (single-turn prompts fill it), not from the peak: follow-ups
  that stayed on their GPU reused 0.70 of their prompt with the full cache, 0.63 with 25%.
* **Reuse is understated for all policies by the canned history.** The history holds the dataset's
  original reply, not the reply our model generated (and vLLM cached), so the reply part of every
  follow-up is recomputed: cached tokens equal the previous prompt in 99% of follow-ups. With the
  model's own replies, follow-ups would reuse ~0.88 of their prompt instead of 0.70.
* Noise: seed 2 has a burst of conversation starts around 240-300 s that pushes every policy's
  TTFT p99 to 4.5-7.5 s; paired comparisons cancel it.

## Change after Block A: follow-ups carry the model's own replies (default from Block B on)

`--history own` (new default): each follow-up's history holds the reply the model actually
generated for the previous turn, as a chat app sends it. vLLM caches generated tokens while
decoding, so the reply part of a follow-up can now hit the cache (Block A, with the dataset's
replies, could only reuse the previous prompt: 99% of follow-ups stopped exactly there). The router
indexes prompt + generated reply when a request finishes, so its predictions include it.

Cost: prompts are identical across policies only while the generated replies are (greedy decoding,
but GPU batching isn't bit-exact). Every request's reply hash is stored per cell, and validation
reports the share of requests whose replies were identical in all policies (WARN below 90%).
Block A stays as the `--history canned` baseline.

## Block B as run (2026-11-15): 50% cache, own replies, 1 seed (VALID, 23 pass)

This ran the notebook's earlier Block B cell (50% cache = 8,188 blocks, seed 0) with the new code
(`--history own`); the scarce sizes (1,638 / 819 blocks) are still to run. Files:
`results/phase3b/B_x0.5_own/` (sanitized). One seed, so indicative only.

| policy | hit | TTFT p50 | TTFT p99 | ITL p99 | goodput |
|---|---|---|---|---|---|
| round_robin | 0.23 | 0.69 s | 2.95 s | 0.52 s | 0.946 |
| least_requests | 0.23 | 0.83 s | 3.14 s | 0.50 s | 0.904 |
| session_affinity | 0.35 | 0.57 s | 1.98 s | 0.44 s | 0.988 |
| cache_aware | 0.35 | 0.56 s | 1.82 s | 0.45 s | 0.986 |

* **Own replies work on real vLLM.** Follow-ups that stayed on their GPU reused 0.935 of their
  prompt (Block A, same trace, dataset replies: 0.72 full cache, 0.67 at 25%); in 92% of them the
  hit went past the previous prompt into the generated reply. vLLM caches generated tokens, and
  re-tokenizing the reply text reproduces them.
* **Replies differ between policies for 22% of requests** (18% of first turns, whose prompts are
  identical: GPU batching isn't bit-exact), so 8% of requests (a third of follow-ups) had slightly
  different prompt text. The work was identical: total prompt tokens 446,450-446,451 in every
  policy, because reply lengths are fixed per turn. Validation now checks equal totals (FAIL if
  they differ by > 0.5%) and reports the identical-reply share. vLLM 0.32's `VLLM_BATCH_INVARIANT`
  would make replies reproducible but uses slower kernels, which would distort the latency
  comparison, so it stays off.

## Block B2 (2026-11-16): scarce cache, 1,638 blocks/GPU (~26k tokens), own replies, 2 seeds

Files: `results/phase3b/B2/` (sanitized; `cells_reanalyzed.json` / `validation_reanalyzed.json`
use the corrected prediction metric). 0 errors; 3 preemptions (round-robin, seed 0).

| policy | seed 0: hit / p99 / goodput | seed 1: hit / p99 / goodput |
|---|---|---|
| round_robin | 0.13 / 4.42 s / 0.908 | 0.09 / 3.21 s / 0.930 |
| least_requests | 0.14 / 4.55 s / 0.854 | 0.08 / 3.52 s / 0.926 |
| session_affinity | 0.25 / 2.67 s / 0.948 | 0.18 / 2.71 s / 0.959 |
| cache_aware | 0.23 / 2.62 s / 0.960 | 0.17 / 2.33 s / 0.967 |

Paired over 2 seeds (median [min, max], same sign in both): cache-aware vs round-robin TTFT p99
**-34%** [-41, -28], goodput +4.8%; vs least-requests p99 -38%, p50 -26%, goodput +8.4%;
**vs session affinity p99 -8.1% [-14.2, -1.9], p50 -2.1%, goodput +1.0%**, hit rate -5.8%.

* The cache really was scarce: session-affinity follow-ups reused 0.63 of their prompt (0.935 at
  50% cache); 37% of follow-ups reused less than half.
* Why cache-aware edges out session affinity here (the reverse of block A's 25% result, where the
  cache was barely scarce): 82-92% of its moves were conversations whose cache had **already been
  evicted** from their GPU (router saw 0 cached tokens there), so moving cost nothing; session
  affinity keeps returning them to a GPU that lost their history. Cache-aware's follow-ups that
  stayed reused 0.68-0.73 vs 0.58-0.63, and follow-up TTFT p99 was 1.96 / 2.27 s vs 2.61 / 3.44 s.
  Small (2 seeds, -2% and -14%), but consistent and explained.
* The router's view gets less accurate when the engine evicts on its own: 87-91% of predictions
  within 5 points (99% elsewhere). This is the case the KV-event index (Phase 4) is for.

## Block D (2026-11-16): Mooncake production prefix pattern, real speed, full cache, 2 windows

Files: `results/phase3b/D/`. 0 errors, 0 preemptions, validation 47 pass / 0 warn / 0 fail
(after the metric fix). ~940-1,060 requests per window, reuse ceiling 0.22-0.23. No sessions, so
session affinity routes each request like a cache-blind balancer that uses prefill-backlog estimates.

| policy | window 0: hit / p50 / p99 / goodput | window 1: hit / p50 / p99 / goodput |
|---|---|---|
| round_robin | 0.15 / 0.68 / 4.52 / 0.772 | 0.16 / 0.68 / 4.10 / 0.811 |
| least_requests | 0.13 / 0.77 / 4.88 / 0.767 | 0.15 / 0.71 / 4.62 / 0.790 |
| session_affinity | 0.13 / 0.70 / 3.37 / 0.827 | 0.16 / 0.65 / 3.47 / 0.874 |
| cache_aware | **0.20 / 0.63 / 3.19 / 0.873** | **0.21 / 0.60 / 3.04 / 0.869** |

Paired over both windows (same sign in both unless noted): cache-aware vs round-robin TTFT p99
**-28%**, p50 -10%, goodput +10%, hit +35%; vs least-requests p99 -34%, p50 -17%, goodput +12%;
**vs session affinity: hit +44%, p99 -8.8% [-12.3, -5.3], p50 -8.3% [-9.8, -6.8]**, goodput +2.5%
(mixed).

* This is where cache-aware routing earns its keep: requests from different users share prefixes,
  so the router can send a request to the GPU that already holds its prefix. The reuse visible on
  the other GPU but missed drops from 43-66 to 14-21 tokens per request; cache-aware reaches ~90%
  of the reuse ceiling vs ~60-70% for the others. Load stays balanced enough (439/497 and 479/581
  requests per GPU).
* Session affinity (= prefill-backlog balancing here) beats least-requests on TTFT p99 (3.4 vs
  4.6-4.9 s): counting estimated prefill work balances better than counting requests.

## Phase 3b conclusions

| workload | what drives reuse | best policy | vs cache-blind (TTFT p99) |
|---|---|---|---|
| VisionArena, ample cache (A, B) | follow-ups in a conversation | session affinity = cache-aware | -22 to -50% |
| VisionArena, scarce cache (B2) | follow-ups whose cache survived | cache-aware, slightly (-8% p99 vs affinity) | -34 to -38% |
| Mooncake (D) | prefixes shared across users | **cache-aware** (-9% p99, +44% reuse vs affinity) | -28 to -34% |

Claims that hold (paired, same sign in every seed/window): routing on cache state cuts TTFT p99 by
22-50% vs cache-blind balancers on real VLM chats and by 28-34% on a production prefix trace, on
2x T4 with Qwen2-VL-2B. Cache-aware routing beats simple session pinning only when the cache is
scarce or prefixes are shared across sessions; on ample-cache chat traffic they are equivalent.

## Router v2: `cache_aware_v2` (built 2026-11-16, block V to run)

Two changes, each from a gap measured above (`router/policies.py`, `CacheAwareV2`):

1. **Decode-aware estimate.** Over 30,509 routed requests from blocks A-D, measured TTFT /
   router estimate was ~1.25x at low load and grew with the requests in flight on the chosen
   worker (1.24 at 2-3, 1.39 at 10-12, 1.63 at 13+): v1 counted only prefill backlog, not requests
   decoding. v2 multiplies each worker's estimate by (1 + 0.02 x requests in flight), fitted from
   those medians. The constant part (~0.2 s of image preprocessing / HTTP) is the same on every
   worker, so it doesn't change decisions and is not modeled.
2. **Switching cost.** v1's moves cost the whole cache when home still had it (block A, 25%:
   +12% TTFT p99 vs session affinity) and were free when home had already evicted it (block B2:
   82-92% of moves, -8% p99). v2 charges every non-home worker 1x the prefill time of the prefix
   still cached at home; nothing once home's copy is gone.

Tested: unit tests (stays when home holds the cache, moves once it is evicted, prefers the worker
with fewer requests decoding) and mock runs on VisionArena and Mooncake (VALID).

Block V (`deploy/kaggle_phase3b.ipynb`): V1 = B2's scarce-cache setup with seeds 0-3, V2 =
Mooncake with windows 0-2; policies round-robin, session affinity, cache-aware v1 and v2 (~4 h).

## Remaining runs (as of 2026-11-16, ~7 of 6-8 GPU-hours used)

Done: B2 and D. Optional: reduced load sweep C (round-robin, session affinity, cache-aware at 0.4
and 0.8 conversations/s, ~50 min).

## Remaining runs (as planned on 2026-11-15)

| block | what | why | ~time |
|---|---|---|---|
| B2 | VisionArena, **1,638 blocks/GPU (~26k tokens)**, own replies, seeds 0-1 | below the measured demand peak (23-52k): evictions certain; the scarce-cache test of cache-aware vs session affinity | 1.1 h |
| D | Mooncake, real speed, outputs capped at 64, full cache, seeds 0-1 (two windows) | cross-user prefix sharing with no sessions: where cache-aware could beat everything else; live set ~700k tokens vs ~200k cache per GPU | 1.1 h |

Skipped: 5% cache (819 blocks: running requests alone use ~8-11k of its ~13k tokens, so it would
measure preemptions, not routing), the load sweep (C) and 4 workers (E).

## Open points

* `think_scale` 0.1 compresses real think times 10x. Results are reported against the cache size
  relative to the live working set, which is what the compression changes.
* The 0.34 VisionArena ceiling comes from a 100-conversation sample; each Kaggle run records the
  ceiling of the trace it actually replayed (`workload.json`).
* Cross-conversation image reuse can't be measured on the sample (100 distinct images); the
  full shards will show it.
