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

## Open points

* `think_scale` 0.1 compresses real think times 10x. Results are reported against the cache size
  relative to the live working set, which is what the compression changes.
* The 0.34 VisionArena ceiling comes from a 100-conversation sample; each Kaggle run records the
  ceiling of the trace it actually replayed (`workload.json`).
* Cross-conversation image reuse can't be measured on the sample (100 distinct images); the
  full shards will show it.
