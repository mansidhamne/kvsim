# Phase 3b plan: fix the audit gaps, scarce caches, real traffic

Phase 3 (runs 1-2) showed affinity routing beats cache-blind balancing, but the audit found the
setup couldn't answer the more interesting questions, and that the traffic was invented. This
plan fixes the pipeline, adds scarce-cache experiments, and replays real traffic.

Decisions already made: ~6-8 GPU-hours on Kaggle is fine; conversation starts stay on a fixed
schedule while think time starts when the previous reply finishes; cache scarcity via
`--num-gpu-blocks-override` (main) plus one 4-worker run (check); use public traces.

## 1. What each public source gives us (all checked at the source)

| source | contents (verified) | license / terms | what we use it for |
|---|---|---|---|
| **VisionArena-Chat** (LMArena, [HF](https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat)) | 199,036 real user-VLM conversations (Feb-Sep 2024), 45 models, 138 languages, **one image per conversation**, ~43k unique images (so images repeat across conversations, partly from a "random image" button: `is_preset`). Fields: `images` (bytes), `conversation` (user and model turns), `num_turns`, `tstamp` (conversation start), `user_id` (hashed), `language`, `categories`. 43 parquet shards, 1.9-2.0 GB each, 84.6 GB total | Use allowed for research/commercial development. **No redistribution**: "should not distribute, copy, disclose ... host ... the dataset to any third party". Deletion on request | Real images, real questions, real turn counts, real reply lengths, real image reuse across conversations. **Not** per-turn timing |
| **ServeGen** (Alibaba, NSDI '26, [GitHub](https://github.com/alibaba/ServeGen)) | `data/conversations/conversations_hashed.json` (50 MB): real multi-turn conversations with a **timestamp per turn**, input/output token counts, content replaced by token hashes. `data/multimodal/mm-image/`: per-time-window distributions of text tokens, image tokens, image count, output tokens; arrival-rate traces with fitted burstiness (Gamma/Weibull) | Apache-2.0 | **Think time between turns** (real), turn-count and prompt-growth checks, **arrival burstiness** for image workloads |
| **Mooncake** (Kimi/Moonshot, FAST '25, [GitHub](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release)) | `traces/conversation_trace.jsonl` (3 MB): 12,031 requests from one hour of production, ms timestamps, input/output lengths (avg 12,035 / 343 tokens), `hash_ids` = chained block hashes (shared leading ids = shared prefix). No text, no images, no session ids | Apache-2.0 | A **production prefix-sharing pattern** at real timing; third-party analyses report a ~38-40% reuse ceiling vs 86% in our synthetic trace |

None of these alone is a multi-turn *image* trace with timing; together they cover it:
content from VisionArena, timing from ServeGen, a production reuse pattern from Mooncake.

## 2. Three workloads

**W1 Synthetic, recalibrated** (controlled sweeps). Our generator, with its knobs set from data
instead of guessed: turns per conversation and reply lengths from VisionArena, think time from
ServeGen, burstiness from ServeGen mm-image. One change is forced by the data: real VLM chats
here carry **one image**, not 2-6. Used where one factor must change at a time (cache size,
hot-prefix share).

**W2 VisionArena replay** (headline, realistic multimodal). Real conversations: real image bytes
(uuid = hash of the bytes, so repeated images are recognised), real user turns, the original
model replies as fixed history (output length = original reply length, capped), conversations
longer than the context limit dropped and counted. Conversation starts at a fixed rate scaled to
the target load (shape from ServeGen); each follow-up sent a think time (ServeGen distribution)
after the previous reply finishes. Reuse ceiling reported per trace.

**W3 Mooncake replay** (production prefix structure, text only). Each hash id becomes a fixed
block of deterministic text, so requests that share hash ids share a prefix exactly. Scaled to
the T4: tokens per hash id and arrival rate reduced by documented factors so prompts fit the 8k
context and the load matches W2; the sharing structure is unchanged. Tests whether the gains
survive a ~40% reuse ceiling instead of 86%.

**Licensing rule for the repo:** VisionArena content never goes into git or into the results
(no text, no images, no per-conversation ids). The repo holds the loader code, aggregate
statistics (distributions), and latency results. Trace files built from it stay on the Kaggle
disk and are gitignored. ServeGen/Mooncake-derived settings are committed with attribution.

## 3. Pipeline fixes (from the audit), before any GPU time

| # | gap found | fix | check that proves it |
|---|---|---|---|
| 1 | Only the prefix cache was reset; image caches carried over; RR always ran first on cold kernels | also `POST /reset_mm_cache` and `/reset_encoder_cache`; a discarded warm-up replay per session; **rotate policy order** across repeats | identical-request TTFT no longer depends on policy position |
| 2 | One run per cell; p99 varied 3.1-4.4 s between runs | 3 seeds for headline cells; report median and range across runs | spread reported with every claim |
| 3 | Think time counted from the previous send | think time starts when the previous reply ends; conversation starts fixed | `late_s` ~0 by construction; offered load logged per cell |
| 4 | ITL inflated when one streamed chunk carries 2 tokens | `stream_options.continuous_usage_stats` (supported in vLLM 0.32): divide each gap by tokens advanced | chunk/token mismatch handled for the 26% of affected requests |
| 5 | Policy ranking only shown at 2 loads | 4-level load sweep | knee visible per policy |
| 6 | Scarce-cache behaviour untested | `--num-gpu-blocks-override` at ~100% / 50% / 25% of the working set; record **preemptions** (metric name to confirm in the 0.32 source) | eviction misses and preemptions reported separately |
| 7 | 1 request lost to a stale keep-alive connection | done: router keep-alive 3 s < vLLM's 5 s, plus safe retry | 0 errors, retries counted |
| 8 | Cache-aware's added value invisible at a 99%-of-ceiling setup | scarce caches, hot-prefix share sweep (W1), real image reuse (W2) | reuse ceiling printed per workload; claims only where headroom exists |

All of it is tested on mock workers first (the mock gains the reset endpoints, a cache cap and
per-chunk usage).

## 4. GPU plan (~7 h on 2x T4, in sessions of 2-3 h)

Policies: round_robin, least_requests, session_affinity, cache_aware (least_loaded dropped: it
behaved like least_requests in both runs). Each cell ~8 min.

| block | workload | cache | load | seeds | cells | ~time |
|---|---|---|---|---|---|---|
| A headline | W2 VisionArena | full, 25% | high | 3 (rotated order) | 24 | 3.2 h |
| B cache sweep | W2 | 50% (25%/full from A) | high | 1 | 4 | 0.5 h |
| C load sweep | W2 | full | low, mid (high from A) | 1 | 8 | 1.1 h |
| D production pattern | W3 Mooncake | full, 25% | high | 1 | 8 | 1.1 h |
| E more workers | W2, 4 workers (2 per T4) | full | high | 1 | 4 | 0.6 h |
| | | | | | 48 | **~6.5 h** + setup |

"Load" levels are set from a short calibration pass on the real workload (where round-robin
starts sending follow-ups late), not guessed. Block E first gets a startup test (two vLLM
servers + vision encoders on one 15 GB T4 may not fit).

## 5. What it lets you claim (if the data supports it)

* "Evaluated on real VLM traffic (VisionArena: real images, questions, turn counts) with
  production think-time and burstiness (Alibaba ServeGen), and on a production prefix trace
  (Kimi Mooncake)."
* Affinity/cache-aware vs cache-blind, as ranges over 3 seeds.
* Whether cache-aware beats session affinity when caches are scarce or images repeat across
  conversations, and by how much. Either answer is a result.
* How the benefit depends on the reuse ceiling (86% synthetic vs ~40% production).

## 6. Order of work

1. Pipeline fixes 1-8 on mock workers, with tests.
2. Loaders: VisionArena (streams the shards it needs on Kaggle), ServeGen think-time and
   burstiness extraction, Mooncake replay. Print each workload's statistics and reuse ceiling.
3. Kaggle session 1: calibration pass + block A (seeds 1-2).
4. Sessions 2-3: blocks A (seed 3), B, C, D, E.
5. Analysis with spread across seeds; README and docs rewritten around the validated results.
