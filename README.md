# kvsim — KV-cache-aware routing for multimodal inference

When a user returns to a conversation, much of the prompt may already be cached on one inference worker. Sending the next request to that worker saves prefill work, but it may also mean waiting in a longer queue. **When is reuse worth the wait?**

I built `kvsim` to explore that tradeoff. It compares routing policies on synthetic, multi-turn, image-heavy traffic, then checks parts of the model against real inference runs with Ollama on an Apple M1.

The project was inspired by [Pinterest's post on VLM serving with NVIDIA Dynamo](https://medium.com/pinterest-engineering/building-pinterests-vlm-serving-stack-on-nvidia-dynamo-0dce6e93d0f3). It is an independent experiment, not an implementation of Dynamo's router.

**A note on the numbers:** latency in the main simulation plots is *modeled TTFT* (queue wait + estimated prefill), not measured GPU latency. Real-engine measurements are reported separately below.

![Routing policies across concurrency levels](results/concurrency_sweep.png)

## How it works

The simulator models three prefill workers, each with a FIFO queue and an LRU KV cache. Requests come from generated conversations with a shared system prompt, 4–16 images, and 3–8 turns. Some conversations also share popular image boards, creating opportunities for reuse across sessions.

Each worker has a 150k-token cache budget. The block-cache model uses 64-token blocks; image tokens are assigned 1.5× the recomputation cost of text tokens. These are **simulation assumptions**, not measurements from production GPUs.

I compare four policies:

| Policy | Decision |
| --- | --- |
| Round-robin | Cycle through workers |
| Least-loaded | Pick the worker with the least estimated queued work, without looking at its cache |
| Session affinity | Keep every turn of a conversation on its original worker |
| Cache-aware | Choose the worker with the lowest estimated queue wait + prefill cost, accounting for cached prefixes |

The cache-aware policy uses:

```text
score(worker) = queue_wait(worker) + α × estimated_prefill(request, worker)
```

At `α = 1`, it chooses the worker with the lowest estimated time to first token. Larger values favor cache reuse more strongly, even when a cached worker is busy.

All policies receive the same seeded, open-loop request trace. The main simulation results average three seeds, exclude the first 120 seconds as warm-up, and report mean and tail TTFT estimates, cache-hit rates, recomputed tokens, throughput, and load imbalance.

![Synthetic conversation workload](results/workload.png)

## Results

### Cache reuse improves modeled latency — until queues dominate

At 100 active conversations, cache-aware routing increased the simulated token hit rate from **48% to 76%**, reduced recomputed prefix tokens per request from **1,830 to 469**, and lowered modeled p99 TTFT by **1.4×** compared with least-loaded routing.

| Active conversations | Hit rate: least-loaded → cache-aware | Recomputed tokens/request | Modeled p99 TTFT improvement |
| ---: | ---: | ---: | ---: |
| 10 | 64% → 84% | 1,016 → 6 | 1.2× |
| 100 | 48% → 76% | 1,830 → 469 | 1.4× |
| 150 | 42% → 63% | 2,144 → 1,118 | 5.6× |
| 200 | 39% → 54% | 2,283 → 1,520 | 24× |

The very large gains at high load aren't a steady speedup: under these assumptions, cache-aware routing pushes the saturation point higher, so the other policies accumulate long queues. Those tail-latency figures depend heavily on the cost and traffic models.

### Session affinity is a surprisingly strong baseline

At 100 active conversations, session affinity reached a **75%** hit rate, versus **76%** for cache-aware routing. Simply keeping a conversation on one worker captures most of the reuse.

Cache-aware routing did better on modeled p99 TTFT in this setup (**0.32 s vs. 0.40 s** at 100 conversations) because it could also balance queues and reuse shared prefixes across sessions. Whether that extra complexity is worthwhile depends on the workload and cache implementation.

### Not all cache misses are routing problems

I split recomputed tokens into three groups: **routing misses** (the prefix is on another worker), **eviction misses** (it was previously cached but is now gone), and **cold tokens** (not seen before).

At 100 active conversations, 1,097 of the least-loaded policy's 1,830 recomputed tokens/request were routing misses. At 200, eviction misses dominated instead (1,510 of 2,283). A better router can avoid sending requests to the wrong worker; it cannot recover blocks that no worker still holds.

### More affinity is not always better

![Break-even point for waiting versus recomputing](results/breakeven.png)

If one worker has a cached prefix and another is idle, waiting makes sense only while the queue delay is smaller than the work saved by avoiding prefill. With the simulation's assumed 25k tokens/s prefill speed, redoing a 2k-token prefix costs about 80 ms; an 8k-token prefix costs about 320 ms.

![Routing stress tests](results/stress_tests.png)

A few stress tests made the tradeoff clearer:

- **Hot prefixes:** When 90% of conversations shared two frequently reused prefixes, aggressive affinity (`α = 16`) produced *worse* modeled p99 TTFT than least-loaded routing (**0.49 s vs. 0.32 s**) by concentrating traffic on the same workers.
- **Cache capacity:** The advantage narrowed with very small caches (frequent eviction) and very large caches (prefixes eventually replicated across workers).
- **Single-turn traffic:** With little opportunity for reuse, cache-aware and least-loaded routing had essentially the same modeled mean latency (**0.17 s** each).
- **Pending prefixes:** Tracking prefixes queued for future insertion had little effect until the system approached saturation.

## Checking the model against real inference

I used **Ollama / llama.cpp on an 8 GB Apple M1**, with `qwen2.5:0.5b` for text experiments and Moondream for a small image-caching check. These experiments test cache behavior, prefill cost, and queueing assumptions; they do not reproduce a vLLM or multi-GPU deployment.

| Experiment | Observation on the M1 |
| --- | --- |
| Cold prefill | About **870 tokens/s** over 70–2,900 tokens; linear fit **R² ≈ 0.98** |
| Repeated prefix | A prompt with a 1,953-token cached prefix and ~8 new tokens took **47 ms**, versus **2,138 ms** cold |
| Queueing | Replaying measured per-request service times matched observed latency within roughly **10%** |
| Routing | Across **179 requests** and three local server processes, simulated cache-hit rates were within **3 percentage points** of the measured rates |
| Repeated image | With Moondream, a follow-up question about the same image took **0.12 s** versus **4.1 s** for the initial request; this is a small exploratory test, not a general VLM speedup claim |

The real engine used a **whole-prompt cache with longest-common-prefix reuse**, rather than the block-based cache in the main simulation. I added a separate `SequenceCache` model so the replay would use the same cache semantics.

![Prefill calibration](calibration/results/prefill_fit.png)

![Queueing validation](calibration/results/queueing_validation.png)

### Routing on local inference servers

I ran four policies against three separate `ollama serve` processes, each maintaining its own prompt cache. They ran on the **same physical M1 GPU**, so latency comparisons are affected by contention; cache-hit counts are more useful for validating the simulator here.

| Policy | Measured hit rate | Simulated hit rate |
| --- | ---: | ---: |
| Round-robin | 38% | 40% |
| Least-loaded | 38% | 39% |
| Session affinity | 53% | 56% |
| Cache-aware | 46% | 48% |

![Real routing measurements](calibration/results/routing_real.png)

One result changed my initial expectations: **session affinity beat cache-aware routing on hit rate** in this real-engine setup. Under a small whole-prompt cache, cache-aware routing moved 26% of follow-up turns to a different worker at a 40 MiB cache budget. Those moves often lost the cached conversation context and duplicated data across workers. In additional *simulated* cache-capacity sweeps over eight seeds, session affinity matched or beat cache-aware routing on hit rate across the tested budgets.

That contrasts with the block-cache simulation, where cache-aware routing had better modeled tail latency. The lesson from this experiment is that **routing decisions depend on how the engine stores and reuses cached prefixes**, not just on how much request overlap exists.

The real routing benchmark used one trace and one run per policy. The model captured hit rates much better than latency rankings, which were sensitive to GPU sharing and changes in laptop performance.

## Run it

### Simulation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest

python experiments/00_workload.py
python experiments/01_concurrency_sweep.py
python experiments/02_breakeven.py
python experiments/03_stress_tests.py
```

Plots and per-point results are written under `results/`.

### Real-engine experiments

Install [Ollama](https://ollama.com) and pull the models before running these scripts:

```bash
ollama pull qwen2.5:0.5b
ollama pull moondream  # only needed for the image experiment

python calibration/01_prefill_and_cache.py
python calibration/02_vision.py
python calibration/03_queueing.py
python calibration/04_routing_real.py
python calibration/05_sim_regimes.py
```

The scripts start their own Ollama server processes on separate ports. The full routing run is longer than the others; use `python calibration/04_routing_real.py --smoke` for a quick check. Actual timings will vary with hardware and system load. Calibration artifacts are stored under `calibration/results/`.

## Project structure

```text
kvsim/         workload, cache, worker, router, simulator, and metrics code
experiments/   simulation sweeps and plots
calibration/   Ollama benchmarking and simulator replay
results/       simulation figures and JSON results
tests/         cache, routing, and simulator checks
```

## Scope

This models the **prefill stage** of multimodal serving, not a complete encoder/prefill/decode stack. It does not include continuous batching, chunked prefill, decode scheduling, KV transfers, offload tiers, worker failures, or delayed cache-state updates. The synthetic request distribution and prefill rates also affect the results.

A natural next step would be to run the same routing harness against independent vLLM workers on GPUs, with actual TTFT and prefix-cache metrics, and test how the policies behave when cache-state information is stale.
