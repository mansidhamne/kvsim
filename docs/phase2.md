# Phase 2: vLLM on a Kaggle T4

Goal: confirm the pinned vLLM (0.32.0) serves a small VLM on a free T4, answer the open
questions in [vllm_api.md](vllm_api.md), and replace the laptop cost model with one measured on vLLM.

## Run it on Kaggle

1. Kaggle → *Create* → *New Notebook* → *File* → *Import Notebook* → upload
   [`deploy/kaggle_phase2.ipynb`](../deploy/kaggle_phase2.ipynb).
2. Sidebar: *Accelerator* = **GPU T4 x2**, **Internet on** (one-time phone verification).
   Optional: *Add Input* → a "COCO 2017" dataset with a `val2017/` folder (real photos).
3. *Run All*. Expect 25-35 minutes; most of it is installing vLLM and downloading the model.
4. Download `phase2.zip` from the Output panel and unzip it into `calibration/results/vllm_t4/`.

## What it does

`python -m deploy.kaggle_run phase2 --out DIR` launches one `vllm serve` worker on GPU 0
(flags in [`deploy/workers.py`](../deploy/workers.py)), then:

| step | what | output |
|---|---|---|
| environment | GPU, Python, vLLM/torch/CUDA versions, attention-backend lines from the log | `env.json` |
| smoke check | text request; image request with a `uuid`; image tokens per size; same image + new question hits the cache; metric names on `/metrics` | `smoke.json` |
| A. text_cold | cold prompts, 64-4096 words, 3 reps | `calibration_raw.json` |
| B. image_cold | 1/2/4 never-seen images at 448x448 and 672x448 | |
| C. hits | long prefix: cold, then new question, then identical resend; same for 4 images | |
| D. load | closed loop, 1-32 concurrent clients, 45 s each, 64 output tokens | |
| fit | `ttft = overhead + text/rate + m * image/rate` (least squares) | `calibration_vllm.json` (loads with `CostModel.from_calibration`) |
| plot | prefill fit, cold vs hit, latency vs load | `calibration_vllm.png` |

Only a broken request path (text or image request failing) stops the run. Everything else is
recorded as an answer (WARN), because "the metric is named differently" is a finding, not a failure.

## Test it on the Mac first

```bash
python -m bench.mock_worker --n 1 --base-port 8001 --time-scale 10 &
python -m deploy.kaggle_run phase2 --external http://127.0.0.1:8001 --quick --out results/phase2_mock
```

## If something fails on Kaggle

| symptom | try |
|---|---|
| `pip install vllm` conflicts or Python too old | check `python --version` (needs 3.11+); restart the kernel after install |
| worker exits during startup, log mentions attention / FlashAttention / kernel | rerun with `--attention-backend TRITON_ATTN` |
| out of memory at startup | `--gpu-mem 0.85` or `--max-model-len 4096` |
| bf16 error | should not happen (`--dtype float16` is set); check the model's config |
| image request returns 400 mentioning `uuid` | the field is rejected: note it, rerun loadgen with `--no-image-uuids` |

## Validation: when are the numbers usable?

Every run ends with `validation.json` and a printed report ([`deploy/validate.py`](../deploy/validate.py)).
`calibration_vllm.json` carries `"valid": true/false`, and the command exits with code 2 when
anything FAILs. **Only numbers from a VALID run go into the README or a resume.**

| step | FAIL when |
|---|---|
| smoke | text or image request broken; the model doesn't answer "Paris" to "What is the capital of France?" (fp16 sanity); an image description isn't readable text |
| every experiment | any request error; any request without a TTFT; no usage details from the server; output length differs from what was requested |
| text_cold / image_cold | a "cold" request hit the cache; prompt tokens don't grow with prompt length; image tokens ≠ images × tokens per image |
| hits | the cold step hit the cache; a new question reused < 90% of the prompt; a hit wasn't faster than cold |
| load | failed requests; missing TTFT; wrong output length; prompts weren't cold; degenerate output text (e.g. `!!!!!!`, one word repeated) |
| fit | R² < 0.98; a coefficient significantly negative; fewer than 80% of image requests measured |

WARN (reported, not blocking): repeats differ by more than 1.3×, throughput drops as load rises,
the server's cache counters disagree with the per-request numbers by more than 5%.

### Runs

| run | result | files |
|---|---|---|
| 1st (2026-11-14 03:27) | NOT VALID: 10/18 image and 9/27 hit requests had no TTFT (first token decoded to ""), no fp16 check | git history, commit `123439b` |
| 2nd (2026-11-14 04:07) | **VALID** (42 pass, 5 warn, 0 fail) | `calibration/results/phase2/` |

Two problems found in the 2nd run and fixed in code for later runs:

* **One contaminated row.** The first cold 448x448 one-image request reused the smoke check's
  image uuid, so vLLM's vision-encoder cache served it: 73 ms vs 192-193 ms for its repeats
  (caught by the "repeats agree" warning). It is excluded from `calibration_vllm_final.json`
  (listed under `excluded_rows`). The calibration now draws images from a uuid range that never
  overlaps the smoke check's, and a FAIL check verifies it.
* **Server cache counters not read** (WARN). vLLM exports counters with a `_total` suffix
  (`vllm:prefix_cache_hits_total`); the parser now accepts both spellings. The server-vs-client
  cross-check runs from the next GPU run on.

## Results (Qwen2-VL-2B, fp16, 1x T4, vLLM 0.32.0, TRITON_ATTN)

Use `calibration/results/phase2/calibration_vllm_final.json`.

**Prefill cost** (66 requests, R² 0.999; a straight line only reaches R² 0.954):

    ttft = 43 ms + n / 11,100 tok/s + 3.2e-7 s * n * (cached + n/2)  [+ 0.45 ms per new image token]

| cold prompt | 124 tok | 1,080 tok | 2,105 tok | 4,154 tok |
|---|---|---|---|---|
| TTFT | 36 ms | 0.33 s | 0.94 s | 3.13 s |

Attention is quadratic: at 4k tokens it is ~2.8 s of the 3.1 s.

**Images:** 256 tokens at 448x448, 384 at 672x448 (Qwen2-VL: one token per 28x28 pixels).
An image token costs ~5.9x a text token (the vision encoder): about 115 ms per 448x448 image and
170 ms per 672x448 image, linear in the number of images.

**Cache hits** (mean of 3):

| prompt | cold | new question, prefix cached | speed-up |
|---|---|---|---|
| 1,024-word text prefix (~1,080 tok) | 0.34 s | 48 ms | 7x |
| 4,096-word text prefix (~4,150 tok) | 3.25 s | 95 ms | 34x |
| 4 images 672x448 (~1,590 tok) | 1.27 s | 60 ms | 21x |

A hit is not free: new tokens still attend to the cached prefix (95 ms with 4k cached vs 48 ms with 1k).

**Load** (closed loop, ~1,040-token cold prompts, 64 output tokens):

| concurrent | req/s | output tok/s | TTFT p50 | TTFT p99 | ITL p50 |
|---|---|---|---|---|---|
| 1 | 0.73 | 47 | 0.32 s | 0.33 s | 16.7 ms |
| 4 | 1.66 | 106 | 0.90 s | 1.22 s | 18.9 ms |
| 8 | 2.11 | 135 | 0.94 s | 2.38 s | 22.6 ms |
| 16 | 2.47 | 158 | 1.70 s | 4.65 s | 29.5 ms |
| 32 | 2.71 | 174 | 2.04 s | 9.23 s | 46.0 ms |

Throughput saturates near 2.7 req/s; beyond that, extra load becomes queueing (TTFT p99) and
slower decoding for everyone (ITL up 2.8x, because prefills share the GPU with decodes). The load
numbers reproduced the 1st run within 1% at every level.
