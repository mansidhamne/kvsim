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
