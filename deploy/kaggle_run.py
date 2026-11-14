"""Phase 2 entry point: one command per GPU session.

    # on Kaggle (launches vLLM on GPU 0, then smoke check + calibration + fit + plot)
    python -m deploy.kaggle_run phase2 --out /kaggle/working/phase2

    # on the Mac, against mock workers (tests the whole pipeline in seconds)
    python -m bench.mock_worker --n 1 --base-port 8001 --time-scale 10 &
    python -m deploy.kaggle_run phase2 --external http://127.0.0.1:8001 --quick --out results/phase2_mock

Outputs in --out:
  env.json            GPU, Python, vLLM version, chosen attention backend (from the log)
  smoke.json          answers to the open questions in docs/vllm_api.md
  calibration_raw.json, calibration_vllm.json (fitted cost model), calibration_vllm.png
  logs/vllm_<port>.log
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from bench.workload import ImagePool
from deploy.calibrate import CalibConfig, calibrate, fit, plot, save, smoke, vllm_backend_lines
from deploy.workers import ServeConfig, VllmWorkers, WorkerSpec


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi"] = f"unavailable: {e}"
    try:
        import vllm  # type: ignore
        info["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        info["vllm"] = f"not importable: {type(e).__name__}"
    try:
        import torch  # type: ignore
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return info


def _print_checks(sm: dict) -> None:
    print("\nSmoke check")
    for k, v in sm["checks"].items():
        label = "PASS" if v else ("FAIL" if k in sm.get("blocking", []) else "WARN")
        print(f"  {label}  {k}")
    for size, v in sm.get("images", {}).items():
        print(f"  image {size}: {v['image_tokens']} tokens; second question cached "
              f"{v['second'].get('cached_tokens')} of {v['second'].get('prompt_tokens')}")
    missing = [m for m, ok in sm.get("metrics_present", {}).items() if not ok]
    if missing:
        print("  missing metrics:", ", ".join(missing))


def phase2(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CalibConfig.quick() if a.quick else CalibConfig()
    pool = ImagePool(a.images) if a.images else ImagePool()
    serve = ServeConfig(model=a.model, attention_backend=a.attention_backend,
                        max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem)
    info = env_info()
    info["serve_config"] = serve.__dict__ if not a.external else {"external": a.external}
    info["images"] = a.images or "synthetic"
    save(info, out / "env.json")
    print(json.dumps(info, indent=2, default=str))

    workers = (nullcontext(None) if a.external else
               VllmWorkers([WorkerSpec(port=a.port, gpu=a.gpu)], serve, out / "logs"))
    with workers as ws:
        url = a.external or ws.urls[0]
        if ws is not None:
            info["backend_log_lines"] = vllm_backend_lines(ws.procs[0].log_path)
            save(info, out / "env.json")
            print("\n".join(info["backend_log_lines"][:10]))

        sm = asyncio.run(smoke(url, cfg, pool))
        save(sm, out / "smoke.json")
        _print_checks(sm)
        if not sm["ok"] and not a.force:
            print("\nSmoke check failed; stopping. Fix the issue (see smoke.json and logs/) "
                  "or rerun with --force to calibrate anyway.")
            return 1
        if a.smoke_only:
            return 0

        print("\nCalibration")
        raw = asyncio.run(calibrate(url, cfg, pool))
        save(raw, out / "calibration_raw.json")

    fitted = fit(raw, block_size=serve.block_size)
    fitted["env"] = {k: info.get(k) for k in ("nvidia_smi", "vllm", "python", "time")}
    save(fitted, out / "calibration_vllm.json")
    plot(raw, fitted, out / "calibration_vllm.png")
    cm, f = fitted["cost_model"], fitted["fit"]
    print(f"\nprefill: {cm['prefill_tokens_per_s'] or 0:.0f} tok/s, overhead {cm['fixed_overhead_s'] * 1e3:.0f} ms, "
          f"image cost x{cm['image_cost_multiplier'] or 0:.2f}, R2 {f['r2']:.3f}")
    print("image tokens per image:", fitted["image_tokens_per_image"])
    for k, v in fitted["hits"].items():
        print(f"  {k:28s} TTFT {v['ttft_mean_s'] * 1e3:7.1f} ms   cached {v['cached_frac']:.0%}")
    for r in fitted["load"]:
        print(f"  concurrency {r['concurrency']:3d}: TTFT p50 {r['ttft_p50_s'] or 0:.3f}s "
              f"p99 {r['ttft_p99_s'] or 0:.3f}s  ITL p50 {r['itl_p50_s'] or 0:.4f}s  "
              f"{r['req_per_s'] or 0:.2f} req/s")
    print(f"\nResults in {out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p2 = sub.add_parser("phase2", help="smoke check + calibration on one worker")
    p2.add_argument("--out", required=True)
    p2.add_argument("--external", help="use an already running server (e.g. a mock) instead of launching vLLM")
    p2.add_argument("--model", default=ServeConfig.model)
    p2.add_argument("--gpu", type=int, default=0)
    p2.add_argument("--port", type=int, default=8001)
    p2.add_argument("--max-model-len", type=int, default=ServeConfig.max_model_len)
    p2.add_argument("--gpu-mem", type=float, default=ServeConfig.gpu_memory_utilization)
    p2.add_argument("--attention-backend", default=None, help="e.g. TRITON_ATTN if the default fails")
    p2.add_argument("--images", help="folder of real photos (e.g. COCO val2017); default synthetic")
    p2.add_argument("--quick", action="store_true", help="tiny experiment sizes (for testing)")
    p2.add_argument("--smoke-only", action="store_true")
    p2.add_argument("--force", action="store_true", help="calibrate even if the smoke check fails")
    a = ap.parse_args()
    if a.cmd == "phase2":
        sys.exit(phase2(a))


if __name__ == "__main__":
    main()
