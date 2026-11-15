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
from deploy.validate import print_report, validate_run
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
    print(f"  sample reply (fp16 sanity): {sm.get('sanity_reply', '')!r}")
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
        smoke_report = validate_run(sm, None, None)
        print_report(smoke_report)
        if not smoke_report["ok"] and not a.force:
            print("\nSmoke check failed; stopping. Fix the issue (see smoke.json and logs/) "
                  "or rerun with --force to calibrate anyway.")
            return 1
        if a.smoke_only:
            return 0

        print("\nCalibration")
        only = tuple(a.experiments.split(","))
        raw = asyncio.run(calibrate(url, cfg, pool, only=only))
        save(raw, out / "calibration_raw.json")

    fitted = fit(raw, block_size=serve.block_size)
    fitted["env"] = {k: info.get(k) for k in ("nvidia_smi", "vllm", "python", "time")}
    report = validate_run(sm, raw, fitted, block_size=serve.block_size)
    save(report, out / "validation.json")
    fitted["valid"] = report["ok"]
    fitted["validation_counts"] = report["counts"]
    save(fitted, out / "calibration_vllm.json")
    plot(raw, fitted, out / "calibration_vllm.png")
    cm, f = fitted["cost_model"], fitted["fit"]
    print(f"\nprefill: {cm['prefill_tokens_per_s'] or 0:.0f} tok/s linear part, attention "
          f"{cm['attn_s_per_token_pair']:.2e} s/tok^2, overhead {cm['fixed_overhead_s'] * 1e3:.0f} ms, "
          f"image cost x{cm['image_cost_multiplier'] or 0:.2f}, R2 {f['r2']:.4f} "
          f"(linear-only R2 {f['linear_only']['r2']:.3f})")
    print("image tokens per image:", fitted["image_tokens_per_image"])
    for k, v in fitted["hits"].items():
        print(f"  {k:28s} TTFT {v['ttft_mean_s'] * 1e3:7.1f} ms   cached {v['cached_frac']:.0%}")
    for r in fitted["load"]:
        print(f"  concurrency {r['concurrency']:3d}: TTFT p50 {r['ttft_p50_s'] or 0:.3f}s "
              f"p99 {r['ttft_p99_s'] or 0:.3f}s  ITL p50 {r['itl_p50_s'] or 0:.4f}s  "
              f"{r['req_per_s'] or 0:.2f} req/s")
    print_report(report)
    print(f"\nResults in {out}")
    return 0 if report["ok"] else 2


def phase3_cmd(a: argparse.Namespace) -> int:
    from bench.workload import RequestOptions
    from deploy.phase3 import Phase3Config, run_phase3
    from deploy.validate import validate_phase3
    from kvsim.config import CostModel

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cost = CostModel.from_calibration(a.cost)
    pool = ImagePool(a.images) if a.images else ImagePool()
    cfg = Phase3Config(policies=tuple(a.policies.split(",")),
                       active_convs=tuple(int(x) for x in a.convs.split(",")),
                       duration_s=a.duration, warmup_s=a.warmup, seed=a.seed,
                       time_scale=a.time_scale, slo_ttft_s=a.slo_ttft, slo_itl_s=a.slo_itl)
    if a.quick:
        cfg.trace_overrides = {"think_mean_s": 4.0, "pool_size": 16}
    info = env_info()
    info.update({"phase": 3, "config": cfg.__dict__, "cost_model": cost.__dict__,
                 "images": a.images or "synthetic"})
    n_gpus = max(1, a.gpus)
    per_gpu = max(1, -(-a.workers // n_gpus))
    serve = ServeConfig(model=a.model, gpu_memory_utilization=round(0.9 / per_gpu - 0.02, 2),
                        max_model_len=a.max_model_len, attention_backend=a.attention_backend)
    if not a.external:
        info["serve_config"] = serve.__dict__
        info["workers"] = [{"port": 8001 + i, "gpu": i % n_gpus} for i in range(a.workers)]
    save(info, out / "env.json")
    print(json.dumps({k: info[k] for k in ("nvidia_smi", "vllm", "config")}, indent=2, default=str))

    if a.external:
        ctx = nullcontext(None)
    else:
        specs = [WorkerSpec(port=8001 + i, gpu=i % n_gpus) for i in range(a.workers)]
        ctx = VllmWorkers(specs, serve, out / "logs")
    with ctx as ws:
        urls = a.external.split(",") if a.external else ws.urls
        print("workers:", urls)
        # every worker must answer the sanity question before the experiment counts
        for u in urls:
            sm = asyncio.run(smoke(u, CalibConfig.quick() if a.quick else CalibConfig(), pool))
            if "paris" not in (sm.get("sanity_reply") or "").lower() or not sm["ok"]:
                save(sm, out / "smoke_failed.json")
                print(f"worker {u} failed the smoke check: {sm.get('sanity_reply')!r}, {sm['checks']}")
                return 1
        print("all workers passed the smoke check")
        opts = RequestOptions(model="mock" if a.external else "model")
        title = (f"Routing policies, {len(urls)} mock workers" if a.external else
                 f"Routing policies, {len(urls)} vLLM workers ({a.model.split('/')[-1]}, {n_gpus}x GPU)")
        cells = asyncio.run(run_phase3(urls, cfg, out, cost, pool, opts, title=title))

    report = validate_phase3(cells, cost.block_size)
    save(report, out / "validation.json")
    print_report(report)
    print(f"\nResults in {out}")
    return 0 if report["ok"] else 2


def _fetch_inputs(a: argparse.Namespace, out: Path) -> dict:
    """Download trace sources on the GPU machine (never committed). VisionArena shards come from
    Hugging Face (terms: no redistribution), Mooncake from its GitHub release (Apache-2.0)."""
    got: dict = {}
    if a.workload == "visionarena" and not a.va_paths:
        from huggingface_hub import hf_hub_download
        paths = []
        for i in [int(x) for x in a.va_shards.split(",")]:
            fn = f"data/train-{i:05d}-of-00043.parquet"
            print(f"downloading VisionArena-Chat {fn} (~2 GB)", flush=True)
            paths.append(hf_hub_download("lmarena-ai/VisionArena-Chat", fn, repo_type="dataset",
                                         local_dir=str(out.parent / "va_data")))
        got["va_paths"] = paths
    elif a.va_paths:
        got["va_paths"] = a.va_paths.split(",")
    if a.workload == "mooncake":
        path = a.mooncake_path or str(out.parent / "mooncake_conversation_trace.jsonl")
        if not Path(path).exists():
            import urllib.request
            url = ("https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/"
                   "FAST25-release/traces/conversation_trace.jsonl")
            print(f"downloading {url} (~3 MB)", flush=True)
            urllib.request.urlretrieve(url, path)
        got["mooncake_path"] = path
    return got


def _cache_blocks(label: str, live_blocks: int) -> int | None:
    """'full' -> None (vLLM default); '2000' -> 2000 blocks; 'x0.25' -> 25% of the trace's
    peak live working set (at least 512 blocks so running requests still fit)."""
    if label == "full":
        return None
    if label.startswith("x"):
        return max(512, int(float(label[1:]) * live_blocks))
    return int(label)


def phase3b_cmd(a: argparse.Namespace) -> int:
    from bench.analyze import aggregate, paired, plot_aggregate
    from bench.traces.stats import workload_stats
    from bench.workload import RequestOptions
    from deploy.phase3b import Phase3bConfig, TraceFactory, run_session
    from deploy.validate import validate_phase3b
    from kvsim.config import CostModel

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cost = CostModel.from_calibration(a.cost)
    inputs = _fetch_inputs(a, out)
    va_over = {"think_scale": a.think_scale, "system_text": a.system_text,
               # exact length filter with the served model's tokenizer (already downloaded by vLLM)
               "tokenizer": "" if a.external else a.model}
    cfg = Phase3bConfig(
        workload=a.workload, loads=tuple(float(x) for x in a.loads.split(",")),
        seeds=tuple(int(x) for x in a.seeds.split(",")), policies=tuple(a.policies.split(",")),
        duration_s=a.duration, warmup_s=a.warmup, session_warmup_s=a.session_warmup,
        time_scale=a.time_scale, slo_ttft_s=a.slo_ttft, slo_itl_s=a.slo_itl,
        va_paths=tuple(inputs.get("va_paths", ())), va_max_rows=a.va_max_rows, va_overrides=va_over,
        mooncake_path=inputs.get("mooncake_path", ""),
        synthetic_overrides={"think_mean_s": 4.0, "pool_size": 16} if a.quick else {})
    # size the scarce caches from the first trace's live working set
    tr, pool, _ = TraceFactory(cfg).build(cfg.loads[-1], cfg.seeds[0])
    ws = workload_stats(tr, pool, cost.block_size)
    caches = a.caches.split(",")
    info = env_info()
    info.update({"phase": "3b", "config": {k: v for k, v in cfg.__dict__.items() if k != "va_paths"},
                 "va_shards": a.va_shards if a.workload == "visionarena" else None,
                 "cost_model": cost.__dict__, "first_trace_stats": ws,
                 "caches": {c: _cache_blocks(c, ws["peak_live_blocks"]) for c in caches}})
    n_gpus = max(1, a.gpus)
    per_gpu = max(1, -(-a.workers // n_gpus))
    save(info, out / "env.json")
    print(json.dumps({k: info[k] for k in ("nvidia_smi", "vllm", "first_trace_stats", "caches")},
                     indent=2, default=str))
    opts = RequestOptions(model="mock" if a.external else "model")
    cells: list[dict] = []
    for label in (["external"] if a.external else caches):
        blocks = None if a.external else _cache_blocks(label, ws["peak_live_blocks"])
        serve = ServeConfig(model=a.model, gpu_memory_utilization=round(0.9 / per_gpu - 0.02, 2),
                            max_model_len=a.max_model_len, attention_backend=a.attention_backend,
                            num_gpu_blocks_override=blocks)
        ctx = nullcontext(None) if a.external else VllmWorkers(
            [WorkerSpec(port=8001 + i, gpu=i % n_gpus) for i in range(a.workers)], serve,
            out / "logs" / f"cache-{label}")
        print(f"\n=== cache {label}: {blocks or 'vLLM default'} blocks per worker ===", flush=True)
        with ctx as ws_:
            urls = a.external.split(",") if a.external else ws_.urls
            for u in urls:
                sm = asyncio.run(smoke(u, CalibConfig.quick(), ImagePool()))
                if "paris" not in (sm.get("sanity_reply") or "").lower() or not sm["ok"]:
                    save(sm, out / f"smoke_failed_{label}.json")
                    print(f"worker {u} failed the smoke check: {sm.get('sanity_reply')!r}")
                    return 1
            cells += asyncio.run(run_session(urls, cfg, out, cost, label, opts))
        (out / "cells.json").write_text(json.dumps(cells, indent=2))   # saved after each cache size

    rows = aggregate(cells)
    pairs = [p for m in ("ttft_p99_s", "goodput_frac", "hit_rate")
             for a_, b_ in (("cache_aware", "session_affinity"), ("cache_aware", "round_robin"),
                            ("session_affinity", "round_robin"), ("cache_aware", "least_requests"))
             for p in paired(cells, a_, b_, m)]
    save({"aggregate": rows, "paired": pairs}, out / "summary.json")
    plot_aggregate(rows, out / "phase3b.png", f"{a.workload}: median over seeds, bars = min-max")
    report = validate_phase3b(cells, cost.block_size)
    save(report, out / "validation.json")
    print_report(report)
    print(f"\nResults in {out}")
    return 0 if report["ok"] else 2


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
    p2.add_argument("--experiments", default="text_cold,image_cold,hits,load",
                    help="comma-separated subset to run (e.g. skip the 5-minute load sweep)")
    p2.add_argument("--force", action="store_true", help="calibrate even if the smoke check fails")
    p3 = sub.add_parser("phase3", help="all routing policies on the same trace, through the router")
    p3.add_argument("--out", required=True)
    p3.add_argument("--external", help="comma-separated running workers (e.g. mocks) instead of launching vLLM")
    p3.add_argument("--workers", type=int, default=2, help="vLLM workers to launch")
    p3.add_argument("--gpus", type=int, default=2, help="GPUs to spread workers over (round-robin)")
    p3.add_argument("--model", default=ServeConfig.model)
    p3.add_argument("--max-model-len", type=int, default=ServeConfig.max_model_len)
    p3.add_argument("--attention-backend", default=None)
    p3.add_argument("--cost", default="calibration/results/phase2/calibration_vllm_final.json")
    p3.add_argument("--images", help="folder of real photos (e.g. COCO val2017)")
    p3.add_argument("--policies", default=",".join(__import__("router.policies", fromlist=["POLICIES"]).POLICIES))
    p3.add_argument("--convs", default="12,24", help="comma-separated load levels (active conversations)")
    p3.add_argument("--duration", type=float, default=300.0, help="measured seconds per cell")
    p3.add_argument("--warmup", type=float, default=60.0)
    p3.add_argument("--seed", type=int, default=0)
    p3.add_argument("--slo-ttft", type=float, default=2.0)
    p3.add_argument("--slo-itl", type=float, default=0.1)
    p3.add_argument("--time-scale", type=float, default=1.0, help=">1 compresses the trace (mock only)")
    p3.add_argument("--quick", action="store_true", help="short think times (testing)")
    p3b = sub.add_parser("phase3b", help="audit-fixed routing experiment on real or synthetic traffic")
    p3b.add_argument("--out", required=True)
    p3b.add_argument("--workload", default="visionarena", choices=["synthetic", "visionarena", "mooncake"])
    p3b.add_argument("--loads", default="0.3", help="synthetic: active convs; visionarena: conv starts/s; "
                     "mooncake: time dilation")
    p3b.add_argument("--seeds", default="0,1,2")
    p3b.add_argument("--policies", default="round_robin,least_requests,session_affinity,cache_aware")
    p3b.add_argument("--caches", default="full", help="comma list: full | <blocks> | x<fraction of live set>")
    p3b.add_argument("--external", help="comma-separated running workers (mock) instead of vLLM")
    p3b.add_argument("--workers", type=int, default=2)
    p3b.add_argument("--gpus", type=int, default=2)
    p3b.add_argument("--model", default=ServeConfig.model)
    p3b.add_argument("--max-model-len", type=int, default=ServeConfig.max_model_len)
    p3b.add_argument("--attention-backend", default=None)
    p3b.add_argument("--cost", default="calibration/results/phase2/calibration_vllm_final.json")
    p3b.add_argument("--va-shards", default="0", help="VisionArena shard indices to download (0-42)")
    p3b.add_argument("--va-paths", default="", help="local VisionArena parquet files instead")
    p3b.add_argument("--va-max-rows", type=int, default=5000)
    p3b.add_argument("--mooncake-path", default="")
    p3b.add_argument("--think-scale", type=float, default=0.1, help="x ServeGen think times")
    p3b.add_argument("--system-text", default="", help="system prompt (VisionArena has none)")
    p3b.add_argument("--duration", type=float, default=300.0)
    p3b.add_argument("--warmup", type=float, default=60.0)
    p3b.add_argument("--session-warmup", type=float, default=90.0)
    p3b.add_argument("--slo-ttft", type=float, default=2.0)
    p3b.add_argument("--slo-itl", type=float, default=0.1)
    p3b.add_argument("--time-scale", type=float, default=1.0)
    p3b.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.cmd == "phase3b":
        sys.exit(phase3b_cmd(a))
    if a.cmd == "phase3":
        sys.exit(phase3_cmd(a))
    if a.cmd == "phase2":
        sys.exit(phase2(a))


if __name__ == "__main__":
    main()
