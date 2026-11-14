"""Start and stop real `vllm serve` workers (one process per worker, pinned to a GPU).

Flags come from docs/vllm_api.md (vLLM 0.32.0). Every worker uses the same hash settings so
a router can reproduce block hashes: sha256_cbor + a shared PYTHONHASHSEED + a fixed block size.

    with VllmWorkers([WorkerSpec(port=8001, gpu=0)], ServeConfig(), log_dir) as ws:
        ws.urls  # ['http://127.0.0.1:8001']
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServeConfig:
    model: str = "Qwen/Qwen2-VL-2B-Instruct"
    dtype: str = "float16"                 # T4 has no bf16
    block_size: int = 16
    hash_algo: str = "sha256_cbor"
    hash_seed: str = "0"                   # PYTHONHASHSEED, identical on every worker
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    num_gpu_blocks_override: int | None = None
    max_images: int = 8
    attention_backend: str | None = None   # e.g. "TRITON_ATTN" if the default fails on T4
    kv_events: bool = False                # Phase 4
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerSpec:
    port: int
    gpu: int
    events_port: int | None = None         # ZMQ PUB port (Phase 4)


def serve_command(spec: WorkerSpec, cfg: ServeConfig) -> list[str]:
    cmd = [
        "vllm", "serve", cfg.model,
        "--host", "127.0.0.1", "--port", str(spec.port),
        "--dtype", cfg.dtype,
        "--block-size", str(cfg.block_size),
        "--prefix-caching-hash-algo", cfg.hash_algo,
        "--enable-prompt-tokens-details",           # usage.prompt_tokens_details.cached_tokens
        "--max-model-len", str(cfg.max_model_len),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--limit-mm-per-prompt", json.dumps({"image": cfg.max_images}),
        "--served-model-name", "model",              # requests use model="model"
    ]                                                # (request logging is off by default in 0.32)
    if cfg.num_gpu_blocks_override:
        cmd += ["--num-gpu-blocks-override", str(cfg.num_gpu_blocks_override)]
    if cfg.attention_backend:
        cmd += ["--attention-backend", cfg.attention_backend]
    if cfg.kv_events and spec.events_port:
        cmd += ["--kv-events-config", json.dumps({
            "enable_kv_cache_events": True, "publisher": "zmq",
            "endpoint": f"tcp://*:{spec.events_port}",
            "replay_endpoint": f"tcp://*:{spec.events_port + 1}", "topic": "kv-events"})]
    return cmd + list(cfg.extra_args)


def _get(url: str, timeout: float = 2.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


@dataclass
class _Proc:
    spec: WorkerSpec
    popen: subprocess.Popen
    log_path: Path
    url: str


@dataclass
class VllmWorkers:
    specs: list[WorkerSpec]
    cfg: ServeConfig = field(default_factory=ServeConfig)
    log_dir: Path = Path("logs")
    startup_timeout_s: float = 900.0       # first start downloads the model
    procs: list[_Proc] = field(default_factory=list)

    @property
    def urls(self) -> list[str]:
        return [p.url for p in self.procs]

    def __enter__(self) -> "VllmWorkers":
        if shutil.which("vllm") is None:
            raise RuntimeError("`vllm` not found on PATH; pip install vllm==0.32.0")
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for s in self.specs:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(s.gpu), "VLLM_SERVER_DEV_MODE": "1",
                   "PYTHONHASHSEED": self.cfg.hash_seed}
            log = self.log_dir / f"vllm_{s.port}.log"
            fh = log.open("w")
            cmd = serve_command(s, self.cfg)
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
            self.procs.append(_Proc(s, p, log, f"http://127.0.0.1:{s.port}"))
        try:
            self._wait_ready()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        pending = list(self.procs)
        while pending:
            for p in list(pending):
                if p.popen.poll() is not None:
                    raise RuntimeError(f"worker on port {p.spec.port} exited during startup:\n"
                                       + tail(p.log_path, 40))
                if _get(p.url + "/health")[0] == 200:
                    pending.remove(p)
            if time.time() > deadline:
                raise TimeoutError("workers not healthy in time:\n"
                                   + "\n".join(tail(p.log_path, 20) for p in pending))
            time.sleep(2.0)

    def __exit__(self, *exc) -> None:
        for p in self.procs:
            if p.popen.poll() is None:
                try:
                    os.killpg(p.popen.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in self.procs:
            try:
                p.popen.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.popen.pid, signal.SIGKILL)
        self.procs = []


def tail(path: Path, n: int) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def grep_log(path: Path, *needles: str, limit: int = 20) -> list[str]:
    """Log lines containing any needle (case-insensitive), e.g. the chosen attention backend."""
    out = []
    try:
        for line in Path(path).read_text(errors="replace").splitlines():
            low = line.lower()
            if any(n.lower() in low for n in needles):
                out.append(line.strip()[:300])
                if len(out) >= limit:
                    break
    except OSError:
        pass
    return out
