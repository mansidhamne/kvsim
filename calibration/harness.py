"""Shared plumbing for the calibration experiments (Ollama / llama.cpp on a laptop).

Two measurement channels, because neither is enough alone:

* CLIENT wall-clock around a request with num_predict=1  ->  TTFT proxy
  (what a user feels: queueing + prefill + one decode step + HTTP).
* SERVER log (llama-server, which Ollama wraps)           ->  exact cache accounting
  per task: total prompt tokens, tokens served from cache, tokens actually evaluated.
  The HTTP API does NOT expose this: `prompt_eval_count` includes cached tokens
  (verified: an identical re-send reports the same count in ~1/20 of the time).

We spawn our own `ollama serve` processes on private ports so that nothing depends on
the state of the desktop app and so that env settings (parallelism, cache budget) are
controlled and recorded.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

TEXT_MODEL = "qwen2.5:0.5b"
VISION_MODEL = "moondream"

# ------------------------------------------------------------------ deterministic text
_SYL = ["ka", "lo", "mi", "ra", "su", "te", "vo", "ni", "pa", "de", "bu", "fi"]


def words(n: int, seed: int) -> str:
    """n pseudo-words, deterministic in `seed`. Same (n, seed) -> identical text, so identical
    prefixes really are identical token prefixes. ~1.8 tokens/word on Qwen's tokenizer."""
    r = random.Random(seed)
    return " ".join("".join(r.choice(_SYL) for _ in range(r.randint(1, 3))) for _ in range(n))


def noise_png_b64(seed: int, size: int = 378) -> str:
    """A unique random-noise PNG (unique so no image-level cache can hit)."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from matplotlib.image import imsave
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    imsave(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ server log parsing
_NEW = re.compile(r"task (\d+) \| new prompt, .*task\.n_tokens = (\d+)")
_CACHED = re.compile(r"task (\d+) \| cached n_tokens = (\d+)")
_TIMING = re.compile(r"print_timing: id\s+\d+ \| task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
_LIMITS = re.compile(r"prompt cache is enabled, size limit: ([\d.]+) MiB")


@dataclass
class TaskInfo:
    task: int
    n_tokens: int                 # total prompt tokens
    cached: Optional[int] = None  # tokens reused from a cache (first 'cached n_tokens' line)
    evaluated: Optional[int] = None   # tokens actually run through the model
    prompt_ms: Optional[float] = None

    @property
    def complete(self) -> bool:
        return self.evaluated is not None and self.cached is not None


def parse_tasks(text: str) -> list[TaskInfo]:
    tasks: dict[int, TaskInfo] = {}
    for line in text.splitlines():
        m = _NEW.search(line)
        if m:
            tasks[int(m.group(1))] = TaskInfo(int(m.group(1)), int(m.group(2)))
            continue
        m = _CACHED.search(line)
        if m and int(m.group(1)) in tasks and tasks[int(m.group(1))].cached is None:
            tasks[int(m.group(1))].cached = int(m.group(2))
            continue
        m = _TIMING.search(line)
        if m and int(m.group(1)) in tasks:
            t = tasks[int(m.group(1))]
            t.prompt_ms, t.evaluated = float(m.group(2)), int(m.group(3))
    return [t for _, t in sorted(tasks.items()) if t.n_tokens > 1]   # drop the 1-token warmup


# ------------------------------------------------------------------ server + client
@dataclass
class GenResult:
    wall_s: float
    api_prompt_tokens: int
    load_ms: float
    task: Optional[TaskInfo]      # exact accounting from the log (None if not found in time)


class OllamaServer:
    def __init__(self, port: int, num_parallel: int = 1, cache_ram_mib: Optional[int] = None,
                 num_ctx: int = 4096, tag: str = "srv"):
        self.port, self.num_ctx = port, num_ctx
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = RESULTS / f"{tag}_{port}.log"
        self.env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}",
                        OLLAMA_NUM_PARALLEL=str(num_parallel), OLLAMA_KEEP_ALIVE="2h",
                        OLLAMA_MAX_LOADED_MODELS="1")
        if cache_ram_mib is not None:
            self.env["LLAMA_ARG_CACHE_RAM"] = str(cache_ram_mib)   # llama.cpp --cache-ram (MiB)
        self.num_parallel, self.cache_ram_mib = num_parallel, cache_ram_mib
        self._proc: Optional[subprocess.Popen] = None
        self._pos = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        exe = shutil.which("ollama") or "/usr/local/bin/ollama"
        self._logf = open(self.log_path, "w")
        self._proc = subprocess.Popen([exe, "serve"], env=self.env, stdout=self._logf,
                                      stderr=subprocess.STDOUT)
        for _ in range(120):
            try:
                urllib.request.urlopen(self.base + "/api/version", timeout=1)
                return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError(f"ollama did not start on port {self.port}; see {self.log_path}")

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._logf.close()
            self._proc = None

    def version(self) -> str:
        return json.load(urllib.request.urlopen(self.base + "/api/version"))["version"]

    # ---- requests
    def _post(self, body: dict, timeout: float = 600) -> dict:
        req = urllib.request.Request(self.base + "/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))

    def generate(self, prompt: str, model: str = TEXT_MODEL, images: Optional[list[str]] = None,
                 raw: bool = True, num_predict: int = 1, read_log: bool = True) -> GenResult:
        body = {"model": model, "prompt": prompt, "raw": raw, "stream": False, "keep_alive": "2h",
                "options": {"num_predict": num_predict, "temperature": 0, "num_ctx": self.num_ctx}}
        if images:
            body["images"] = images
        mark = self._log_size() if read_log else 0
        t = time.perf_counter()
        r = self._post(body)
        wall = time.perf_counter() - t
        task = self._wait_task(mark) if read_log else None
        return GenResult(wall, r.get("prompt_eval_count", 0), r.get("load_duration", 0) / 1e6, task)

    def warmup(self, model: str = TEXT_MODEL, n: int = 4, **kw):
        """First requests after a model load are several x slower (GPU/kernel warm-up)."""
        for i in range(n):
            self.generate(words(60 + 120 * i, 424200 + i), model=model, read_log=False, **kw)

    # ---- log access
    def _log_size(self) -> int:
        self._logf.flush()
        return self.log_path.stat().st_size

    def _read_from(self, pos: int) -> str:
        with open(self.log_path, "rb") as f:
            f.seek(pos)
            return f.read().decode(errors="replace")

    def _wait_task(self, mark: int, timeout: float = 3.0) -> Optional[TaskInfo]:
        end = time.time() + timeout
        while time.time() < end:
            done = [t for t in parse_tasks(self._read_from(mark)) if t.complete]
            if done:
                return done[-1]
            time.sleep(0.02)
        return None

    def all_tasks(self) -> list[TaskInfo]:
        self._logf.flush()
        return parse_tasks(self._read_from(0))

    def log_text(self) -> str:
        self._logf.flush()
        return self._read_from(0)


def system_info(server: OllamaServer) -> dict:
    import platform
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "ollama_server_version": server.version(), "text_model": TEXT_MODEL,
            "num_parallel": server.num_parallel, "num_ctx": server.num_ctx,
            "cache_ram_mib": server.cache_ram_mib}
    return info
