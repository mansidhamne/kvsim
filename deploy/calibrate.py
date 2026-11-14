"""Phase 2: smoke check + calibration measurements against ONE OpenAI-compatible worker.

Works against a real vLLM worker (on Kaggle) or a mock worker (on the Mac, for testing).

Smoke check answers the open questions in docs/vllm_api.md:
  * does a text request work; does an image request with a `uuid` field work
  * image token count per image size (usage.prompt_tokens_details.multimodal_tokens)
  * does a second question about the same image hit the cache (cached_tokens > 0)
  * which metric names exist on /metrics

Calibration experiments (all requests use max_tokens=1 except the load sweep):
  A. text_cold   cold prompts of increasing length            -> prefill speed
  B. image_cold  cold prompts with 1/2/4 images per size       -> cost of an image token
  C. hits        long prefix cold, then +new question, then identical resend (text and images)
  D. load        closed-loop concurrency sweep                 -> TTFT/ITL vs concurrent requests
Every cold request starts with a unique nonce, so nothing is shared with earlier requests
(not even the chat template's first block).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid as uuidlib
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import numpy as np

from bench.loadgen import Record, _send
from bench.workload import _WORDS, ImagePool, ImageRef

METRICS_WANTED = [
    "vllm:prefix_cache_queries", "vllm:prefix_cache_hits", "vllm:num_requests_running",
    "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
]


@dataclass(frozen=True)
class CalibConfig:
    model: str = "model"
    text_words: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)
    image_counts: tuple[int, ...] = (1, 2, 4)
    image_sizes: tuple[tuple[int, int], ...] = ((448, 448), (672, 448))
    hit_prefix_words: tuple[int, ...] = (1024, 4096)
    reps: int = 3
    load_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    load_words: int = 1000
    load_output_tokens: int = 64
    load_seconds: float = 45.0
    timeout_s: float = 300.0

    @classmethod
    def quick(cls) -> "CalibConfig":
        """Tiny version for tests against the mock worker."""
        return cls(text_words=(64, 256, 1024), image_counts=(1, 2), image_sizes=((448, 448),),
                   hit_prefix_words=(512,), reps=1, load_levels=(1, 4), load_words=200,
                   load_output_tokens=8, load_seconds=1.5)


# ------------------------------------------------------------------------------- prompts

class _Prompts:
    def __init__(self, seed: int = 0, pool: ImagePool | None = None):
        self.rng = random.Random(seed)
        self.pool = pool or ImagePool()
        self._next_image = 10_000            # pool indices for "never seen" images

    def words(self, n: int) -> str:
        return " ".join(self.rng.choice(_WORDS) for _ in range(n))

    def nonce(self) -> str:
        return uuidlib.UUID(int=self.rng.getrandbits(128)).hex

    def fresh_images(self, n: int, size: tuple[int, int]) -> list[ImageRef]:
        refs = []
        for _ in range(n):
            refs.append(ImageRef(self._next_image, *size))
            self._next_image += 1
        return refs

    def image_part(self, ref: ImageRef, salt: str = "") -> dict:
        return {"type": "image_url", "image_url": {"url": self.pool.data_url(ref)},
                "uuid": ref.uuid + salt}

    def messages(self, system: str, user_text: str, images: list[ImageRef] = (),
                 salt: str = "") -> list[dict]:
        content = [self.image_part(r, salt) for r in images] + [{"type": "text", "text": user_text}]
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _body(cfg: CalibConfig, messages: list[dict], max_tokens: int = 1) -> dict:
    return {"model": cfg.model, "messages": messages, "max_tokens": max_tokens,
            "min_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}


async def _one(session, url: str, cfg: CalibConfig, messages, max_tokens: int = 1, **tags) -> dict:
    rec = Record(-1, 0, url, None, 0.0, 0.0, 0.0, None, None)
    await _send(session, url, _body(cfg, messages, max_tokens), {}, rec, time.perf_counter(),
                cfg.timeout_s)
    d = asdict(rec)
    keep = {k: d[k] for k in ("ttft_s", "e2e_s", "prompt_tokens", "cached_tokens", "mm_tokens",
                              "output_chunks", "status", "error")}
    keep["itl_p50_s"] = float(np.median(rec.itl_s)) if rec.itl_s else None
    return {**tags, **keep}


async def _metrics(session, url: str) -> str:
    try:
        async with session.get(url.rstrip("/") + "/metrics") as r:
            return await r.text()
    except Exception as e:  # noqa: BLE001
        return f"# error {e}"


def parse_metric(text: str, name: str) -> float | None:
    """Sum of all samples of a counter/gauge (all label sets)."""
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                total += float(line.rsplit(" ", 1)[1])
                found = True
            except ValueError:
                pass
    return total if found else None


# ------------------------------------------------------------------------------- smoke

async def smoke(url: str, cfg: CalibConfig = CalibConfig(), pool: ImagePool | None = None) -> dict:
    P = _Prompts(seed=1, pool=pool)
    out: dict = {"url": url, "checks": {}}
    async with aiohttp.ClientSession() as s:
        sysmsg = "Smoke " + P.nonce()
        t = await _one(s, url, cfg, P.messages(sysmsg, "Say hello."), max_tokens=8, kind="text")
        out["text"] = t
        out["checks"]["text_request_ok"] = t["status"] == 200 and t["ttft_s"] is not None

        sizes = {}
        for size in cfg.image_sizes:
            ref = P.fresh_images(1, size)
            sysm = "Smoke " + P.nonce()
            first = await _one(s, url, cfg, P.messages(sysm, "Describe the image.", ref), 8,
                               kind="image", size=list(size))
            second = await _one(s, url, cfg, P.messages(sysm, "What colors are in it?", ref), 8,
                                kind="image_again", size=list(size))
            sizes[f"{size[0]}x{size[1]}"] = {"first": first, "second": second,
                                             "image_tokens": (first.get("mm_tokens") or {}).get("image")}
        out["images"] = sizes
        firsts = [v["first"] for v in sizes.values()]
        seconds = [v["second"] for v in sizes.values()]
        out["checks"]["image_with_uuid_ok"] = all(r["status"] == 200 for r in firsts)
        out["checks"]["image_token_counts_reported"] = all(v["image_tokens"] for v in sizes.values())
        out["checks"]["same_image_second_question_hits"] = all(
            (r.get("cached_tokens") or 0) > 0 for r in seconds)

        text = await _metrics(s, url)
        present = {m: (m in text) for m in METRICS_WANTED}
        out["metrics_present"] = present
        out["checks"]["all_metric_names_present"] = all(present.values())
    # Only a broken request path blocks calibration; the rest are answers to record.
    out["blocking"] = ["text_request_ok", "image_with_uuid_ok"]
    out["ok"] = all(out["checks"][k] for k in out["blocking"])
    return out


# ------------------------------------------------------------------------------- experiments

async def exp_text_cold(s, url, cfg, P) -> list[dict]:
    rows = []
    for n in cfg.text_words:
        for rep in range(cfg.reps):
            msgs = P.messages(P.nonce() + " " + P.words(16), P.words(n))
            rows.append(await _one(s, url, cfg, msgs, exp="text_cold", words=n, rep=rep))
    return rows


async def exp_image_cold(s, url, cfg, P) -> list[dict]:
    rows = []
    for size in cfg.image_sizes:
        for k in cfg.image_counts:
            for rep in range(cfg.reps):
                refs = P.fresh_images(k, size)
                msgs = P.messages(P.nonce(), "Describe these images briefly.", refs)
                rows.append(await _one(s, url, cfg, msgs, exp="image_cold", size=list(size),
                                       images=k, rep=rep))
    return rows


async def exp_hits(s, url, cfg, P) -> list[dict]:
    rows = []
    for n in cfg.hit_prefix_words:
        for rep in range(cfg.reps):
            sysm = P.nonce() + " " + P.words(n)
            q1, q2 = P.words(12), P.words(12)
            for step, q in (("cold", q1), ("new_question", q2), ("identical", q2)):
                rows.append(await _one(s, url, cfg, P.messages(sysm, q), exp="hits_text",
                                       prefix_words=n, rep=rep, step=step))
    size = cfg.image_sizes[-1]
    for rep in range(cfg.reps):
        refs = P.fresh_images(max(cfg.image_counts), size)
        sysm = P.nonce()
        for step, q in (("cold", "Describe these images."), ("new_question", "Which image is brightest?"),
                        ("identical", "Which image is brightest?")):
            rows.append(await _one(s, url, cfg, P.messages(sysm, q, refs), exp="hits_image",
                                   images=len(refs), size=list(size), rep=rep, step=step))
    return rows


async def exp_load(s, url, cfg, P) -> list[dict]:
    """Closed loop: N clients each send a fresh cold prompt as soon as their last one finishes."""
    rows = []
    for n in cfg.load_levels:
        reqs: list[dict] = []
        before = await _metrics(s, url)
        t_end = time.perf_counter() + cfg.load_seconds

        async def client(i: int):
            while time.perf_counter() < t_end:
                msgs = P.messages(P.nonce(), P.words(cfg.load_words))
                r = await _one(s, url, cfg, msgs, max_tokens=cfg.load_output_tokens)
                r["finished_at"] = time.perf_counter()
                reqs.append(r)

        t0 = time.perf_counter()
        await asyncio.gather(*(client(i) for i in range(n)))
        wall = time.perf_counter() - t0
        ok = [r for r in reqs if r["error"] is None and r["ttft_s"] is not None]
        ttft = [r["ttft_s"] for r in ok]
        itl = [r["itl_p50_s"] for r in ok if r["itl_p50_s"] is not None]
        rows.append({
            "exp": "load", "concurrency": n, "requests": len(reqs), "ok": len(ok), "wall_s": wall,
            "req_per_s": len(ok) / wall if wall else None,
            "output_tok_per_s": sum(r["output_chunks"] for r in ok) / wall if wall else None,
            "ttft_p50_s": float(np.percentile(ttft, 50)) if ttft else None,
            "ttft_p99_s": float(np.percentile(ttft, 99)) if ttft else None,
            "itl_p50_s": float(np.median(itl)) if itl else None,
            "itl_p99_s": float(np.percentile(itl, 99)) if itl else None,
            "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] or 0 for r in ok])) if ok else None,
            "metrics_before": {m: parse_metric(before, m) for m in METRICS_WANTED[:2]},
        })
    return rows


async def calibrate(url: str, cfg: CalibConfig = CalibConfig(), pool: ImagePool | None = None,
                    progress=print) -> dict:
    P = _Prompts(seed=2, pool=pool)
    out: dict = {"url": url, "config": asdict(cfg)}
    async with aiohttp.ClientSession() as s:
        progress("warm-up")
        for _ in range(2):
            await _one(s, url, cfg, P.messages(P.nonce(), P.words(64)), max_tokens=4)
        for name, fn in (("text_cold", exp_text_cold), ("image_cold", exp_image_cold),
                         ("hits", exp_hits), ("load", exp_load)):
            progress(f"experiment {name}")
            t0 = time.perf_counter()
            out[name] = await fn(s, url, cfg, P)
            progress(f"  done in {time.perf_counter() - t0:.0f}s, {len(out[name])} rows")
    return out


# ------------------------------------------------------------------------------- fit

def fit(raw: dict, block_size: int = 16) -> dict:
    """Least squares:  ttft = overhead + text_tokens / rate + image_tokens * m / rate.

    Returns a `cost_model` section loadable by kvsim.CostModel.from_calibration plus the
    measured curves that the step-based simulator (Phase 5) will need.
    """
    rows = [r for r in raw.get("text_cold", []) + raw.get("image_cold", [])
            if r.get("ttft_s") is not None and r.get("prompt_tokens")]
    X, y = [], []
    for r in rows:
        img = (r.get("mm_tokens") or {}).get("image", 0) or 0
        X.append([1.0, r["prompt_tokens"] - img, img])
        y.append(r["ttft_s"])
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    overhead, per_text, per_img = (float(c) for c in coef)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12

    hits = {}
    for exp in ("hits_text", "hits_image"):
        rs = [r for r in raw.get("hits", []) if r.get("exp") == exp and r.get("ttft_s") is not None]
        for step in ("cold", "new_question", "identical"):
            v = [r["ttft_s"] for r in rs if r["step"] == step]
            c = [r.get("cached_tokens") or 0 for r in rs if r["step"] == step]
            p = [r.get("prompt_tokens") or 0 for r in rs if r["step"] == step]
            if v:
                hits[f"{exp}.{step}"] = {"ttft_mean_s": float(np.mean(v)),
                                         "cached_frac": float(np.sum(c) / max(1, np.sum(p)))}

    img_tokens = {}
    for r in raw.get("image_cold", []):
        if r.get("mm_tokens") and r.get("images"):
            img_tokens.setdefault("x".join(map(str, r["size"])), set()).add(
                r["mm_tokens"].get("image", 0) // r["images"])

    rate = 1.0 / per_text if per_text > 0 else None
    return {
        "cost_model": {
            "block_size": block_size,
            "prefill_tokens_per_s": rate,
            "image_cost_multiplier": (per_img / per_text) if per_text > 0 else None,
            "fixed_overhead_s": max(0.0, overhead),
            "attn_s_per_token_pair": 0.0,
        },
        "fit": {"n": len(y), "r2": 1 - ss_res / ss_tot, "overhead_s": overhead,
                "s_per_text_token": per_text, "s_per_image_token": per_img},
        "image_tokens_per_image": {k: sorted(v) for k, v in img_tokens.items()},
        "hits": hits,
        "load": [{k: r[k] for k in ("concurrency", "req_per_s", "output_tok_per_s", "ttft_p50_s",
                                    "ttft_p99_s", "itl_p50_s", "itl_p99_s")}
                 for r in raw.get("load", [])],
    }


def save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def vllm_backend_lines(log_path: Path) -> list[str]:
    from deploy.workers import grep_log
    return grep_log(log_path, "attention backend", "using flashinfer", "using triton",
                    "using flex", "backend", "dtype", "compute capability", "error")


def plot(raw: dict, fitted: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tc = [r for r in raw.get("text_cold", []) if r.get("ttft_s")]
    ic = [r for r in raw.get("image_cold", []) if r.get("ttft_s")]
    ax[0].scatter([r["prompt_tokens"] for r in tc], [r["ttft_s"] for r in tc], s=14, label="text only")
    ax[0].scatter([r["prompt_tokens"] for r in ic], [r["ttft_s"] for r in ic], s=14, label="with images")
    f = fitted["fit"]
    xs = np.linspace(0, max([r["prompt_tokens"] for r in tc + ic] or [1]), 50)
    ax[0].plot(xs, f["overhead_s"] + xs * f["s_per_text_token"], "k--", lw=1,
               label=f"fit, text: {1 / f['s_per_text_token']:.0f} tok/s" if f["s_per_text_token"] > 0 else "fit")
    ax[0].set(xlabel="prompt tokens (cold)", ylabel="TTFT (s)", title="Prefill cost")
    ax[0].legend(frameon=False, fontsize=8)
    h = fitted["hits"]
    labels = [k for k in h]
    ax[1].bar(range(len(labels)), [h[k]["ttft_mean_s"] for k in labels])
    ax[1].set_xticks(range(len(labels)), [k.replace("hits_", "").replace(".", "\n") for k in labels], fontsize=7)
    ax[1].set(ylabel="TTFT (s)", title="Cold vs cache hit")
    ld = fitted["load"]
    if ld:
        c = [r["concurrency"] for r in ld]
        ax[2].plot(c, [r["ttft_p50_s"] for r in ld], "-o", label="TTFT p50")
        ax[2].plot(c, [r["ttft_p99_s"] for r in ld], "-o", label="TTFT p99")
        ax[2].plot(c, [r["itl_p50_s"] for r in ld], "-o", label="ITL p50")
        ax[2].set(xscale="log", yscale="log", xlabel="concurrent requests", ylabel="seconds",
                  title="Latency vs load (closed loop)")
        ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
