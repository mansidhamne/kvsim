"""Phase 1: a fake vLLM server for laptop development.

Speaks enough of vLLM's OpenAI-compatible API for the router and load generator:
  POST /v1/chat/completions   (streaming SSE and non-streaming)
  GET  /metrics               (Prometheus text, vLLM metric names; see docs/vllm_api.md)
  GET  /health

Model of the engine (deliberately simple, NOT a performance model of vLLM):
* Tokenization: 1 word = 1 token; a role marker per message; an image = (w/28)*(h/28) tokens,
  like Qwen2-VL. Image tokens are keyed by the image's uuid (or a hash of its bytes).
* Prefix cache: chained block hashes over full blocks (like vLLM) in an LRU block cache with a
  fixed capacity. A request's hit = longest cached prefix, decided when its prefill STARTS.
* Prefill: one at a time, FIFO, time = overhead + uncached tokens / prefill rate.
  (Real engines batch; Phase 5 replaces this with a step-based model.)
* Decode: each sequence streams tokens at a fixed rate, independently of the others.
* `time_scale` > 1 makes everything faster (for tests and quick demos).

Run 3 workers:  python -m bench.mock_worker --n 3 --base-port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field

from aiohttp import web

from kvsim.cache import LRUBlockCache

ENGINE_KEY: "web.AppKey[MockEngine]"


@dataclass(frozen=True)
class MockConfig:
    worker_id: str = "mock-0"
    block_size: int = 16
    cache_blocks: int = 4096              # KV budget in blocks (16 tokens each)
    prefill_tps: float = 4000.0           # uncached prompt tokens per second
    prefill_overhead_s: float = 0.02
    decode_tps: float = 40.0              # output tokens per second per sequence
    image_patch: int = 28                 # Qwen2-VL: one token per 28x28 pixels
    time_scale: float = 1.0
    tokens_per_chunk: int = 1             # >1 emulates vLLM putting several tokens in one chunk
    empty_content: bool = False           # stream "" deltas (what vLLM sends when a token
                                          # decodes to nothing); for loadgen tests


# ---------------------------------------------------------------------------------- tokens


# Tokenization and block hashing are shared with the router (router/hashing.py), so on the
# mock the router's shadow index sees exactly what the "engine" caches.
from router.hashing import block_hashes, blocks_with_reply, tokenize  # noqa: E402,F401


def messages_of(body: dict) -> list[dict]:
    """Chat messages, or a /v1/completions prompt as one message (same rule as the router)."""
    if "messages" in body:
        return body.get("messages") or []
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(x) for x in prompt)
    return [{"role": "prompt", "content": str(prompt)}]


# ---------------------------------------------------------------------------------- engine


@dataclass
class _Job:
    n_tokens: int
    blocks: list[int]
    image_tokens: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)
    hit_tokens: int = 0
    queued_at: float = 0.0
    prefill_s: float = 0.0


class MockEngine:
    def __init__(self, cfg: MockConfig):
        self.cfg = cfg
        self.cache = LRUBlockCache(cfg.cache_blocks)
        self.queue: asyncio.Queue[_Job] = asyncio.Queue()
        self.prefilling = 0
        self.decoding = 0
        self.prefix_queries = 0
        self.prefix_hits = 0
        self.requests_done = 0
        self.other_resets = 0
        self.ttft_sum = 0.0          # sums only (no buckets): enough for mean values
        self.itl_sum = 0.0
        self.itl_count = 0
        self._img_tokens: dict = {}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._prefill_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _prefill_loop(self) -> None:
        c = self.cfg
        while True:
            job = await self.queue.get()
            self.prefilling = 1
            hit = self.cache.match_len(job.blocks)              # real hit, decided now
            self.cache.touch(job.blocks[:hit])
            job.hit_tokens = hit * c.block_size
            new = job.n_tokens - job.hit_tokens
            job.prefill_s = c.prefill_overhead_s + new / c.prefill_tps
            await asyncio.sleep(job.prefill_s / c.time_scale)
            self.cache.insert(job.blocks)
            self.prefix_queries += job.n_tokens
            self.prefix_hits += job.hit_tokens
            self.prefilling = 0
            job.done.set()

    async def prefill(self, messages: list[dict]) -> _Job:
        toks = tokenize(messages, self.cfg, self._img_tokens)
        job = _Job(len(toks), block_hashes(toks, self.cfg.block_size),
                   image_tokens=sum(1 for t in toks if t[0] == "img"), queued_at=time.perf_counter())
        await self.queue.put(job)
        await job.done.wait()
        self.ttft_sum += time.perf_counter() - job.queued_at
        return job

    def metrics_text(self) -> str:
        c, wid = self.cfg, self.cfg.worker_id
        lab = f'{{model_name="mock",worker="{wid}"}}'
        lines = [
            "# TYPE vllm:prefix_cache_queries counter", f"vllm:prefix_cache_queries{lab} {self.prefix_queries}",
            "# TYPE vllm:prefix_cache_hits counter", f"vllm:prefix_cache_hits{lab} {self.prefix_hits}",
            "# TYPE vllm:num_requests_running gauge", f"vllm:num_requests_running{lab} {self.prefilling + self.decoding}",
            "# TYPE vllm:num_requests_waiting gauge", f"vllm:num_requests_waiting{lab} {self.queue.qsize()}",
            "# TYPE vllm:kv_cache_usage_perc gauge", f"vllm:kv_cache_usage_perc{lab} {len(self.cache) / c.cache_blocks:.4f}",
            "# TYPE vllm:time_to_first_token_seconds histogram",
            f"vllm:time_to_first_token_seconds_sum{lab} {self.ttft_sum:.6f}",
            f"vllm:time_to_first_token_seconds_count{lab} {self.requests_done}",
            "# TYPE vllm:inter_token_latency_seconds histogram",
            f"vllm:inter_token_latency_seconds_sum{lab} {self.itl_sum:.6f}",
            f"vllm:inter_token_latency_seconds_count{lab} {self.itl_count}",
            "# TYPE vllm:cache_config_info gauge",
            f'vllm:cache_config_info{{block_size="{c.block_size}",num_gpu_blocks="{c.cache_blocks}",worker="{wid}"}} 1.0',
            "# TYPE vllm:num_preemptions counter", f"vllm:num_preemptions_total{lab} 0",
            "# TYPE kvroute:mock_requests_done counter", f"kvroute:mock_requests_done{lab} {self.requests_done}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------- http

ENGINE_KEY = web.AppKey("engine", MockEngine)


_REPLY_WORDS = ("the image shows a small wooden table near a bright window with green plants "
                "on the left and a red chair in the corner while soft light falls across").split()


def _reply_tokens(messages: list[dict], n: int) -> list[str]:
    """Readable fake output (so output checks can run on the mock): a fixed answer to the
    sanity question, otherwise varied words."""
    last = messages[-1].get("content", "") if messages else ""
    if isinstance(last, list):
        last = " ".join(p.get("text", "") for p in last if p.get("type") == "text")
    if "capital of france" in str(last).lower():
        words = ["Paris."]
    else:
        words = _REPLY_WORDS
    return [(words[i % len(words)] + " ") for i in range(n)]


def _chunk(rid: str, delta: dict, finish: str | None = None, chat: bool = True) -> bytes:
    if chat:
        obj = {"id": rid, "object": "chat.completion.chunk", "model": "mock",
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    else:   # /v1/completions streams `text` instead of `delta`
        obj = {"id": rid, "object": "text_completion", "model": "mock",
               "choices": [{"index": 0, "text": delta.get("content", ""), "finish_reason": finish}]}
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def make_app(cfg: MockConfig) -> web.Application:
    engine = MockEngine(cfg)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app[ENGINE_KEY] = engine
    counter = {"n": 0}

    async def on_startup(_app):
        engine.start()

    async def on_cleanup(_app):
        await engine.stop()

    async def health(_req):
        return web.Response(text="ok")

    async def metrics(_req):
        return web.Response(text=engine.metrics_text(), content_type="text/plain")

    async def chat(req: web.Request):
        body = await req.json()
        is_chat = req.path.endswith("/chat/completions")
        messages = messages_of(body)
        counter["n"] += 1
        rid = f"{'chatcmpl' if is_chat else 'cmpl'}-{cfg.worker_id}-{counter['n']}"
        n_out = int(body.get("max_tokens") or 16)
        reply = _reply_tokens(messages, n_out)
        job = await engine.prefill(messages)
        usage = {"prompt_tokens": job.n_tokens, "completion_tokens": n_out,
                 "total_tokens": job.n_tokens + n_out,
                 "prompt_tokens_details": {"cached_tokens": job.hit_tokens,
                                           "multimodal_tokens": {"image": job.image_tokens} if job.image_tokens else None}}
        headers = {"x-kvroute-worker": cfg.worker_id}
        gap = 1.0 / cfg.decode_tps / cfg.time_scale
        engine.itl_sum += gap * max(0, n_out - 1)
        engine.itl_count += max(0, n_out - 1)
        engine.decoding += 1
        try:
            if not body.get("stream"):
                await asyncio.sleep(gap * n_out)
                choice = ({"index": 0, "finish_reason": "length",
                           "message": {"role": "assistant", "content": "".join(reply)}} if is_chat else
                          {"index": 0, "finish_reason": "length", "text": "".join(reply)})
                return web.json_response({
                    "id": rid, "object": "chat.completion" if is_chat else "text_completion",
                    "model": "mock", "choices": [choice], "usage": usage}, headers=headers)
            resp = web.StreamResponse(headers={**headers, "Content-Type": "text/event-stream",
                                               "Cache-Control": "no-cache"})
            await resp.prepare(req)
            so = body.get("stream_options") or {}
            continuous = bool(so.get("include_usage") and so.get("continuous_usage_stats"))

            def with_usage(chunk: bytes, done_tokens: int) -> bytes:
                if not continuous:
                    return chunk
                obj = json.loads(chunk[len(b"data: "):])
                obj["usage"] = {"prompt_tokens": job.n_tokens, "completion_tokens": done_tokens,
                                "total_tokens": job.n_tokens + done_tokens}
                return b"data: " + json.dumps(obj).encode() + b"\n\n"

            if is_chat:
                await resp.write(with_usage(_chunk(rid, {"role": "assistant", "content": ""}), 0))
            k = max(1, cfg.tokens_per_chunk)
            done = 0
            while done < n_out:
                step = min(k, n_out - done)
                if done:
                    await asyncio.sleep(gap * step)   # several tokens arrive together after k steps
                text = "" if cfg.empty_content else "".join(reply[done:done + step])
                done += step
                await resp.write(with_usage(_chunk(rid, {"content": text}, chat=is_chat), done))
            await resp.write(with_usage(_chunk(rid, {}, "length", chat=is_chat), n_out))
            if (body.get("stream_options") or {}).get("include_usage"):
                u = {"id": rid, "object": "chat.completion.chunk", "model": "mock",
                     "choices": [], "usage": usage}
                await resp.write(b"data: " + json.dumps(u).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            engine.decoding -= 1
            engine.requests_done += 1
            # like vLLM: generated tokens fill KV blocks that are cached too, so a follow-up that
            # sends this reply back in its history can reuse it
            engine.cache.insert(blocks_with_reply(messages, "".join(reply), cfg.block_size,
                                                  engine._img_tokens, cfg.image_patch))

    async def reset_prefix_cache(_req):
        # like vLLM's dev endpoint: refuses while requests are running
        if engine.prefilling or engine.decoding or engine.queue.qsize():
            return web.json_response({"success": False})
        engine.cache = LRUBlockCache(cfg.cache_blocks)
        return web.json_response({"success": True})

    async def reset_other_cache(_req):
        # vLLM's /reset_mm_cache and /reset_encoder_cache: 200 with no body. The mock has no
        # multimodal or encoder cache, so these only count calls (tests check they were made).
        engine.other_resets += 1
        return web.Response(status=200)

    app.router.add_post("/reset_prefix_cache", reset_prefix_cache)
    app.router.add_post("/reset_mm_cache", reset_other_cache)
    app.router.add_post("/reset_encoder_cache", reset_other_cache)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    async def models(_req):
        return web.json_response({"object": "list", "data": [
            {"id": "mock", "object": "model", "owned_by": "kvroute-mock"}]})

    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/completions", chat)
    app.router.add_get("/v1/models", models)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


class MockCluster:
    """N mock workers in this process, on consecutive ports (port 0 = pick free ports).

        async with MockCluster(3, MockConfig(time_scale=20)) as cl:
            urls = cl.urls
    """

    def __init__(self, n: int, cfg: MockConfig = MockConfig(), base_port: int = 0,
                 host: str = "127.0.0.1"):
        self.n, self.cfg, self.base_port, self.host = n, cfg, base_port, host
        self.runners: list[web.AppRunner] = []
        self.urls: list[str] = []
        self.apps: list[web.Application] = []

    async def __aenter__(self) -> "MockCluster":
        from dataclasses import replace
        for i in range(self.n):
            app = make_app(replace(self.cfg, worker_id=f"mock-{i}"))
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.base_port + i if self.base_port else 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
            self.runners.append(runner)
            self.apps.append(app)
            self.urls.append(f"http://{self.host}:{port}")
        return self

    async def __aexit__(self, *exc) -> None:
        for r in self.runners:
            await r.cleanup()

    def engines(self) -> list[MockEngine]:
        return [a[ENGINE_KEY] for a in self.apps]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run N mock vLLM workers.")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=8001)
    ap.add_argument("--cache-blocks", type=int, default=MockConfig.cache_blocks)
    ap.add_argument("--prefill-tps", type=float, default=MockConfig.prefill_tps)
    ap.add_argument("--decode-tps", type=float, default=MockConfig.decode_tps)
    ap.add_argument("--time-scale", type=float, default=1.0)
    a = ap.parse_args()
    cfg = MockConfig(cache_blocks=a.cache_blocks, prefill_tps=a.prefill_tps,
                     decode_tps=a.decode_tps, time_scale=a.time_scale)

    async def run():
        async with MockCluster(a.n, cfg, a.base_port) as cl:
            print("mock workers:", " ".join(cl.urls), flush=True)
            await asyncio.Event().wait()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
