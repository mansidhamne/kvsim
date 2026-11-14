"""Phase 3: the router. An OpenAI-compatible proxy that picks a worker per request.

    python -m router.server --workers http://127.0.0.1:8001,http://127.0.0.1:8002 \
        --policy cache_aware --cost calibration/results/phase2/calibration_vllm_final.json

Endpoints
  POST /v1/chat/completions   route, forward, stream the worker's response back unchanged
  POST /v1/completions        same, for plain-text prompts (the prompt is hashed as one message)
  GET  /v1/models             forwarded to the first healthy worker (all serve the same model)
  GET  /health                ok
  GET  /metrics               router counters (Prometheus text)
  GET  /stats                 JSON: per-worker load, index size, decisions so far
  POST /reset                 clear router state (index, session map) between experiments

Per request the router:
  1. hashes the prompt into blocks (router.hashing) and estimates its prefill cost,
  2. asks the policy for a worker (router.policies, same code as the simulator),
  3. forwards the request; when the first token comes back it marks the prefill done and
     inserts the blocks into its shadow index,
  4. reads the final usage chunk (cached_tokens) and logs predicted vs actual reuse.

The response carries `x-kvroute-worker` (chosen worker) and `x-kvroute-decision-ms`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from aiohttp import web

from kvsim.config import CostModel

from .hashing import prompt_info
from .index.shadow import ShadowIndex
from .load import InFlight, WorkerLoad, parse_vllm_metrics
from .policies import decide, make_policy

DEFAULT_CAPACITY_BLOCKS = 4096


@dataclass
class RouterConfig:
    workers: list[str]
    policy: str = "cache_aware"
    alpha: float = 1.0
    pending_aware: bool = True
    cost: CostModel = field(default_factory=lambda: CostModel(block_size=16))
    capacity_blocks: int | None = None      # per worker; None = read from worker /metrics
    scrape_interval_s: float = 0.5
    decision_log: str | None = None         # JSONL path
    seed: int = 0
    request_timeout_s: float = 900.0


class Router:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.cost = cfg.cost
        n = len(cfg.workers)
        self.loads = [WorkerLoad(f"w{i}", u.rstrip("/")) for i, u in enumerate(cfg.workers)]
        self.index = ShadowIndex([cfg.capacity_blocks or DEFAULT_CAPACITY_BLOCKS] * n)
        self._capacity_known = [cfg.capacity_blocks is not None] * n
        kw = {"alpha": cfg.alpha, "pending_aware": cfg.pending_aware} if cfg.policy == "cache_aware" else {}
        self.policy = make_policy(cfg.policy, cfg.seed, **kw)
        self._img_cache: dict = {}
        self.log_rows: list[dict] = []
        self.counters = {"requests": 0, "errors": 0, "decision_s_sum": 0.0}

    def reset(self) -> None:
        self.index.reset()
        kw = {"alpha": self.cfg.alpha, "pending_aware": self.cfg.pending_aware} \
            if self.cfg.policy == "cache_aware" else {}
        self.policy = make_policy(self.cfg.policy, self.cfg.seed, **kw)
        for l in self.loads:
            l.inflight.clear()
            l.routed = 0
        self.log_rows.clear()
        self.counters = {"requests": 0, "errors": 0, "decision_s_sum": 0.0}

    # ------------------------------------------------------------------ worker state
    async def scrape_loop(self, session: aiohttp.ClientSession) -> None:
        while True:
            for i, l in enumerate(self.loads):
                try:
                    async with session.get(l.url + "/metrics",
                                           timeout=aiohttp.ClientTimeout(total=2)) as r:
                        m = parse_vllm_metrics(await r.text())
                    l.scraped, l.scraped_at = m, time.perf_counter()
                    if not self._capacity_known[i] and m.get("num_gpu_blocks"):
                        bs = m.get("block_size", self.cost.block_size)
                        cap = m["num_gpu_blocks"] * bs // self.cost.block_size
                        self.index.set_capacity(i, cap)
                        self._capacity_known[i] = True
                except Exception:  # noqa: BLE001 - a missed scrape is not fatal
                    pass
            await asyncio.sleep(self.cfg.scrape_interval_s)

    # ------------------------------------------------------------------ request path
    async def handle(self, request: web.Request) -> web.StreamResponse:
        session: aiohttp.ClientSession = request.app[SESSION_KEY]
        body = await request.json()
        conv = request.headers.get("x-kvroute-conv") or str(uuid.uuid4())
        rid = f"r{self.counters['requests']}-{uuid.uuid4().hex[:6]}"
        self.counters["requests"] += 1

        t0 = time.perf_counter()
        info = prompt_info(messages_of(body), self.cost.block_size,
                           self.cost.image_cost_multiplier, self._img_cache)
        d = decide(self.policy, info, conv, self.loads, self.index, self.cost, t0)
        decision_s = time.perf_counter() - t0
        self.counters["decision_s_sum"] += decision_s

        load = self.loads[d.wid]
        load.routed += 1
        fl = InFlight(rid, info.blocks, d.est_prefill_s, d.full_prefill_s, time.perf_counter())
        load.inflight[rid] = fl
        self.index.on_routed(d.wid, info.blocks)

        row = {"rid": rid, "conv": conv, "turn": request.headers.get("x-kvroute-turn"),
               "policy": self.cfg.policy, "worker": load.name, "t_routed": t0,
               "decision_ms": round(decision_s * 1e3, 3), "prompt_tokens_est": info.n_tokens,
               "blocks": len(info.blocks), "predicted_hit_tokens": d.predicted_hit_blocks * self.cost.block_size,
               "predicted_resident_tokens": d.predicted_resident_blocks * self.cost.block_size,
               "est_prefill_s": round(d.est_prefill_s, 4), "scores": d.scores}
        headers = {"x-kvroute-worker": load.name, "x-kvroute-decision-ms": f"{decision_s * 1e3:.3f}"}
        resp: web.StreamResponse | None = None
        prefilled = False

        def mark_prefilled():
            nonlocal prefilled
            if not prefilled:
                prefilled = True
                fl.first_token_at = time.perf_counter()
                row["router_ttft_s"] = round(fl.first_token_at - t0, 4)
                self.index.on_prefilled(d.wid, info.blocks)

        try:
            async with session.post(load.url + request.path, json=body,
                                    timeout=aiohttp.ClientTimeout(total=self.cfg.request_timeout_s)) as up:
                row["status"] = up.status
                if up.status != 200 or not body.get("stream"):
                    data = await up.read()
                    if up.status == 200:
                        mark_prefilled()
                        try:
                            self._record_usage(row, json.loads(data).get("usage"))
                        except ValueError:
                            pass
                    return web.Response(body=data, status=up.status, headers={
                        **headers, "Content-Type": up.headers.get("Content-Type", "application/json")})
                resp = web.StreamResponse(status=200, headers={
                    **headers, "Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
                await resp.prepare(request)
                buf = b""
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._inspect_line(line.strip(), row, mark_prefilled)
                if buf.strip():
                    self._inspect_line(buf.strip(), row, mark_prefilled)
                await resp.write_eof()
                return resp
        except Exception as e:  # noqa: BLE001
            self.counters["errors"] += 1
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            if resp is None:
                return web.json_response({"error": row["error"]}, status=502, headers=headers)
            return resp
        finally:
            if not prefilled:
                self.index.on_dropped(d.wid, info.blocks)
            load.inflight.pop(rid, None)
            row["router_e2e_s"] = round(time.perf_counter() - t0, 4)
            self.log_rows.append(row)

    def _inspect_line(self, line: bytes, row: dict, mark_prefilled) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        try:
            obj = json.loads(data)
        except ValueError:
            return
        for ch in obj.get("choices") or []:
            # chat: delta.content; completions: text
            if (ch.get("delta") or {}).get("content") or ch.get("text") or ch.get("finish_reason"):
                mark_prefilled()
        if obj.get("usage"):
            self._record_usage(row, obj["usage"])

    @staticmethod
    def _record_usage(row: dict, usage: dict | None) -> None:
        if not usage:
            return
        det = usage.get("prompt_tokens_details") or {}
        row["prompt_tokens"] = usage.get("prompt_tokens")
        row["actual_cached_tokens"] = det.get("cached_tokens")
        row["completion_tokens"] = usage.get("completion_tokens")

    # ------------------------------------------------------------------ observability
    def stats(self) -> dict:
        return {
            "policy": self.cfg.policy, "counters": self.counters,
            "workers": [{"name": l.name, "url": l.url, "inflight": l.n_inflight(), "routed": l.routed,
                         "backlog_s": round(l.backlog_s(), 4), "scraped": l.scraped,
                         "index_blocks": len(self.index.caches[i]),
                         "index_capacity": self.index.caches[i].capacity}
                        for i, l in enumerate(self.loads)],
        }

    def metrics_text(self) -> str:
        c = self.counters
        lines = [f"kvroute:requests_total {c['requests']}", f"kvroute:errors_total {c['errors']}",
                 f"kvroute:decision_seconds_sum {c['decision_s_sum']:.6f}"]
        for l in self.loads:
            lines.append(f'kvroute:inflight{{worker="{l.name}"}} {l.n_inflight()}')
            lines.append(f'kvroute:routed_total{{worker="{l.name}"}} {l.routed}')
        return "\n".join(lines) + "\n"

    def write_log(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for r in self.log_rows:
                f.write(json.dumps(r) + "\n")


def messages_of(body: dict) -> list[dict]:
    """The prompt as chat messages, for hashing. /v1/completions sends `prompt` (a string, or a
    list of strings / token ids); it is hashed as one message so identical prompt prefixes map
    to identical blocks."""
    if "messages" in body:
        return body.get("messages") or []
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(x) for x in prompt)
    return [{"role": "prompt", "content": str(prompt)}]


ROUTER_KEY = web.AppKey("router", Router)
SESSION_KEY = web.AppKey("session", aiohttp.ClientSession)
SCRAPE_KEY = web.AppKey("scrape", asyncio.Task)


def make_app(cfg: RouterConfig) -> web.Application:
    router = Router(cfg)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app[ROUTER_KEY] = router

    async def on_startup(a):
        a[SESSION_KEY] = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024))
        a[SCRAPE_KEY] = asyncio.create_task(router.scrape_loop(a[SESSION_KEY]))

    async def on_cleanup(a):
        a[SCRAPE_KEY].cancel()
        try:
            await a[SCRAPE_KEY]
        except asyncio.CancelledError:
            pass
        if cfg.decision_log:
            router.write_log(cfg.decision_log)
        await a[SESSION_KEY].close()

    async def health(_):
        return web.Response(text="ok")

    async def metrics(_):
        return web.Response(text=router.metrics_text(), content_type="text/plain")

    async def stats(_):
        return web.json_response(router.stats())

    async def reset(_):
        router.reset()
        return web.json_response({"success": True})

    async def models(_):
        session = app[SESSION_KEY]
        last = None
        for l in router.loads:
            try:
                async with session.get(l.url + "/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    data = await r.read()
                    if r.status == 200:
                        return web.Response(body=data, status=200,
                                            content_type=r.headers.get("Content-Type", "application/json").split(";")[0])
                    last = f"{l.name}: HTTP {r.status}"
            except Exception as e:  # noqa: BLE001
                last = f"{l.name}: {type(e).__name__}"
        return web.json_response({"error": f"no worker answered /v1/models ({last})"}, status=502)

    app.router.add_post("/v1/chat/completions", router.handle)
    app.router.add_post("/v1/completions", router.handle)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/stats", stats)
    app.router.add_post("/reset", reset)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


class RouterRunner:
    """Run the router in this process (used by experiments and tests).

        async with RouterRunner(cfg) as rr:
            rr.url, rr.router
    """

    def __init__(self, cfg: RouterConfig, host: str = "127.0.0.1", port: int = 0):
        self.cfg, self.host, self.port = cfg, host, port
        self.runner: web.AppRunner | None = None
        self.url = ""

    @property
    def router(self) -> Router:
        return self.runner.app[ROUTER_KEY]  # type: ignore[union-attr]

    async def __aenter__(self) -> "RouterRunner":
        self.runner = web.AppRunner(make_app(self.cfg), access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://{self.host}:{port}"
        return self

    async def __aexit__(self, *exc) -> None:
        if self.runner:
            await self.runner.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser(description="KV-cache-aware router (OpenAI-compatible proxy).")
    ap.add_argument("--workers", required=True, help="comma-separated worker base URLs")
    ap.add_argument("--policy", default="cache_aware")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cost", help="calibration JSON with a cost_model section")
    ap.add_argument("--capacity-blocks", type=int, help="per-worker KV blocks; default: read from /metrics")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--decision-log", default="router_decisions.jsonl")
    a = ap.parse_args()
    cost = CostModel.from_calibration(a.cost) if a.cost else CostModel(block_size=16)
    cfg = RouterConfig(workers=a.workers.split(","), policy=a.policy, alpha=a.alpha, cost=cost,
                       capacity_blocks=a.capacity_blocks, decision_log=a.decision_log)
    web.run_app(make_app(cfg), host="127.0.0.1", port=a.port, access_log=None)


if __name__ == "__main__":
    main()
