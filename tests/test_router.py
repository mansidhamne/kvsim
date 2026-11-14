"""Phase 3: router hashing, index, load signals, policies, proxy, and the experiment pipeline."""
import asyncio

import aiohttp

from bench.loadgen import run as replay
from bench.loadgen import summarize
from bench.mock_worker import MockCluster, MockConfig
from bench.workload import ImagePool, TraceConfig, build_messages, generate_trace
from kvsim.config import CostModel
from router.hashing import prompt_info
from router.index.shadow import ShadowIndex
from router.load import InFlight, WorkerLoad, parse_vllm_metrics
from router.policies import decide, make_policy
from router.server import RouterConfig, RouterRunner

SMALL = TraceConfig(seed=5, active_convs=6, duration_s=30, think_mean_s=3, pool_size=8,
                    images_per_conv=(1, 2), system_words=60, user_words=(5, 10), reply_words=(8, 16))
COST = CostModel(block_size=16, prefill_tokens_per_s=20_000, fixed_overhead_s=0.005,
                 image_cost_multiplier=2.0)


def test_prompt_info_prefix_and_image_weight():
    tr, pool = generate_trace(SMALL), ImagePool()
    conv = max(tr.conversations, key=lambda c: len(c.turns))
    a = prompt_info(build_messages(tr, conv, 0, pool), 16, 2.0)
    b = prompt_info(build_messages(tr, conv, 1, pool), 16, 2.0)
    assert b.blocks[:len(a.blocks) - 1] == a.blocks[:len(a.blocks) - 1]
    assert a.n_image_tokens == sum(256 if r.width == 448 else 384 for r in conv.images)
    # image tokens weigh double: total weight > token count / block size
    assert a.uncached_weight(0) > a.n_tokens / 16
    assert a.uncached_weight(len(a.blocks)) == a.tail_weight


def test_shadow_index_pending_then_resident():
    idx = ShadowIndex([10, 10])
    blocks = (1, 2, 3)
    idx.on_routed(0, blocks)
    assert idx.match(0, blocks) == 3 and idx.resident_match(0, blocks) == 0
    idx.on_prefilled(0, blocks)
    assert idx.resident_match(0, blocks) == 3 and not idx.pending[0]
    idx.on_routed(1, blocks)
    idx.on_dropped(1, blocks)
    assert idx.match(1, blocks) == 0
    idx.set_capacity(0, 2)
    assert len(idx.caches[0]) <= 2


def test_backlog_counts_only_unfinished_prefills():
    w = WorkerLoad("w0", "http://x")
    w.inflight["a"] = InFlight("a", (), 1.0, 2.0, sent_at=100.0)
    w.inflight["b"] = InFlight("b", (), 1.0, 2.0, sent_at=100.0, first_token_at=100.5)
    assert abs(w.backlog_s(now=100.25) - 0.75) < 1e-9
    assert abs(w.backlog_s(now=100.25, blind=True) - 1.75) < 1e-9
    assert w.backlog_s(now=200.0) == 0.0


def test_parse_vllm_metrics_reads_capacity_and_totals():
    text = ('vllm:cache_config_info{block_size="16",num_gpu_blocks="9000",engine="0"} 1.0\n'
            'vllm:num_requests_waiting{engine="0"} 3.0\n'
            'vllm:prefix_cache_hits_total{engine="0"} 1234.0\n# HELP x\n')
    m = parse_vllm_metrics(text)
    assert m == {"num_gpu_blocks": 9000, "block_size": 16, "waiting": 3.0, "prefix_hits": 1234.0}


def test_cache_aware_prefers_cached_worker_until_queue_is_too_long():
    loads = [WorkerLoad("w0", "a"), WorkerLoad("w1", "b")]
    idx = ShadowIndex([1000, 1000])
    msgs = [{"role": "user", "content": " ".join(["word"] * 3000)}]
    info = prompt_info(msgs, 16)
    idx.on_routed(0, info.blocks)
    idx.on_prefilled(0, info.blocks)                       # w0 holds the prefix
    pol = make_policy("cache_aware", 0)
    assert decide(pol, info, "c", loads, idx, COST, 0.0).wid == 0
    # give w0 a prefill backlog longer than recomputing on w1
    loads[0].inflight["x"] = InFlight("x", (), 5.0, 5.0, sent_at=0.0)
    d = decide(pol, info, "c", loads, idx, COST, 0.0)
    assert d.wid == 1 and d.predicted_hit_blocks == 0
    assert decide(make_policy("least_requests", 0), info, "c", loads, idx, COST, 0.0).wid == 1


def test_router_proxies_and_logs_actual_reuse():
    tr, pool = generate_trace(SMALL), ImagePool()

    async def go():
        async with MockCluster(2, MockConfig(time_scale=20, cache_blocks=2000)) as cl:
            async with RouterRunner(RouterConfig(cl.urls, policy="cache_aware", cost=COST)) as rr:
                await asyncio.sleep(0.6)
                recs = await replay(tr, pool, [rr.url], time_scale=20)
                async with aiohttp.ClientSession() as s:
                    async with s.get(rr.url + "/stats") as r:
                        stats = await r.json()
                return recs, list(rr.router.log_rows), stats

    recs, log, stats = asyncio.run(go())
    s = summarize(recs)
    assert s["errors"] == 0 and s["ok"] == tr.n_requests()
    assert set(s["per_worker"]) <= {"w0", "w1"}
    assert len(log) == tr.n_requests() and all("actual_cached_tokens" in r for r in log)
    # on the mock the router's tokenization IS the engine's, so predictions are near exact
    exact = sum(abs(r["predicted_resident_tokens"] - r["actual_cached_tokens"]) <= 16 for r in log)
    assert exact / len(log) > 0.9
    assert all(w["index_capacity"] == 2000 for w in stats["workers"])     # read from /metrics


def test_phase3_pipeline_on_mocks(tmp_path):
    from deploy.phase3 import Phase3Config, run_phase3
    from deploy.validate import validate_phase3
    from bench.workload import RequestOptions

    cfg = Phase3Config(policies=("round_robin", "cache_aware"), active_convs=(6,), duration_s=30,
                       warmup_s=5, time_scale=20, trace_overrides={"think_mean_s": 3.0, "pool_size": 8})

    async def go():
        async with MockCluster(2, MockConfig(time_scale=20, cache_blocks=3000)) as cl:
            return await run_phase3(cl.urls, cfg, tmp_path, COST, ImagePool(),
                                    RequestOptions(model="mock"), progress=lambda *_: None)

    cells = asyncio.run(go())
    rep = validate_phase3(cells)
    assert rep["ok"], [c for c in rep["checks"] if c["status"] == "FAIL"]
    by = {c["policy"]: c["summary"] for c in cells}
    assert by["cache_aware"]["hit_rate"] > by["round_robin"]["hit_rate"]
    assert (tmp_path / "phase3.png").exists()
    assert (tmp_path / "convs6" / "cache_aware" / "decisions.jsonl").exists()


def test_router_passthrough_models_and_completions():
    async def go():
        async with MockCluster(2, MockConfig(time_scale=20)) as cl:
            async with RouterRunner(RouterConfig(cl.urls, policy="cache_aware", cost=COST)) as rr:
                await asyncio.sleep(0.6)
                async with aiohttp.ClientSession() as s:
                    async with s.get(rr.url + "/v1/models") as r:
                        models = (r.status, await r.json())
                    prompt = "Once upon a time " + " ".join(["there was a long story"] * 60)
                    outs = []
                    for suffix in (" and then", " but later"):            # shared prefix
                        body = {"model": "mock", "prompt": prompt + suffix, "max_tokens": 4,
                                "stream": True, "stream_options": {"include_usage": True}}
                        async with s.post(rr.url + "/v1/completions", json=body) as r:
                            text, worker = (await r.text()), r.headers["x-kvroute-worker"]
                        outs.append((r.status, worker, text))
                    body = {"model": "mock", "prompt": prompt, "max_tokens": 3}
                    async with s.post(rr.url + "/v1/completions", json=body) as r:
                        plain = (r.status, await r.json())
                return models, outs, plain, list(rr.router.log_rows)

    models, outs, plain, log = asyncio.run(go())
    assert models[0] == 200 and models[1]["data"][0]["id"] == "mock"
    assert all(status == 200 and '"text"' in text and "[DONE]" in text for status, _, text in outs)
    assert outs[0][1] == outs[1][1]                     # follow-up routed to the cached worker
    assert log[1]["actual_cached_tokens"] > 0.8 * log[1]["prompt_tokens"]
    assert plain[0] == 200 and plain[1]["choices"][0]["text"]
    assert all("router_ttft_s" in r for r in log)


def test_image_weight_is_not_squared_by_attention():
    """Run-1 bug: the image cost multiplier was applied before the quadratic attention term,
    so image-heavy prompts were estimated ~9x too slow. Attention must use raw tokens."""
    cost = CostModel(block_size=16, prefill_tokens_per_s=11_000, image_cost_multiplier=6.0,
                     fixed_overhead_s=0.04, attn_s_per_token_pair=3.2e-7)
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "uuid": f"img{i}", "image_url": {"url": ImagePool().data_url(
            __import__("bench.workload", fromlist=["ImageRef"]).ImageRef(i, 448, 448))}}
        for i in range(8)] + [{"type": "text", "text": "describe " * 200}]}]
    info = prompt_info(msgs, 16, cost.image_cost_multiplier)
    from router.policies import _Req
    est = cost.request_prefill_time(_Req(info, "c"), 0)
    raw, img = info.n_tokens, info.n_image_tokens
    expected = 0.04 + (raw + img * 5.0) / 11_000 + 3.2e-7 * raw * raw / 2
    assert abs(est - expected) < 1e-6
    assert est < 3.0            # the buggy formula gave ~20 s for this 2.3k-token prompt


def test_validation_fails_on_unrealistic_estimates():
    from deploy.validate import validate_phase3
    base = {"errors": 0, "requests": 100, "hit_rate": 0.8, "late_frac": 0.0, "sample_outputs": [],
            "estimate_ratio_n": 50}
    cells = [{"active_convs": 12, "policy": p, "trace_fingerprint": "x", "reset_ok": [True],
              "server": {"prefix_hits": 10.0}, "client_all": {"cached_tokens": 10, "requests": 100},
              "summary": {**base, "estimate_ratio_median": r}}
             for p, r in (("round_robin", 9.0), ("cache_aware", 9.0), ("least_loaded", 1.1))]
    rep = validate_phase3(cells)
    st = {(c["name"].split(":")[0]): c["status"] for c in rep["checks"] if "estimates" in c["name"]}
    assert st["12 convs / cache_aware"] == "FAIL"
    assert st["12 convs / round_robin"] == "WARN"       # doesn't act on estimates
    assert st["12 convs / least_loaded"] == "PASS"
    assert not rep["ok"]
