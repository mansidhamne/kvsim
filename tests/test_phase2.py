"""Phase 2: vLLM launch flags, cost-model fit, smoke + calibration against a mock worker."""
import asyncio
import json

from bench.mock_worker import MockCluster, MockConfig
from deploy.calibrate import CalibConfig, calibrate, fit, parse_metric, smoke
from deploy.workers import ServeConfig, WorkerSpec, serve_command
from kvsim import CostModel


def test_serve_command_has_hash_and_usage_flags():
    cmd = serve_command(WorkerSpec(port=8001, gpu=0, events_port=5557), ServeConfig(kv_events=True))
    s = " ".join(cmd)
    for flag in ("--dtype float16", "--block-size 16", "--prefix-caching-hash-algo sha256_cbor",
                 "--enable-prompt-tokens-details", "--served-model-name model", "--port 8001"):
        assert flag in s
    ev = json.loads(cmd[cmd.index("--kv-events-config") + 1])
    assert ev["enable_kv_cache_events"] and ev["endpoint"] == "tcp://*:5557"
    assert "--kv-events-config" not in serve_command(WorkerSpec(8001, 0), ServeConfig())


def test_fit_recovers_known_cost_model():
    # ttft = 0.03 + text/2000 + 1.5 * image/2000
    rows = []
    for text in (100, 500, 1000, 3000):
        rows.append({"prompt_tokens": text, "mm_tokens": None, "ttft_s": 0.03 + text / 2000})
    for img in (256, 512, 1024):
        t = 50
        rows.append({"prompt_tokens": t + img, "mm_tokens": {"image": img}, "images": img // 256,
                     "size": [448, 448], "ttft_s": 0.03 + t / 2000 + 1.5 * img / 2000})
    out = fit({"text_cold": rows[:4], "image_cold": rows[4:], "hits": [], "load": []})
    cm = out["cost_model"]
    assert abs(cm["prefill_tokens_per_s"] - 2000) < 1
    assert abs(cm["image_cost_multiplier"] - 1.5) < 1e-6
    assert abs(cm["fixed_overhead_s"] - 0.03) < 1e-6
    assert out["image_tokens_per_image"] == {"448x448": [256]}
    CostModel(**cm)                       # loadable by the simulator


def test_parse_metric_sums_label_sets():
    text = 'x{a="1"} 2\nx{a="2"} 3.5\nxy 9\n# TYPE x counter\n'
    assert parse_metric(text, "x") == 5.5
    assert parse_metric(text, "missing") is None


def test_smoke_and_quick_calibration_on_mock():
    async def go():
        async with MockCluster(1, MockConfig(time_scale=20)) as cl:
            sm = await smoke(cl.urls[0], CalibConfig.quick())
            raw = await calibrate(cl.urls[0], CalibConfig.quick(), progress=lambda *_: None)
        return sm, raw

    sm, raw = asyncio.run(go())
    assert sm["ok"] and all(sm["checks"].values())
    assert sm["images"]["448x448"]["image_tokens"] == 256
    out = fit(raw)
    cm = out["cost_model"]
    # mock: 4000 tok/s x time_scale 20 = 80k tok/s; allow wide tolerance for timer noise
    assert 15_000 < cm["prefill_tokens_per_s"] < 500_000   # timer noise on a busy laptop
    assert out["hits"]["hits_text.new_question"]["cached_frac"] > 0.8
    assert out["hits"]["hits_text.cold"]["cached_frac"] == 0
    assert [r["concurrency"] for r in out["load"]] == [1, 4]


def test_fit_recovers_attention_term():
    # ttft = 0.04 + n/10000 + 3e-7 * n * (cached + n/2), cold and cached rows
    def t(n, c):
        return 0.04 + n / 10000 + 3e-7 * n * (c + n / 2)
    cold = [{"prompt_tokens": n, "cached_tokens": 0, "ttft_s": t(n, 0)} for n in (100, 1000, 2000, 4000)]
    hits = [{"prompt_tokens": n + c, "cached_tokens": c, "ttft_s": t(n, c), "exp": "hits_text",
             "step": "new_question"} for n, c in ((20, 1000), (20, 4000))]
    out = fit({"text_cold": cold, "image_cold": [], "hits": hits, "load": []})
    cm = out["cost_model"]
    assert abs(cm["prefill_tokens_per_s"] - 10000) < 1
    assert abs(cm["attn_s_per_token_pair"] - 3e-7) < 1e-12
    assert cm["image_cost_multiplier"] == 1.0          # no image rows -> no image term
    assert out["fit"]["r2"] > out["fit"]["linear_only"]["r2"]
    model = CostModel(**{**cm, "block_size": 1})
    assert abs(model.prefill_time(20, cached_blocks=4000) - t(20, 4000)) < 1e-9


def test_ttft_recorded_when_tokens_decode_to_empty():
    from deploy.calibrate import _one, _Prompts

    async def go():
        async with MockCluster(1, MockConfig(time_scale=20, empty_content=True)) as cl:
            async with aiohttp.ClientSession() as s:
                P = _Prompts()
                return await _one(s, cl.urls[0], CalibConfig(), P.messages("x", "hello"), max_tokens=1)

    import aiohttp
    r = asyncio.run(go())
    assert r["status"] == 200 and r["ttft_s"] is not None and r["output_chunks"] == 0


# ------------------------------------------------------------------------------- validation

def test_looks_like_text_catches_degenerate_output():
    from deploy.validate import looks_like_text
    assert looks_like_text("The capital of France is Paris.")[0]
    assert not looks_like_text("!!!!!!!!!!!!!!!!")[0]            # typical fp16 overflow output
    assert not looks_like_text("the the the the the the the the the the")[0]
    assert not looks_like_text("")[0]
    assert not looks_like_text("abc��")[0]


def test_full_mock_run_validates():
    # time_scale 4 (not 20): at millisecond timings, laptop scheduling noise breaks the R2 check
    from deploy.validate import validate_run

    async def go():
        async with MockCluster(1, MockConfig(time_scale=4)) as cl:
            sm = await smoke(cl.urls[0], CalibConfig.quick())
            raw = await calibrate(cl.urls[0], CalibConfig.quick(), progress=lambda *_: None)
        return sm, raw

    sm, raw = asyncio.run(go())
    rep = validate_run(sm, raw, fit(raw))
    assert rep["ok"], [c for c in rep["checks"] if c["status"] == "FAIL"]
    assert "Paris" in sm["sanity_reply"]


def test_validation_fails_on_bad_data():
    from deploy.validate import validate_run
    sm = {"checks": {"text_request_ok": True}, "blocking": ["text_request_ok"],
          "sanity_reply": "!!!!!!!!", "images": {}}
    raw = {"text_cold": [
        {"words": 64, "rep": 0, "prompt_tokens": 100, "cached_tokens": 0, "ttft_s": 0.05,
         "completion_tokens": 1, "max_tokens": 1, "status": 200, "error": None},
        {"words": 128, "rep": 0, "prompt_tokens": 200, "cached_tokens": 64, "ttft_s": None,
         "completion_tokens": 1, "max_tokens": 1, "status": 200, "error": None}]}
    rep = validate_run(sm, raw, None)
    failed = {c["name"] for c in rep["checks"] if c["status"] == "FAIL"}
    assert not rep["ok"]
    assert "model answers 'capital of France' correctly" in failed
    assert "every request has a TTFT" in failed
    assert "cold requests really were cold (0 cached)" in failed


def test_parse_metric_accepts_prometheus_total_suffix():
    text = '# TYPE vllm:prefix_cache_hits counter\nvllm:prefix_cache_hits_total{model_name="m"} 42.0\n'
    assert parse_metric(text, "vllm:prefix_cache_hits") == 42.0


def test_smoke_and_calibration_never_share_an_image():
    from deploy.calibrate import _Prompts
    smoke_p, calib_p = _Prompts(seed=1), _Prompts(seed=2)
    a = {r.uuid for r in smoke_p.fresh_images(50, (448, 448))}
    b = {r.uuid for r in calib_p.fresh_images(50, (448, 448))}
    assert not a & b


def test_validation_flags_reused_cold_image():
    from deploy.validate import check_image_cold
    row = {"size": [448, 448], "images": 1, "rep": 0, "prompt_tokens": 300, "cached_tokens": 0,
           "mm_tokens": {"image": 256}, "ttft_s": 0.07, "completion_tokens": 1, "max_tokens": 1,
           "status": 200, "error": None, "image_uuids": ["img-10000-448x448"]}
    checks = check_image_cold([row], {"448x448": 256}, earlier_uuids={"img-10000-448x448"})
    assert any(c.status == "FAIL" and "new" in c.name for c in checks)
