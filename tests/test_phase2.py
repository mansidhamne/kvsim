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
    assert 30_000 < cm["prefill_tokens_per_s"] < 200_000
    assert out["hits"]["hits_text.new_question"]["cached_frac"] > 0.8
    assert out["hits"]["hits_text.cold"]["cached_frac"] == 0
    assert [r["concurrency"] for r in out["load"]] == [1, 4]
