"""Phase 3b: per-token ITL, think time after the reply, trace builders, full cache resets,
policy-order rotation, cross-seed analysis and validation. No downloads: VisionArena-shaped
rows are synthesized here."""
import asyncio
import io

import numpy as np
from PIL import Image

from bench.analyze import aggregate, paired
from bench.loadgen import run as replay
from bench.mock_worker import MockCluster, MockConfig
from bench.traces import mooncake as mc
from bench.traces import visionarena as va
from bench.traces.servegen import ThinkTime, load
from bench.traces.stats import workload_stats
from bench.workload import ImagePool, TraceConfig, generate_trace
from deploy.phase3b import reset_all, rotated
from deploy.validate import validate_phase3b


def _png(seed: int, size=(300, 200)) -> bytes:
    img = Image.new("RGB", size, (seed * 37 % 256, seed * 91 % 256, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _va_rows(n=40, shared_every=5):
    rows = []
    for i in range(n):
        img = _png(0 if i % shared_every == 0 else i)          # every 5th conversation: same image
        turns = 1 if i % 3 else 3
        conv = []
        for k in range(turns):
            conv.append([{"role": "user", "content": f"question {k} about the picture {i}"}])
            conv.append([{"role": "assistant", "content": " ".join(["answer"] * (20 + k))}])
        rows.append({"images": [{"bytes": img}], "conversation": conv, "num_turns": turns})
    return rows


def test_servegen_think_time_distribution_is_committed_and_scaled():
    d = load()
    assert d["follow_ups"] > 1000 and len(d["think_s_quantiles"]) == 101
    s = ThinkTime(scale=0.1).sample(np.random.default_rng(0), 2000)
    assert abs(np.median(s) - 0.1 * d["think_s_quantiles"][50]) < 0.15 * d["think_s_quantiles"][50]


def test_visionarena_builder_keeps_real_structure():
    rows = _va_rows()
    tr, pool, stats = va.build_trace(rows, va.VAConfig(conv_rate=0.5, duration_s=60, seed=1,
                                                       think_scale=0.01))
    assert tr.source == "visionarena" and tr.system_text == ""
    assert stats["conversations"] == len(tr.conversations) > 10
    c = next(c for c in tr.conversations if len(c.turns) > 1)
    assert c.turns[1].think_s is not None and c.turns[0].think_s is None
    assert c.images[0].blob in pool.blobs and pool.data_url(c.images[0]).startswith("data:image/png")
    # the same image bytes in different conversations -> the same uuid (cache key)
    by_img = {}
    for cv in tr.conversations:
        by_img.setdefault(cv.images[0].uuid, []).append(cv.conv_id)
    assert any(len(v) > 1 for v in by_img.values()) and stats["images_in_2plus_convs"] >= 1
    ws = workload_stats(tr, pool)
    assert 0 < ws["reuse_ceiling"] < 1 and ws["peak_live_blocks"] > 0


def test_visionarena_drops_follow_ups_after_the_window():
    rows = _va_rows()
    tr, _, stats = va.build_trace(rows, va.VAConfig(conv_rate=0.5, duration_s=30, seed=2,
                                                    think_scale=10.0, tail_grace_s=0))
    assert stats["follow_ups_after_window_dropped"] > 0
    assert all(t.at <= 30 for c in tr.conversations for t in c.turns)


def test_mooncake_builder_shares_prefixes_like_the_hash_ids():
    rows = [{"timestamp": 0, "input_length": 1500, "output_length": 40, "hash_ids": [1, 2, 3]},
            {"timestamp": 500, "input_length": 2000, "output_length": 40, "hash_ids": [1, 2, 3, 4]},
            {"timestamp": 900, "input_length": 500, "output_length": 40, "hash_ids": [9]}]
    tr, stats = mc.build_trace(rows, mc.MooncakeConfig(start_s=0, duration_s=10, tokens_per_hash=8))
    a, b, c = (cv.turns[0].user_text for cv in tr.conversations)
    assert b.startswith(a) and not c.startswith(a[:10])
    assert stats["requests"] == 3 and tr.system_text == ""
    ws = workload_stats(tr, ImagePool())
    assert 0.15 < ws["reuse_ceiling"] < 0.8       # the second request reuses the first's blocks


def test_think_after_reply_and_per_token_itl_on_mock():
    tr = generate_trace(TraceConfig(seed=3, active_convs=4, duration_s=20, think_mean_s=2,
                                    think_min_s=1, pool_size=4, images_per_conv=(1, 1),
                                    system_words=20, user_words=(3, 5), reply_words=(12, 16)))

    async def go():
        async with MockCluster(1, MockConfig(time_scale=10, tokens_per_chunk=3)) as cl:
            return await replay(tr, ImagePool(), cl.urls, time_scale=10)

    recs = asyncio.run(go())
    by = {(r.conv_id, r.turn): r for r in recs}
    for r in recs:
        # one ITL value per token after the first chunk (which carries 3 tokens and the TTFT)
        assert r.error is None and len(r.itl_s) == r.completion_tokens - 3
        assert r.multi_token_chunks > 0
        if r.turn:
            prev = by[(r.conv_id, r.turn - 1)]
            think = tr.conversations[r.conv_id].turns[r.turn].think_s / 10
            # sent think time after the previous reply finished, not on a fixed clock
            assert r.sent_s >= prev.sent_s + prev.e2e_s + think - 0.02
            assert r.late_s < 0.05


def test_reset_all_hits_all_three_caches():
    async def go():
        async with MockCluster(2, MockConfig()) as cl:
            res = await reset_all(cl.urls)
            return res, [e.other_resets for e in cl.engines()]

    res, other = asyncio.run(go())
    assert res == [{"prefix": True, "mm": True, "encoder": True}] * 2 and other == [2, 2]


def test_rotation_puts_every_policy_in_every_position():
    pols = ("a", "b", "c", "d")
    positions = {p: {rotated(pols, k).index(p) for k in range(4)} for p in pols}
    assert all(v == {0, 1, 2, 3} for v in positions.values())


def _cell(policy, seed, p99, hit, order, **kw):
    s = {"hit_rate": hit, "ttft_p50_s": p99 / 4, "ttft_p99_s": p99, "itl_p99_s": 0.1,
         "goodput_frac": 0.9, "recomputed_tokens_per_req": 100, "errors": 0, "requests": 100,
         "sample_outputs": [], "late_frac": 0.0}
    return {"workload": "visionarena", "cache": "full", "load": 0.3, "seed": seed, "policy": policy,
            "order_index": order, "reuse_ceiling": 0.5, "trace_fingerprint": f"fp{seed}",
            "resets": [{"prefix": True, "mm": True, "encoder": True}],
            "server": {"prefix_hits": 10.0, "preemptions": 0.0},
            "client_all": {"cached_tokens": 10, "requests": 100}, "summary": s, **kw}


def test_cross_seed_aggregate_and_paired_comparison():
    cells = [_cell("round_robin", s, 3.0 + s, 0.30, s % 2) for s in range(3)] + \
            [_cell("cache_aware", s, 2.0 + s, 0.45, (s + 1) % 2) for s in range(3)]
    rows = {r["policy"]: r for r in aggregate(cells)}
    assert rows["cache_aware"]["ttft_p99_s"]["median"] == 3.0
    assert abs(rows["cache_aware"]["share_of_ceiling"] - 0.9) < 1e-9
    (p,) = paired(cells, "cache_aware", "round_robin", "ttft_p99_s")
    assert p["seeds"] == 3 and p["same_sign_all_seeds"] and p["max"] < 0
    rep = validate_phase3b(cells)
    assert rep["ok"], [c for c in rep["checks"] if c["status"] == "FAIL"]


def test_validation_flags_fixed_order_and_missing_cache_reset():
    cells = [_cell("round_robin", s, 3.0, 0.3, 0) for s in range(2)] + \
            [_cell("cache_aware", s, 2.0, 0.4, 1) for s in range(2)]
    cells[0]["resets"] = [{"prefix": True, "mm": False, "encoder": True}]
    rep = validate_phase3b(cells)
    failed = {c["name"].split(": ", 1)[-1] for c in rep["checks"] if c["status"] == "FAIL"}
    assert "policy order rotated across seeds" in failed
    assert "multimodal and encoder caches reset" in failed


def test_text_pieces_count_scripts_without_spaces():
    from router.hashing import text_pieces
    assert len(text_pieces("describe the picture")) == 3
    zh = "请详细描述这张图片中的所有内容以及颜色"            # 19 characters, no spaces
    assert len(text_pieces(zh)) == len(zh)
    assert len(text_pieces("https://example.com/a/very/long/path")) > 3
    assert text_pieces("同じ文") == text_pieces("同じ文")        # deterministic


def test_visionarena_length_filter_catches_long_cjk_conversations():
    rows = _va_rows(n=20)
    long_zh = "图" * 9000                                     # 1 "word" by whitespace, ~9k tokens
    rows[3]["conversation"] = [[{"role": "user", "content": long_zh}],
                               [{"role": "assistant", "content": "ok"}]]
    _, _, stats = va.build_trace(rows, va.VAConfig(conv_rate=0.2, duration_s=40, seed=0))
    assert stats["dropped"]["too_long"] == 1
    assert stats["length_filter"].startswith("router estimate")


def _multi_turn_trace():
    return generate_trace(TraceConfig(seed=4, active_convs=3, duration_s=15, think_mean_s=1.5,
                                      think_min_s=1, turns=(3, 3), pool_size=3, images_per_conv=(1, 1),
                                      system_words=20, user_words=(4, 6), reply_words=(30, 40)))


def _follow_up_reuse(history: str):
    from bench.workload import RequestOptions
    from router.server import RouterConfig, RouterRunner
    from kvsim.config import CostModel
    tr = _multi_turn_trace()

    async def go():
        async with MockCluster(2, MockConfig(time_scale=10)) as cl:
            async with RouterRunner(RouterConfig(cl.urls, policy="session_affinity",
                                                 cost=CostModel(block_size=16))) as rr:
                recs = await replay(tr, ImagePool(), [rr.url], time_scale=10,
                                    opts=RequestOptions(history=history))
                return recs, list(rr.router.log_rows)

    recs, log = asyncio.run(go())
    by = {(r.conv_id, r.turn): r for r in recs}
    fu = [(r, by[(r.conv_id, r.turn - 1)]) for r in recs if r.turn > 0]
    beyond_prev_prompt = [r.cached_tokens > p.prompt_tokens + 16 for r, p in fu]
    return recs, log, beyond_prev_prompt


def test_own_history_lets_follow_ups_reuse_the_generated_reply():
    _, _, canned = _follow_up_reuse("canned")
    recs, log, own = _follow_up_reuse("own")
    assert not any(canned)               # canned replies: cache stops at the previous prompt
    assert all(own)                      # own replies: the reply's blocks are reused as well
    assert all(r.error is None and r.reply_sha for r in recs)
    # the router's index includes the reply blocks too, so its predictions stay accurate
    fu = [d for d in log if d.get("turn") not in (None, "0")]
    assert all(abs(d["predicted_resident_tokens"] - d["actual_cached_tokens"]) <= 32 for d in fu)


def test_validation_reports_reply_divergence():
    a = _cell("round_robin", 0, 3.0, 0.3, 0, history="own", reply_shas={"0:0": "x", "0:1": "y"})
    b = _cell("cache_aware", 0, 2.0, 0.4, 1, history="own", reply_shas={"0:0": "x", "0:1": "z"})
    for c in (a, b):
        c["server"] = {**c["server"], "prefix_queries": 1000.0}
    rep = validate_phase3b([a, b])
    (c,) = [c for c in rep["checks"] if "same prompt tokens" in c["name"]]
    assert c["status"] == "PASS" and "50.0%" in c["detail"]     # same totals, wording differs
    b["server"] = {**b["server"], "prefix_queries": 1000.0}
    a["server"] = {**a["server"], "prefix_queries": 900.0}
    rep = validate_phase3b([a, b])
    (c,) = [c for c in rep["checks"] if "same prompt tokens" in c["name"]]
    assert c["status"] == "FAIL"
