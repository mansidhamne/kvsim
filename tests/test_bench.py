"""Phase 1: workload determinism, mock worker cache behavior, end-to-end replay on mock workers."""
import asyncio

import aiohttp
import pytest

from bench.loadgen import run, summarize
from bench.mock_worker import MockCluster, MockConfig, block_hashes, tokenize
from bench.workload import (ImagePool, Trace, TraceConfig, build_messages, build_request,
                            fingerprint, generate_trace)

SMALL = TraceConfig(seed=3, active_convs=6, duration_s=40.0, think_mean_s=4.0, pool_size=8,
                    images_per_conv=(1, 3), system_words=60, user_words=(5, 10),
                    reply_words=(8, 16))


# ------------------------------------------------------------------------------- workload

def test_trace_is_deterministic_and_roundtrips():
    a, b = generate_trace(SMALL), generate_trace(SMALL)
    assert fingerprint(a) == fingerprint(b)
    assert a.n_requests() > 10
    c = Trace.from_json(a.to_json())
    assert fingerprint(c) == fingerprint(a)
    assert fingerprint(generate_trace(TraceConfig(**{**SMALL.__dict__, "seed": 4}))) != fingerprint(a)


def test_turn_times_increase_and_boards_are_shared():
    tr = generate_trace(TraceConfig(seed=0, active_convs=30, duration_s=200, board_prob=0.9,
                                    n_boards=2))
    for c in tr.conversations:
        times = [t.at for t in c.turns]
        assert times == sorted(times)
    boards = [c for c in tr.conversations if c.board is not None]
    assert len(boards) > len(tr.conversations) / 2
    same = [c for c in boards if c.board == boards[0].board]
    assert all(c.images == same[0].images for c in same)


def test_images_are_byte_identical_and_uuids_stable():
    tr = generate_trace(SMALL)
    pool1, pool2 = ImagePool(), ImagePool()
    conv = tr.conversations[0]
    for ref in conv.images:
        assert pool1.jpeg(ref) == pool2.jpeg(ref)
        assert pool1.jpeg(ref)[:2] == b"\xff\xd8"          # JPEG
    m1 = build_messages(tr, conv, 0, pool1)
    m2 = build_messages(tr, conv, len(conv.turns) - 1, pool2)
    assert m1[:2] == m2[:2]                                 # turn k's prompt starts with turn 0's
    img_parts = [p for p in m1[1]["content"] if p["type"] == "image_url"]
    assert [p["uuid"] for p in img_parts] == [r.uuid for r in conv.images]


def test_later_turns_extend_earlier_prompts():
    tr = generate_trace(SMALL)
    pool = ImagePool()
    conv = max(tr.conversations, key=lambda c: len(c.turns))
    prev = None
    for k in range(len(conv.turns)):
        msgs = build_messages(tr, conv, k, pool)
        if prev is not None:
            assert msgs[:len(prev)] == prev
            assert msgs[len(prev)]["role"] == "assistant"
        prev = msgs
    body, headers = build_request(tr, conv, 1, pool)
    assert body["max_tokens"] == conv.turns[1].max_tokens and body["stream"]
    assert headers["x-kvroute-conv"] == str(conv.conv_id)


# ------------------------------------------------------------------------------- mock worker

def test_mock_tokens_share_prefix_across_turns():
    tr, pool, cfg = generate_trace(SMALL), ImagePool(), MockConfig()
    conv = max(tr.conversations, key=lambda c: len(c.turns))
    h0 = block_hashes(tokenize(build_messages(tr, conv, 0, pool), cfg), cfg.block_size)
    h1 = block_hashes(tokenize(build_messages(tr, conv, 1, pool), cfg), cfg.block_size)
    assert len(h1) > len(h0) > 0
    # all but the last block of turn 0 (it holds the end-of-prompt marker) are reused
    assert h1[:len(h0) - 1] == h0[:len(h0) - 1]


def test_image_token_count_follows_size():
    cfg = MockConfig()
    pool = ImagePool()
    from bench.workload import ImageRef
    small = [{"role": "user", "content": [{"type": "image_url", "uuid": "a",
              "image_url": {"url": pool.data_url(ImageRef(0, 448, 448))}}]}]
    big = [{"role": "user", "content": [{"type": "image_url", "uuid": "b",
            "image_url": {"url": pool.data_url(ImageRef(0, 672, 448))}}]}]
    n_small = sum(t[0] == "img" for t in tokenize(small, cfg))
    n_big = sum(t[0] == "img" for t in tokenize(big, cfg))
    assert (n_small, n_big) == (256, 384)


def test_mock_worker_second_turn_hits_cache():
    tr, pool = generate_trace(SMALL), ImagePool()
    conv = max(tr.conversations, key=lambda c: len(c.turns))

    async def go():
        async with MockCluster(1, MockConfig(time_scale=50)) as cl:
            async with aiohttp.ClientSession() as s:
                usages = []
                for k in (0, 1):
                    body, h = build_request(tr, conv, k, pool)
                    body["stream"] = False
                    async with s.post(cl.urls[0] + "/v1/chat/completions", json=body, headers=h) as r:
                        assert r.status == 200
                        usages.append((await r.json())["usage"])
                async with s.get(cl.urls[0] + "/metrics") as r:
                    text = await r.text()
        return usages, text

    usages, text = asyncio.run(go())
    assert usages[0]["prompt_tokens_details"]["cached_tokens"] == 0
    assert usages[1]["prompt_tokens_details"]["cached_tokens"] > 0.5 * usages[1]["prompt_tokens"]
    assert "vllm:prefix_cache_hits" in text and "vllm:num_requests_waiting" in text


# ------------------------------------------------------------------------------- end to end

@pytest.mark.parametrize("pick", ["round_robin", "session"])
def test_replay_against_three_mock_workers(pick):
    tr, pool = generate_trace(SMALL), ImagePool()

    async def go():
        async with MockCluster(3, MockConfig(time_scale=20)) as cl:
            return await run(tr, pool, cl.urls, pick=pick, time_scale=20)

    recs = asyncio.run(go())
    s = summarize(recs)
    assert s["errors"] == 0 and s["ok"] == tr.n_requests()
    assert len(s["per_worker"]) == 3
    assert all(r.ttft_s is not None and r.output_chunks > 0 for r in recs)
    assert s["hit_rate"] is not None and s["hit_rate"] > 0
    if pick == "session":       # pinned conversations reuse their whole history
        assert s["hit_rate"] > 0.5
