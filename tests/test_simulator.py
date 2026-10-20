import numpy as np
import pytest

from kvsim import (CacheAware, ClusterConfig, CostModel, LeastLoaded, RoundRobin,
                   WorkloadConfig, generate_requests, simulate, summarize)
from kvsim.scenarios import breakeven_queue_s, wait_or_recompute

COST = CostModel()
WL = WorkloadConfig(active_convs=60, horizon_s=200, warmup_s=50)
REQS = generate_requests(WL, COST)


def run(router, cluster=ClusterConfig()):
    return simulate(REQS, router, cluster, COST)


ROUTERS = [lambda: RoundRobin(), lambda: LeastLoaded(np.random.default_rng(0)),
           lambda: CacheAware(np.random.default_rng(0))]


@pytest.mark.parametrize("mk", ROUTERS)
def test_every_request_completes_and_tokens_are_conserved(mk):
    res = run(mk())
    assert len(res.records) == len(REQS)
    for r in res.records:
        assert r.start >= r.arrival - 1e-12
        assert r.finish == pytest.approx(r.start + r.service_s)
        assert r.ttft >= r.service_s - 1e-12
        assert (r.hit_tokens + r.routing_tokens + r.evicted_tokens + r.cold_tokens
                == r.prompt_tokens)
    assert sum(w.completed for w in res.workers) == len(REQS)
    for w in res.workers:                       # cache budget is respected
        assert len(w.cache) <= w.cache.capacity


@pytest.mark.parametrize("mk", ROUTERS)
def test_each_worker_serves_one_request_at_a_time(mk):
    res = run(mk())
    for wid in range(3):
        rs = sorted((r for r in res.records if r.wid == wid), key=lambda r: r.start)
        for a, b in zip(rs, rs[1:]):
            assert b.start >= a.finish - 1e-9


def test_policies_coincide_with_one_worker():
    one = ClusterConfig(n_workers=1)
    outs = [summarize(run(mk(), one), WL.warmup_s, WL.horizon_s) for mk in ROUTERS]
    assert outs[0] == outs[1] == outs[2]


def test_cache_aware_beats_cache_blind_on_multiturn_traffic():
    s = {n: summarize(run(mk()), WL.warmup_s, WL.horizon_s)
         for n, mk in zip(["rr", "ll", "ca"], ROUTERS)}
    assert s["ca"]["hit_rate"] > s["ll"]["hit_rate"]
    assert s["ca"]["recomputed_prefix_tokens_per_req"] < s["ll"]["recomputed_prefix_tokens_per_req"]


def test_infinite_cache_never_recomputes_for_eviction():
    big = ClusterConfig(cache_capacity_tokens=10**9)
    s = summarize(run(CacheAware(np.random.default_rng(0)), big), WL.warmup_s, WL.horizon_s)
    assert s["recomputed_evicted_tokens_per_req"] == 0.0


@pytest.mark.parametrize("prefix_blocks,new_blocks", [(40, 4), (120, 6), (400, 10)])
def test_wait_or_recompute_flips_at_breakeven(prefix_blocks, new_blocks):
    router = CacheAware(np.random.default_rng(0), alpha=1.0)
    q_star = breakeven_queue_s(COST, prefix_blocks, new_blocks)
    assert wait_or_recompute(router, COST, prefix_blocks, new_blocks, 0.9 * q_star) == 0
    assert wait_or_recompute(router, COST, prefix_blocks, new_blocks, 1.1 * q_star) == 1
    # a sticky router (alpha large) keeps waiting well past break-even
    sticky = CacheAware(np.random.default_rng(0), alpha=50.0)
    assert wait_or_recompute(sticky, COST, prefix_blocks, new_blocks, 1.5 * q_star) == 0


def test_session_affinity_pins_each_conversation_to_one_worker():
    from kvsim import make_router
    res = run(make_router("session_affinity", np.random.default_rng(0)))
    homes = {}
    by_rid = {r.rid: r for r in REQS}
    for rec in res.records:
        homes.setdefault(by_rid[rec.rid].conv_id, set()).add(rec.wid)
    assert all(len(w) == 1 for w in homes.values())


def test_sequence_cache_simulation_conserves_tokens_and_respects_budget():
    cl = ClusterConfig(cache_kind="sequence_lru", cache_capacity_tokens=40_000)
    res = run(CacheAware(np.random.default_rng(0)), cl)
    assert len(res.records) == len(REQS)
    for r in res.records:
        assert (r.hit_tokens + r.routing_tokens + r.evicted_tokens + r.cold_tokens
                == r.prompt_tokens)
    for w in res.workers:
        assert len(w.cache) <= w.cache.capacity + max(len(r.blocks) for r in REQS)


def test_attention_term_makes_hits_cost_something():
    flat = CostModel()
    attn = CostModel(attn_s_per_token_pair=1e-8)
    assert attn.prefill_time(4, cached_blocks=100) > flat.prefill_time(4, cached_blocks=100)
    assert attn.prefill_time(4, cached_blocks=0) == flat.prefill_time(4, cached_blocks=0)
