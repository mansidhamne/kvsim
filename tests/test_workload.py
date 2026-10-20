from collections import defaultdict

from kvsim import CostModel, WorkloadConfig, generate_requests

COST = CostModel()


def test_deterministic_for_seed():
    a = generate_requests(WorkloadConfig(seed=3, active_convs=30), COST)
    b = generate_requests(WorkloadConfig(seed=3, active_convs=30), COST)
    assert [(r.arrival, r.blocks) for r in a] == [(r.arrival, r.blocks) for r in b]
    c = generate_requests(WorkloadConfig(seed=4, active_convs=30), COST)
    assert [r.arrival for r in a] != [r.arrival for r in c]


def test_arrivals_sorted_and_ids_dense():
    reqs = generate_requests(WorkloadConfig(active_convs=30), COST)
    assert [r.rid for r in reqs] == list(range(len(reqs)))
    assert all(x.arrival <= y.arrival for x, y in zip(reqs, reqs[1:]))


def test_next_turn_extends_previous_turn_prefix():
    reqs = generate_requests(WorkloadConfig(active_convs=30), COST)
    by_conv = defaultdict(list)
    for r in reqs:
        by_conv[r.conv_id].append(r)
    checked = 0
    for turns in by_conv.values():
        turns.sort(key=lambda r: r.turn)
        for prev, nxt in zip(turns, turns[1:]):
            assert nxt.blocks[: prev.n_blocks] == prev.blocks
            assert nxt.n_blocks > prev.n_blocks
            checked += 1
    assert checked > 0


def test_system_prompt_shared_by_everyone_but_own_images_are_not():
    wl = WorkloadConfig(active_convs=40, popular_prob=0.0)
    reqs = generate_requests(wl, COST)
    sys_blocks = -(-wl.system_tokens // COST.block_size)
    first = reqs[0].blocks[:sys_blocks]
    assert all(r.blocks[:sys_blocks] == first for r in reqs)
    firsts = {r.blocks[sys_blocks] for r in reqs if r.turn == 0}
    assert len(firsts) == len({r.conv_id for r in reqs})   # no image sharing at popular_prob=0


def test_cum_weight_consistent_and_images_cost_more():
    r = generate_requests(WorkloadConfig(active_convs=20), COST)[0]
    assert len(r.cum_weight) == r.n_blocks + 1
    assert r.cum_weight[-1] > r.n_blocks            # image blocks weigh > 1
    assert r.uncached_weight(r.n_blocks) == 0.0
