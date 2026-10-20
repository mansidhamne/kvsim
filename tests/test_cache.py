from kvsim.cache import LRUBlockCache


def test_prefix_semantics_stop_at_first_miss():
    c = LRUBlockCache(10)
    c.insert([1, 2, 3])
    assert c.match_len([1, 2, 3, 4]) == 3
    assert c.match_len([9, 2, 3]) == 0          # later blocks don't count without the prefix
    assert c.match_len([1, 2, 9, 3]) == 2


def test_match_len_is_read_only():
    c = LRUBlockCache(3)
    c.insert([1])
    c.insert([2])
    c.insert([3])
    c.match_len([1])                            # looking must not refresh block 1
    c.insert([4])                               # evicts LRU = 1
    assert 1 not in c and 2 in c


def test_capacity_is_a_hard_bound():
    c = LRUBlockCache(5)
    for i in range(100):
        c.insert([i, i + 1000])
        assert len(c) <= 5
    assert c.evictions > 0


def test_lru_order_and_touch():
    c = LRUBlockCache(3)
    c.insert([1])
    c.insert([2])
    c.insert([3])
    c.touch([1])                                # 1 becomes most recent
    c.insert([4])                               # evicts 2
    assert 2 not in c and 1 in c and 3 in c and 4 in c


def test_no_orphans_leaf_evicted_before_trunk():
    """Shared trunk [1,2] with two branches; under pressure branches go first."""
    c = LRUBlockCache(6)
    c.insert([1, 2, 3, 4])
    c.insert([1, 2, 5, 6])
    c.insert([1, 2, 7, 8])                      # forces evictions
    assert 1 in c and 2 in c, "shared trunk must outlive its leaves"
    chains = [[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 7, 8]]
    for ch in chains:
        n = c.match_len(ch)
        assert all((b in c) == (i < n) for i, b in enumerate(ch)), "orphaned block present"


# ---- SequenceCache (llama.cpp-style whole-prompt cache) -------------------------------
from kvsim.cache import SequenceCache, make_cache  # noqa: E402


def test_sequence_cache_longest_common_prefix_across_entries():
    c = SequenceCache(100)
    c.insert([1, 2, 3, 4])
    c.insert([1, 2, 9, 9, 9])
    assert c.match_len([1, 2, 3, 4, 5]) == 4
    assert c.match_len([1, 2, 9, 9, 9, 9]) == 5
    assert c.match_len([1, 2, 7]) == 2
    assert c.match_len([8, 8]) == 0


def test_sequence_cache_replaces_prefix_entries_and_dedups_nothing_else():
    c = SequenceCache(100)
    c.insert([1, 2, 3])
    c.insert([1, 2, 3, 4, 5])             # extends the first -> first is redundant
    assert len(c) == 5 and c.match_len([1, 2, 3]) == 3
    c.insert([1, 2, 7, 8])                # shares only [1, 2]: stored in full, no dedup
    assert len(c) == 5 + 4


def test_sequence_cache_budget_and_lru_eviction():
    c = SequenceCache(10)
    c.insert([1, 1, 1, 1])
    c.insert([2, 2, 2, 2])
    c.touch([1, 1, 1, 1])                 # entry 1 becomes most recent
    c.insert([3, 3, 3, 3])                # 12 > 10 -> evict LRU = entry 2
    assert len(c) <= 10
    assert c.match_len([2, 2, 2, 2]) == 0 and c.match_len([1, 1, 1, 1]) == 4
    assert c.evictions == 1


def test_make_cache_kinds():
    assert isinstance(make_cache("block_lru", 4), LRUBlockCache)
    assert isinstance(make_cache("sequence_lru", 4), SequenceCache)
