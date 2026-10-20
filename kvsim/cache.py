"""Block 1: a per-worker prefix cache with a hard token budget and LRU eviction.

Key idea (same as vLLM automatic prefix caching / SGLang RadixAttention):
a request's prompt is a sequence of fixed-size blocks, and each block's identity
is the hash of *everything before it plus its own content*. So block i can only
be reused if blocks 0..i-1 were also identical. Reuse is therefore a
longest-common-PREFIX match, never a "bag of blocks" match.

In this simulator a block's identity is an integer id from a trie (see
workload.py): same id  <=>  same token sequence from the start of the prompt.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence


class LRUBlockCache:
    def __init__(self, capacity_blocks: int):
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        self.capacity = capacity_blocks
        # key order: first = least recently used, last = most recently used
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._lru)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._lru

    def match_len(self, blocks: Sequence[int], extra=None) -> int:
        """Longest cached prefix, in blocks. Read-only: does NOT refresh LRU.

        The router calls this to *look*; only an actual execution should
        count as a use. `extra` (optional) is a container of block ids that
        will be resident soon (queued requests); they count as present.
        """
        n = 0
        lru = self._lru
        for b in blocks:
            if b not in lru and (extra is None or b not in extra):
                break
            n += 1
        return n

    def touch(self, blocks: Iterable[int]) -> None:
        """Mark blocks as just used (LRU refresh). Missing blocks are ignored."""
        self._touch_deepest_first(list(blocks))

    def insert(self, blocks: Sequence[int]) -> None:
        """Make all blocks resident (prefill finished), evicting LRU as needed."""
        lru = self._lru
        for b in reversed(blocks):          # deepest first -> see invariant below
            if b in lru:
                lru.move_to_end(b)
            else:
                lru[b] = None
        while len(lru) > self.capacity:
            lru.popitem(last=False)
            self.evictions += 1

    def _touch_deepest_first(self, blocks: list[int]) -> None:
        lru = self._lru
        for b in reversed(blocks):
            if b in lru:
                lru.move_to_end(b)

    # Invariant (why we touch deepest-first): every time a block is used, its
    # whole prefix path is used too, with parents touched AFTER children. So a
    # parent is never older than its child, and LRU evicts leaves before the
    # shared trunk. Without this, evicting a parent would orphan its children:
    # they would still occupy memory but could never match again, because a
    # prefix match stops at the first missing block. vLLM does the same thing
    # (evict the longest-prefix block first among equally old ones).


class SequenceCache:
    """Engine-style prompt cache: whole prompts, not blocks.

    Models what llama.cpp's server (and therefore recent Ollama) does, as observed in its
    logs: after a request finishes, the whole prompt's KV state is kept as ONE entry; a new
    request reuses the entry with the longest common prefix; entries are evicted LRU when a
    memory budget is exceeded; an entry that is a strict prefix of a newer prompt is dropped
    as redundant.

    Difference from LRUBlockCache that matters for routing: shared prefixes are NOT
    deduplicated (every entry stores its own copy of the system prompt), so the same budget
    holds fewer conversations. Capacity is the sum of entry lengths, in blocks.
    """

    def __init__(self, capacity_blocks: int):
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        self.capacity = capacity_blocks
        self._entries: list[tuple[int, ...]] = []      # LRU first, MRU last
        self._total = 0
        self.evictions = 0

    def __len__(self) -> int:
        return self._total

    @staticmethod
    def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _best(self, blocks: Sequence[int]) -> tuple[int, int]:
        best_i, best = -1, 0
        for i, e in enumerate(self._entries):
            l = self._lcp(e, blocks)
            if l >= best and l > 0:          # ties -> more recently used entry
                best_i, best = i, l
        return best_i, best

    def match_len(self, blocks: Sequence[int], extra=None) -> int:
        """Longest common prefix with any cached prompt. `extra` is ignored: queued
        requests are not modeled as entries."""
        return self._best(blocks)[1]

    def touch(self, blocks: Iterable[int]) -> None:
        blocks = tuple(blocks)
        i, l = self._best(blocks)
        if i >= 0:
            self._entries.append(self._entries.pop(i))

    def insert(self, blocks: Sequence[int]) -> None:
        blocks = tuple(blocks)
        # already contained in a longer cached prompt -> just refresh it
        for i, e in enumerate(self._entries):
            if len(e) >= len(blocks) and e[: len(blocks)] == blocks:
                self._entries.append(self._entries.pop(i))
                return
        # drop cached prompts that are strict prefixes of this one (now redundant)
        kept = []
        for e in self._entries:
            if len(e) < len(blocks) and blocks[: len(e)] == e:
                self._total -= len(e)
            else:
                kept.append(e)
        self._entries = kept
        self._entries.append(blocks)
        self._total += len(blocks)
        while self._total > self.capacity and len(self._entries) > 1:
            self._total -= len(self._entries.pop(0))
            self.evictions += 1


def make_cache(kind: str, capacity_blocks: int):
    if kind == "block_lru":
        return LRUBlockCache(capacity_blocks)
    if kind == "sequence_lru":
        return SequenceCache(capacity_blocks)
    raise ValueError(f"unknown cache kind {kind!r}")
