"""Phase 3: prefix index built from what the router sent where (no engine feedback).

For each worker the router keeps:
* a resident set: an LRU block cache sized to the worker's KV capacity. A request's blocks are
  inserted when its first token arrives (prefill done = KV computed and cached).
* a pending set: blocks of requests routed there whose prefill hasn't finished. Counting them
  lets two follow-ups sent close together land on the same worker.

What it can't see (Phase 4 fixes this with KV events): evictions caused by the engine's own
memory pressure (running sequences, decode growth, preemption), and differences between the
router's approximate tokenization and the engine's. Its accuracy is measured per request by
comparing predicted vs actual cached tokens.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from kvsim.cache import LRUBlockCache


class ShadowIndex:
    def __init__(self, capacity_blocks: Sequence[int]):
        self.caches = [LRUBlockCache(max(1, c)) for c in capacity_blocks]
        self.pending: list[Counter] = [Counter() for _ in capacity_blocks]

    def __len__(self) -> int:
        return len(self.caches)

    def set_capacity(self, w: int, capacity_blocks: int) -> None:
        old = self.caches[w]
        new = LRUBlockCache(max(1, capacity_blocks))
        new.insert(list(old._lru.keys()))          # keep LRU order (oldest first)
        self.caches[w] = new

    def match(self, w: int, blocks: Sequence[int], include_pending: bool = True) -> int:
        return self.caches[w].match_len(blocks, self.pending[w] if include_pending else None)

    def resident_match(self, w: int, blocks: Sequence[int]) -> int:
        return self.caches[w].match_len(blocks)

    def on_routed(self, w: int, blocks: Sequence[int]) -> None:
        self.pending[w].update(blocks)

    def on_prefilled(self, w: int, blocks: Sequence[int]) -> None:
        self.caches[w].insert(list(blocks))
        self._unpend(w, blocks)

    def on_extended(self, w: int, blocks: Sequence[int]) -> None:
        """Generated tokens became cached after decoding (prompt + reply blocks)."""
        self.caches[w].insert(list(blocks))

    def on_dropped(self, w: int, blocks: Sequence[int]) -> None:
        """Request failed before its prefill finished."""
        self._unpend(w, blocks)

    def _unpend(self, w: int, blocks: Sequence[int]) -> None:
        p = self.pending[w]
        for b in blocks:
            c = p[b] - 1
            if c > 0:
                p[b] = c
            else:
                p.pop(b, None)

    def reset(self) -> None:
        self.caches = [LRUBlockCache(c.capacity) for c in self.caches]
        self.pending = [Counter() for _ in self.caches]
