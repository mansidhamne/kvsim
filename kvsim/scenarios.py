"""Hand-built micro-scenarios (no event loop) for probing a router's decision directly."""
from __future__ import annotations

import numpy as np

from .config import CostModel
from .routers import Router
from .worker import Queued, Running, Worker
from .workload import Request


def wait_or_recompute(router: Router, cost: CostModel, prefix_blocks: int, new_blocks: int,
                      queue_s: float) -> int:
    """Worker 0 holds the prefix but is busy for `queue_s` more seconds.
    Worker 1 is idle with an empty cache. Returns the worker the router picks
    (0 = wait for the cache, 1 = recompute everything on the idle GPU)."""
    cap = 10 * (prefix_blocks + new_blocks) + 16
    a, b = Worker(0, cap), Worker(1, cap)

    prefix = tuple(range(1, prefix_blocks + 1))
    a.cache.insert(prefix)

    # A is mid-way through unrelated work that finishes in `queue_s` seconds.
    filler = Request(-1, -1, 0, 0.0, (10_000_000,), (0.0, 1.0))
    a.enqueue(Queued(filler, queue_s, queue_s))
    a.running = Running(a.pop_next(), 0.0, queue_s, queue_s)

    blocks = prefix + tuple(range(5_000_000, 5_000_000 + new_blocks))
    cum = tuple(float(i) for i in range(len(blocks) + 1))   # all text blocks, weight 1
    req = Request(0, 0, 1, 0.0, blocks, cum)
    wid, _ = router.route(req, [a, b], 0.0, cost)
    return wid


def breakeven_queue_s(cost: CostModel, prefix_blocks: int, new_blocks: int) -> float:
    """Queue wait on A at which waiting and recomputing give equal TTFT.
    wait: q + cost(new)   recompute: cost(prefix + new)   ->  q* = cost(prefix) - overhead."""
    return cost.prefill_time(prefix_blocks + new_blocks) - cost.prefill_time(new_blocks)
