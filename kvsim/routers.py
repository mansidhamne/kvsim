"""Block 4: the three routing policies.

All routers implement  route(req, workers, now, cost) -> (worker_id, est_service_s)
where est_service_s is the prefill time the router *expects* on that worker; the
worker adds it to its queue-wait estimate for later routing decisions.

The scoring function of the cache-aware router is the interesting part:

    score(w) = est_wait(w)  +  alpha * est_prefill(req | prefix cached on w)
               ^ queue penalty   ^ cost of the tokens we'd still have to recompute

With alpha = 1 this is exactly *estimated TTFT on worker w*, so "wait for the
worker with the prefix, or recompute on the idle one?" is answered by the
arithmetic of the score instead of a hand-tuned threshold:

    wait on A  <=>  queue_A + cost(uncached | A)  <  queue_B + cost(uncached | B)

alpha > 1 over-weights cache affinity (sticky); alpha < 1 under-weights it
(closer to least-loaded). As I understand NVIDIA Dynamo's KV router (NOT stated in the
Pinterest post; verify against the Dynamo docs), it has a similar shape: a weight on
new-prefill blocks plus a load term, with a softmax temperature on top.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .config import CostModel
from .worker import Worker
from .workload import Request


class Router(Protocol):
    name: str

    def route(self, req: Request, workers: Sequence[Worker], now: float,
              cost: CostModel) -> tuple[int, float]: ...


def _argmin_random_tiebreak(scores: Sequence[float], rng: np.random.Generator,
                            eps: float = 1e-9) -> int:
    best = min(scores)
    ties = [i for i, s in enumerate(scores) if s <= best + eps]
    return ties[0] if len(ties) == 1 else int(rng.choice(ties))


class RoundRobin:
    name = "round_robin"

    def __init__(self) -> None:
        self._next = 0

    def route(self, req, workers, now, cost):
        wid = self._next % len(workers)
        self._next += 1
        return wid, cost.prefill_time(req.cum_weight[-1])


class LeastLoaded:
    """Send to the worker with the least outstanding work, as a cache-blind balancer sees it
    (every queued request counts as a full, uncached prefill)."""

    name = "least_loaded"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def route(self, req, workers, now, cost):
        wid = _argmin_random_tiebreak([w.blind_wait(now) for w in workers], self.rng)
        return wid, cost.prefill_time(req.cum_weight[-1])


class SessionAffinity:
    """Extra baseline (not one of the three required policies): the obvious cheap alternative.
    A conversation's first turn goes to the least-loaded worker; every later turn is pinned
    to that same worker. It never looks at what is actually cached, so it can't share
    prefixes ACROSS conversations, can't notice an eviction, and can't rebalance."""

    name = "session_affinity"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._home: dict[int, int] = {}

    def route(self, req, workers, now, cost):
        wid = self._home.get(req.conv_id)
        if wid is None:
            wid = _argmin_random_tiebreak([w.blind_wait(now) for w in workers], self.rng)
            self._home[req.conv_id] = wid
        return wid, cost.prefill_time(req.cum_weight[-1])


class CacheAware:
    name = "cache_aware"

    def __init__(self, rng: np.random.Generator, alpha: float = 1.0,
                 pending_aware: bool = True):
        self.rng = rng
        self.alpha = alpha
        self.pending_aware = pending_aware

    def route(self, req, workers, now, cost):
        scores, ests = [], []
        for w in workers:
            hit = w.match_len(req.blocks, include_pending=self.pending_aware)
            est = cost.prefill_time(req.uncached_weight(hit), hit)
            ests.append(est)
            scores.append(w.est_wait(now) + self.alpha * est)
        wid = _argmin_random_tiebreak(scores, self.rng)
        return wid, ests[wid]


def make_router(name: str, rng: np.random.Generator, **kw) -> Router:
    if name == "round_robin":
        return RoundRobin()
    if name == "least_loaded":
        return LeastLoaded(rng)
    if name == "cache_aware":
        return CacheAware(rng, **kw)
    if name == "session_affinity":
        return SessionAffinity(rng)
    raise ValueError(f"unknown policy {name!r}")
