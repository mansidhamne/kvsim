"""Routing policies for the real router.

The four policies from the simulator (`kvsim.routers`) run UNCHANGED here: this module only
adapts the router's live state to the interface they expect. So a policy result measured on
GPUs and the same policy in the simulator are the same code.

    kvsim policy expects             live router provides
    worker.est_wait(now)        <-   prefill backlog using cache-aware estimates
    worker.blind_wait(now)      <-   prefill backlog counting every prompt as fully uncached
    worker.match_len(blocks, p) <-   shadow index (resident + pending blocks)
    req.blocks / uncached_weight<-   PromptInfo from router.hashing
    req.conv_id                 <-   x-kvroute-conv header (session key)

Extra router-only baseline: `least_requests` (fewest requests in flight), what a generic
HTTP load balancer does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kvsim.config import CostModel
from kvsim.routers import make_router as make_sim_router

from .hashing import PromptInfo
from .index.shadow import ShadowIndex
from .load import WorkerLoad

POLICIES = ("round_robin", "least_requests", "least_loaded", "session_affinity", "cache_aware")
ALL_POLICIES = POLICIES + ("cache_aware_v2",)


@dataclass
class _Req:
    """kvsim.workload.Request look-alike."""
    info: PromptInfo
    conv_id: str

    @property
    def blocks(self):
        return self.info.blocks

    @property
    def cum_weight(self):
        return self.info.cum_weight + (self.info.cum_weight[-1] + self.info.tail_weight,)

    def uncached_weight(self, hit_blocks: int) -> float:
        return self.info.uncached_weight(hit_blocks)

    def uncached_tokens(self, hit_blocks: int) -> int:
        return self.info.uncached_tokens(hit_blocks)


class _WorkerAdapter:
    """kvsim.worker.Worker look-alike over live state."""

    def __init__(self, wid: int, load: WorkerLoad, index: ShadowIndex):
        self.wid, self.load, self.index = wid, load, index

    def est_wait(self, now: float) -> float:
        return self.load.backlog_s(now)

    def blind_wait(self, now: float) -> float:
        return self.load.backlog_s(now, blind=True)

    def match_len(self, blocks, include_pending: bool = False) -> int:
        return self.index.match(self.wid, blocks, include_pending)

    def queue_len(self) -> int:
        return self.load.n_inflight()


class LeastRequests:
    name = "least_requests"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def route(self, req, workers, now, cost):
        counts = [w.queue_len() for w in workers]
        best = min(counts)
        ties = [i for i, c in enumerate(counts) if c == best]
        wid = ties[0] if len(ties) == 1 else int(self.rng.choice(ties))
        return wid, cost.request_prefill_time(req, 0)


class CacheAwareV2:
    """Cache-aware routing, version 2. Two changes from `cache_aware`, each from a measured gap:

    1. Decode-aware estimate. The v1 score (prefill backlog + estimated prefill) ignored requests
       that are DECODING on the worker; on 2x T4 the measured TTFT / estimate ratio grows from
       1.24 (2-3 requests in flight) to 1.39 (10-12) and 1.63 (13+) over 30,509 routed requests
       (phase 3b runs). Here the estimate is scaled by (1 + decode_k * requests in flight),
       decode_k = 0.02 fitted from those per-bin medians. (The ~1.25x constant part of the error
       is the same on every worker, so it doesn't change the choice and is not modeled.)
    2. Switching cost. Moving a follow-up away from the worker that served its conversation
       ("home") while home still holds its prefix wastes that cache and duplicates it elsewhere.
       Block A (25% cache): v1's moves lost the cache every time and cost +12% TTFT p99 vs
       session affinity; block B2: v1's moves of conversations whose cache was ALREADY evicted
       were free and helped. So: other workers pay `switch_beta` x the prefill time of the
       prefix still cached on home; nothing if home's copy is gone.

        score(w) = (backlog(w) + prefill(req | cached on w)) * (1 + decode_k * inflight(w))
                   + [w != home] * switch_beta * prefill(prefix cached on home)
    """

    name = "cache_aware_v2"

    def __init__(self, rng: np.random.Generator, decode_k: float = 0.02, switch_beta: float = 1.0,
                 pending_aware: bool = True):
        self.rng = rng
        self.decode_k = decode_k
        self.switch_beta = switch_beta
        self.pending_aware = pending_aware
        self._home: dict = {}

    def route(self, req, workers, now, cost):
        hits = [w.match_len(req.blocks, include_pending=self.pending_aware) for w in workers]
        home = self._home.get(req.conv_id)
        penalty = 0.0
        if home is not None and home < len(workers) and hits[home] > 0:
            h = hits[home]
            raw = h * cost.block_size
            penalty = self.switch_beta * cost.prefill_time(req.cum_weight[h], 0, new_tokens=raw)
        scores, ests = [], []
        for i, w in enumerate(workers):
            est = cost.request_prefill_time(req, hits[i])
            ests.append(est)
            s = (w.est_wait(now) + est) * (1.0 + self.decode_k * w.queue_len())
            if home is not None and i != home:
                s += penalty
            scores.append(s)
        best = min(scores)
        ties = [i for i, x in enumerate(scores) if x <= best + 1e-9]
        wid = ties[0] if len(ties) == 1 else int(self.rng.choice(ties))
        self._home[req.conv_id] = wid
        return wid, ests[wid]


@dataclass
class Decision:
    wid: int
    est_prefill_s: float            # router's estimate on the chosen worker (cache-aware)
    full_prefill_s: float           # estimate if nothing were cached
    predicted_hit_blocks: int       # on the chosen worker, resident + pending
    predicted_resident_blocks: int  # resident only (what the engine can actually reuse now)
    scores: list[dict]              # per worker: backlog, hit blocks, est prefill


def make_policy(name: str, seed: int = 0, **kw):
    rng = np.random.default_rng(seed)
    if name == "least_requests":
        return LeastRequests(rng)
    if name == "cache_aware_v2":
        return CacheAwareV2(rng, **kw)
    if name in ("round_robin", "least_loaded", "session_affinity", "cache_aware"):
        return make_sim_router(name, rng, **kw)
    raise ValueError(f"unknown policy {name!r}; choose from {ALL_POLICIES}")


def decide(policy, info: PromptInfo, conv_id: str, loads: Sequence[WorkerLoad],
           index: ShadowIndex, cost: CostModel, now: float) -> Decision:
    workers = [_WorkerAdapter(i, l, index) for i, l in enumerate(loads)]
    req = _Req(info, conv_id)
    wid, est = policy.route(req, workers, now, cost)
    full = cost.request_prefill_time(req, 0)
    scores = []
    for w in workers:
        hit = w.match_len(info.blocks, include_pending=True)
        scores.append({"backlog_s": round(w.est_wait(now), 4), "hit_blocks": hit,
                       "est_prefill_s": round(cost.request_prefill_time(req, hit), 4),
                       "inflight": w.queue_len()})
    hit_chosen = scores[wid]["hit_blocks"]
    est_chosen = cost.request_prefill_time(req, hit_chosen)
    return Decision(wid, est_chosen, full, hit_chosen,
                    index.resident_match(wid, info.blocks), scores)
