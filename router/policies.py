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
        return wid, cost.prefill_time(req.cum_weight[-1])


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
    if name in ("round_robin", "least_loaded", "session_affinity", "cache_aware"):
        return make_sim_router(name, rng, **kw)
    raise ValueError(f"unknown policy {name!r}; choose from {POLICIES}")


def decide(policy, info: PromptInfo, conv_id: str, loads: Sequence[WorkerLoad],
           index: ShadowIndex, cost: CostModel, now: float) -> Decision:
    workers = [_WorkerAdapter(i, l, index) for i, l in enumerate(loads)]
    req = _Req(info, conv_id)
    wid, est = policy.route(req, workers, now, cost)
    full = cost.prefill_time(info.uncached_weight(0))
    scores = []
    for w in workers:
        hit = w.match_len(info.blocks, include_pending=True)
        scores.append({"backlog_s": round(w.est_wait(now), 4), "hit_blocks": hit,
                       "est_prefill_s": round(cost.prefill_time(info.uncached_weight(hit), hit), 4),
                       "inflight": w.queue_len()})
    hit_chosen = scores[wid]["hit_blocks"]
    est_chosen = cost.prefill_time(info.uncached_weight(hit_chosen), hit_chosen)
    return Decision(wid, est_chosen, full, hit_chosen,
                    index.resident_match(wid, info.blocks), scores)
