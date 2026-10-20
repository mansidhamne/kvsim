"""Block 3: one simulated GPU prefill worker.

Modeling choices (all deliberate simplifications):
* Single-server FIFO prefill queue. Prefill is compute-bound and effectively
  serial per GPU, and TTFT = queue wait + prefill time. This matches the
  prefill worker of a disaggregated E/P/D deployment like Pinterest's. Decode
  is out of scope (it would need a separate model and doesn't benefit from
  prefix-aware routing in the same way).
* The cache lookup that determines the *real* hit happens when the request
  STARTS executing, not when it is routed. The router only had an estimate
  when it decided; things may have changed while the request waited.
* Blocks become resident only when prefill COMPLETES (insert at commit).
* The router can see two things per worker: what is resident, and (optionally)
  what is *pending* (blocks of requests already queued/running there, which
  will be resident soon). Dynamo's router similarly predicts blocks for routed
  requests instead of waiting for worker events.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from .cache import LRUBlockCache
from .workload import Request


@dataclass
class Queued:
    req: Request
    est_service_s: float    # router's estimate at routing time (uses cache knowledge, if any)
    blind_service_s: float  # cache-blind cost: full prompt recomputed


@dataclass
class Running:
    q: Queued
    start: float
    finish: float
    service_s: float


class Worker:
    def __init__(self, wid: int, capacity_blocks: int, cache=None):
        self.wid = wid
        self.cache = cache if cache is not None else LRUBlockCache(capacity_blocks)
        self.queue: deque[Queued] = deque()
        self.running: Optional[Running] = None
        self.queued_est_s = 0.0
        self.queued_blind_s = 0.0
        self._pending: dict[int, int] = {}   # block id -> #queued/running requests holding it
        self.busy_s = 0.0
        self.completed = 0

    # ---- what a router can observe -------------------------------------------------
    def _remaining_running(self, now: float) -> float:
        return max(0.0, self.running.finish - now) if self.running else 0.0

    def est_wait(self, now: float) -> float:
        """Estimated seconds until this worker would START a newly routed request."""
        return self._remaining_running(now) + self.queued_est_s

    def blind_wait(self, now: float) -> float:
        """Same, but a cache-blind balancer's view: every queued request costs a full prefill."""
        return self._remaining_running(now) + self.queued_blind_s

    def queue_len(self) -> int:
        return len(self.queue) + (1 if self.running else 0)

    def match_len(self, blocks: Sequence[int], include_pending: bool = False) -> int:
        """Longest prefix (in blocks) that is, or will soon be, reusable here. Read-only."""
        return self.cache.match_len(blocks, self._pending if include_pending else None)

    # ---- state transitions (driven by the simulator) -------------------------------
    def enqueue(self, q: Queued) -> None:
        self.queue.append(q)
        self.queued_est_s += q.est_service_s
        self.queued_blind_s += q.blind_service_s
        pend = self._pending
        for b in q.req.blocks:
            pend[b] = pend.get(b, 0) + 1

    def pop_next(self) -> Queued:
        q = self.queue.popleft()
        self.queued_est_s -= q.est_service_s
        self.queued_blind_s -= q.blind_service_s
        if not self.queue:               # kill float drift
            self.queued_est_s = self.queued_blind_s = 0.0
        return q

    def commit(self) -> Running:
        """Prefill finished: KV blocks become resident and reusable."""
        run = self.running
        assert run is not None
        self.cache.insert(run.q.req.blocks)
        pend = self._pending
        for b in run.q.req.blocks:
            c = pend[b] - 1
            if c:
                pend[b] = c
            else:
                del pend[b]
        self.busy_s += run.service_s
        self.completed += 1
        self.running = None
        return run
