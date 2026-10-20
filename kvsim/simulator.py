"""Block 5: discrete-event simulation loop.

Two event types, processed in time order (completions before arrivals on ties,
so the router always sees state that is exact "as of now"):

  ARRIVAL(req)    -> router picks a worker -> enqueue -> start it if the worker is idle
  COMPLETE(worker)-> commit KV blocks to the cache -> start the next queued request

Every prompt token of every request ends up in exactly one bucket:

  hit        served from this worker's cache
  routing    NOT here, but a prefix WAS cached on another worker  -> the router's miss
  evicted    reusable (seen before) but cached nowhere              -> capacity's miss
  cold       never computed before                                  -> unavoidable

"recomputed prefix tokens" = routing + evicted. Splitting them tells you whether
a bad latency number is a routing problem or simply a too-small cache.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cache import make_cache
from .config import ClusterConfig, CostModel
from .routers import Router
from .worker import Queued, Running, Worker
from .workload import Request

_COMPLETE, _ARRIVAL = 0, 1


@dataclass
class Record:
    rid: int
    wid: int
    arrival: float
    start: float
    finish: float
    prompt_tokens: int
    hit_tokens: int
    routing_tokens: int
    evicted_tokens: int
    cold_tokens: int
    service_s: float

    @property
    def ttft(self) -> float:        # MODELED time-to-first-token proxy
        return self.finish - self.arrival


@dataclass
class SimResult:
    records: list[Record]
    workers: list[Worker]


def simulate(requests: Sequence[Request], router: Router, cluster: ClusterConfig,
             cost: CostModel) -> SimResult:
    cap = cluster.cache_capacity_blocks(cost)
    workers = [Worker(i, cap, make_cache(cluster.cache_kind, cap)) for i in range(cluster.n_workers)]
    seen: set[int] = set()             # prefix nodes computed at least once, anywhere
    records: list[Record] = []
    bs = cost.block_size

    heap: list[tuple[float, int, int, object]] = []
    seq = 0
    for r in requests:
        heap.append((r.arrival, _ARRIVAL, seq, r))
        seq += 1
    heapq.heapify(heap)

    def start_next(w: Worker, now: float) -> None:
        nonlocal seq
        if w.running is not None or not w.queue:
            return
        q = w.pop_next()
        req = q.req
        blocks = req.blocks

        hit = w.cache.match_len(blocks)                 # the REAL hit, decided now
        w.cache.touch(blocks[:hit])

        best_peer = max((p.cache.match_len(blocks) for p in workers if p is not w), default=0)
        routing_blocks = max(0, best_peer - hit)
        rest = hit + routing_blocks
        evicted_blocks = 0
        for b in blocks[rest:]:                          # seen nodes form a prefix of `rest`
            if b not in seen:
                break
            evicted_blocks += 1
        cold_blocks = len(blocks) - rest - evicted_blocks
        seen.update(blocks)

        service = cost.prefill_time(req.uncached_weight(hit), hit)
        w.running = Running(q, now, now + service, service)
        records.append(Record(
            req.rid, w.wid, req.arrival, now, now + service,
            len(blocks) * bs, hit * bs, routing_blocks * bs, evicted_blocks * bs,
            cold_blocks * bs, service,
        ))
        heapq.heappush(heap, (now + service, _COMPLETE, seq, w))
        seq += 1

    while heap:
        now, kind, _, payload = heapq.heappop(heap)
        if kind == _COMPLETE:
            w = payload
            w.commit()
            start_next(w, now)
        else:
            req = payload
            wid, est = router.route(req, workers, now, cost)
            w = workers[wid]
            w.enqueue(Queued(req, est, cost.prefill_time(req.cum_weight[-1])))
            start_next(w, now)

    return SimResult(records, workers)
