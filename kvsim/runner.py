"""Run policies on the SAME generated trace, optionally in parallel across jobs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .config import ClusterConfig, CostModel, WorkloadConfig
from .metrics import summarize
from .routers import make_router
from .simulator import simulate
from .workload import generate_requests


@dataclass(frozen=True)
class PolicySpec:
    label: str
    name: str                                  # round_robin | least_loaded | cache_aware
    kwargs: tuple[tuple[str, Any], ...] = ()   # hashable so specs can cross process boundaries


DEFAULT_POLICIES = (
    PolicySpec("round-robin", "round_robin"),
    PolicySpec("least-loaded", "least_loaded"),
    PolicySpec("cache-aware", "cache_aware"),
)


@dataclass(frozen=True)
class Job:
    wl: WorkloadConfig
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    cost: CostModel = field(default_factory=CostModel)
    policies: tuple[PolicySpec, ...] = DEFAULT_POLICIES
    tag: Any = None                            # opaque label the caller uses to find results


def run_job(job: Job) -> tuple[Any, dict[str, dict[str, float]]]:
    reqs = generate_requests(job.wl, job.cost)          # generated once, reused by every policy
    out: dict[str, dict[str, float]] = {}
    for spec in job.policies:
        # router RNG only breaks ties; seeded identically per policy for reproducibility
        router = make_router(spec.name, np.random.default_rng(job.wl.seed + 12345),
                             **dict(spec.kwargs))
        res = simulate(reqs, router, job.cluster, job.cost)
        out[spec.label] = summarize(res, job.wl.warmup_s, job.wl.horizon_s)
    return job.tag, out


def run_jobs(jobs: Iterable[Job], workers: int | None = None):
    jobs = list(jobs)
    if workers == 1 or len(jobs) == 1:
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_job, jobs))
