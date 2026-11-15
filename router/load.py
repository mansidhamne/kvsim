"""Per-worker load signals the router can observe.

* Router-side, exact and instant: requests in flight, and for each one whether its prefill has
  finished (first token seen) plus the router's estimate of its prefill time.
* Engine-side, scraped from /metrics every ~0.5 s: requests running/waiting, KV usage, and the
  KV capacity (vllm:cache_config_info, num_gpu_blocks).

Prefill backlog estimate (seconds until a new request's prefill could start), assuming the
engine works through pending prefills roughly in order:
    backlog = max(0, sum(est of requests still prefilling) - time since the oldest was sent)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


@dataclass
class InFlight:
    rid: str
    blocks: tuple[int, ...]
    est_prefill_s: float
    full_prefill_s: float          # cache-blind estimate (whole prompt recomputed)
    sent_at: float
    first_token_at: float | None = None


@dataclass
class WorkerLoad:
    name: str
    url: str
    inflight: dict[str, InFlight] = field(default_factory=dict)
    scraped: dict = field(default_factory=dict)
    scraped_at: float = 0.0
    routed: int = 0

    def _prefilling(self):
        return [f for f in self.inflight.values() if f.first_token_at is None]

    def backlog_s(self, now: float | None = None, blind: bool = False) -> float:
        pend = self._prefilling()
        if not pend:
            return 0.0
        now = time.perf_counter() if now is None else now
        total = sum(f.full_prefill_s if blind else f.est_prefill_s for f in pend)
        return max(0.0, total - (now - min(f.sent_at for f in pend)))

    def n_inflight(self) -> int:
        return len(self.inflight)


_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$')


def parse_vllm_metrics(text: str) -> dict:
    """The few gauges/counters the router uses (summed over label sets)."""
    out: dict = {}
    want = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
            "vllm:kv_cache_usage_perc": "kv_usage",
            "vllm:prefix_cache_hits": "prefix_hits", "vllm:prefix_cache_hits_total": "prefix_hits",
            "vllm:prefix_cache_queries": "prefix_queries",
            "vllm:prefix_cache_queries_total": "prefix_queries",
            "vllm:num_preemptions": "preemptions", "vllm:num_preemptions_total": "preemptions"}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.groups()
        if name == "vllm:cache_config_info" and labels:
            lab = dict(re.findall(r'(\w+)="([^"]*)"', labels))
            for k in ("num_gpu_blocks", "block_size"):
                if lab.get(k, "").isdigit():
                    out[k] = int(lab[k])
            continue
        key = want.get(name)
        if key:
            try:
                out[key] = out.get(key, 0.0) + float(val)
            except ValueError:
                pass
    return out
