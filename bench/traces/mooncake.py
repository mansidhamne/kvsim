"""Replay Kimi's production prefix-sharing pattern from the Mooncake FAST'25 trace (Apache-2.0).

Source: github.com/kvcache-ai/Mooncake, FAST25-release/traces/conversation_trace.jsonl: 12,031
requests from one hour of production, with ms timestamps, input/output token counts and
`hash_ids` (chained block hashes, ~500 tokens per id: shared leading ids = shared prefix).
No text, no images, no session ids.

Each hash id becomes a fixed block of deterministic text, so requests that share hash ids share
a prefix exactly, as in production. Scaled to fit 2x T4 (documented in the trace's meta):
  * `tokens_per_hash`: words per hash id (production ~500 tokens; default 64), so a median
    request becomes ~900 words instead of ~6,900 tokens
  * `max_hashes`: longer requests are cut to their first N ids (prefix kept)
  * `time_dilation`: timestamps are multiplied by this (>1 = slower arrivals)
  * outputs scaled by `output_scale` and capped
Every request is its own one-turn "conversation" (the trace has no session ids), so session
affinity has nothing to pin: it behaves like least-loaded. That is part of the result.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.workload import _WORDS, Conversation, Trace, Turn


@dataclass(frozen=True)
class MooncakeConfig:
    # Sharing in this trace builds up with request volume (measured on the full trace: a 6 min
    # window has a 0.21 reuse ceiling, 20 min 0.31, the whole hour 0.37; random thinning keeps
    # only 0.12-0.27). So the default replays a 6-minute steady-state window at real speed
    # (~3 req/s) and shrinks the prompts instead of the request rate.
    tokens_per_hash: int = 32
    max_hashes: int = 120
    time_dilation: float = 1.0
    start_s: float = 600.0             # window of the original trace to replay (seconds)
    duration_s: float = 360.0          # after dilation
    output_scale: float = 0.5
    # 64: at real speed (3.2 req/s) this keeps decode load (~174 output tok/s) near VisionArena's
    # high load (~240 tok/s on 2x T4); 128 would be ~320 tok/s and overload the cache-blind baselines
    output_cap: int = 64
    min_output: int = 8


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _block_text(h: int, n: int) -> str:
    r = random.Random(h * 7919 + 17)
    return " ".join(r.choice(_WORDS) for _ in range(n))


def build_trace(rows: list[dict], cfg: MooncakeConfig) -> tuple[Trace, dict]:
    rows = sorted(rows, key=lambda r: r["timestamp"])
    t_first = rows[0]["timestamp"] / 1000.0
    convs, cut = [], 0
    cache: dict[int, str] = {}
    for r in rows:
        t = (r["timestamp"] / 1000.0 - t_first - cfg.start_s) * cfg.time_dilation
        if t < 0:
            continue
        if t > cfg.duration_s:
            break
        ids = r["hash_ids"]
        if len(ids) > cfg.max_hashes:
            ids, cut = ids[:cfg.max_hashes], cut + 1
        text = " ".join(cache.setdefault(h, _block_text(h, cfg.tokens_per_hash)) for h in ids)
        n_out = max(cfg.min_output, min(cfg.output_cap, math.ceil(r["output_length"] * cfg.output_scale)))
        convs.append(Conversation(len(convs), None, [], [Turn(round(t, 4), text, "", n_out)]))
    stats = {"requests": len(convs), "cut_to_max_hashes": cut,
             "rate_per_s": len(convs) / cfg.duration_s if cfg.duration_s else None}
    trace = Trace(asdict(cfg), "", convs, source="mooncake", meta={"stats": stats})
    return trace, stats
