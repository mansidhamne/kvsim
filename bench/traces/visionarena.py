"""Replay real user-VLM conversations from LMArena's VisionArena-Chat.

Source: huggingface.co/datasets/lmarena-ai/VisionArena-Chat (199,036 conversations, Feb-Sep
2024, one image per conversation, 43 parquet shards of ~2 GB). Terms forbid redistribution:
this module reads the parquet files where they are (Kaggle disk, or a local sample) and never
writes conversation text, images or ids into the repo or into results. Trace files built from
it hold image hashes and text, so they stay out of git (results/**/traces/ is ignored).

What comes from the data: the image (bytes sent unchanged; uuid = hash of the bytes, so an image
that appears in several conversations is recognised as the same), every user turn, every
original model reply (used as fixed history; output length = the original reply length, capped).
What is modeled: when conversations start (rate set for the target load, optional burstiness),
and the think time between turns (ServeGen distribution, see servegen.py).

    rows = load_rows(["train-00000-of-00043.parquet"], max_rows=5000)
    trace, pool, stats = build_trace(rows, VAConfig(conv_rate=0.3, duration_s=360, seed=0))
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench.workload import Conversation, ImagePool, ImageRef, Trace, Turn
from router.hashing import image_tokens

from .servegen import ThinkTime

TOKENS_PER_WORD = 1.3      # rough, for sizing only (filters and caps); the engine reports real counts


@dataclass(frozen=True)
class VAConfig:
    seed: int = 0
    duration_s: float = 360.0          # conversations START during [0, duration]
    conv_rate: float = 0.3             # conversation starts per second
    burst_cv: float = 1.0              # inter-arrival CV: 1 = Poisson, >1 = burstier (gamma)
    think_scale: float = 0.1           # ServeGen think times x this (median 274 s -> 27 s)
    max_context_tokens: int = 7000     # drop conversations whose last turn would exceed this
    reply_cap_tokens: int = 384        # cap on output length per turn (T4 decode time)
    min_reply_tokens: int = 8
    system_text: str = ""              # VisionArena chats have no system prompt
    # A follow-up whose nominal send time falls more than this many seconds after the trace's
    # last conversation start is dropped (the user "left"): ServeGen's think-time tail reaches
    # hours, which would stretch a run far beyond the measured window. Counted in stats.
    tail_grace_s: float = 60.0


def load_rows(paths, max_rows: int | None = None, columns: list[str] | None = None) -> list[dict]:
    """Rows from local parquet files (dicts with images, conversation, num_turns, ...)."""
    import pyarrow.parquet as pq
    rows: list[dict] = []
    for p in paths:
        pf = pq.ParquetFile(str(p))
        for i in range(pf.num_row_groups):
            rows.extend(pf.read_row_group(i, columns=columns).to_pylist())
            if max_rows and len(rows) >= max_rows:
                return rows[:max_rows]
    return rows


def turns_of(conv_field) -> list[tuple[str, str]]:
    """[(user_text, assistant_text), ...] from VisionArena's conversation field (a list of
    turns, each a list holding one {role, content} message)."""
    msgs = []
    for t in conv_field or []:
        msgs.extend(t if isinstance(t, list) else [t])
    pairs, user = [], None
    for m in msgs:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "user":
            user = content
        elif role == "assistant" and user is not None:
            pairs.append((user, content))
            user = None
    return pairs


def _image_size(data: bytes) -> tuple[int, int]:
    import io
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def arrivals(rng: np.random.Generator, rate: float, duration: float, cv: float = 1.0) -> list[float]:
    """Start times with mean rate `rate` over [0, duration]; gamma inter-arrivals with CV `cv`."""
    out, t = [], 0.0
    k = 1.0 / (cv * cv)                      # gamma shape: CV = 1/sqrt(k)
    while True:
        t += rng.gamma(k, 1.0 / (rate * k))
        if t > duration:
            return out
        out.append(t)


def build_trace(rows: list[dict], cfg: VAConfig, pool: ImagePool | None = None):
    """-> (Trace, ImagePool with the trace's images registered, stats dict)."""
    rng = np.random.default_rng(cfg.seed)
    pool = pool or ImagePool()
    think = ThinkTime(cfg.think_scale)
    usable, dropped = [], {"no_image": 0, "no_turns": 0, "too_long": 0}
    for r in rows:
        imgs = r.get("images") or []
        pairs = turns_of(r.get("conversation"))
        if not imgs:
            dropped["no_image"] += 1
            continue
        if not pairs:
            dropped["no_turns"] += 1
            continue
        data = imgs[0]["bytes"] if isinstance(imgs[0], dict) else imgs[0]
        w, h = _image_size(data)
        total = image_tokens((w, h)) + sum((len(u.split()) + len(a.split())) * TOKENS_PER_WORD
                                           for u, a in pairs)
        if total > cfg.max_context_tokens:
            dropped["too_long"] += 1
            continue
        usable.append((data, (w, h), pairs))
    if not usable:
        raise ValueError(f"no usable conversations ({dropped})")

    starts = arrivals(rng, cfg.conv_rate, cfg.duration_s, cfg.burst_cv)
    if len(starts) > len(usable):
        raise ValueError(f"need {len(starts)} conversations, only {len(usable)} usable: load more rows")
    order = rng.permutation(len(usable))
    convs = []
    late_dropped = [0]
    for cid, (t0, j) in enumerate(zip(starts, order)):
        data, (w, h), pairs = usable[j]
        key = pool.add_blob(data)
        conv = Conversation(cid, None, [ImageRef(-1, w, h, blob=key)])
        t = t0
        for k, (u, a) in enumerate(pairs):
            n_out = max(cfg.min_reply_tokens,
                        min(cfg.reply_cap_tokens, math.ceil(len(a.split()) * TOKENS_PER_WORD)))
            th = None
            if k:
                th = float(think.sample(rng))
                t += th
                if t > cfg.duration_s + cfg.tail_grace_s:
                    late_dropped[0] += len(pairs) - k
                    break
            conv.turns.append(Turn(round(t, 4), u, a, n_out, None if th is None else round(th, 4)))
        convs.append(conv)

    n_turns = [len(c.turns) for c in convs]
    blobs = [c.images[0].blob for c in convs]
    counts: dict[str, int] = {}
    for b in blobs:
        counts[b] = counts.get(b, 0) + 1
    stats = {
        "rows_read": len(rows), "usable": len(usable), "dropped": dropped,
        "conversations": len(convs), "requests": sum(n_turns),
        "follow_ups_after_window_dropped": late_dropped[0],
        "single_turn_share": float(np.mean([n == 1 for n in n_turns])) if convs else None,
        "turns_mean": float(np.mean(n_turns)) if convs else None,
        "distinct_images": len(counts),
        "images_in_2plus_convs": sum(1 for v in counts.values() if v > 1),
    }
    trace = Trace(asdict(cfg), cfg.system_text, convs, source="visionarena",
                  meta={"stats": stats, "think": f"ServeGen quantiles x {cfg.think_scale:g}"})
    return trace, pool, stats
