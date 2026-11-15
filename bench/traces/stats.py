"""Workload statistics that bound what any routing policy can achieve.

* reuse ceiling: share of prompt tokens that ONE unbounded cache could serve, replaying requests
  in nominal send order. No policy can beat it; a policy at ~99% of it has no headroom left.
* live working set: peak tokens that must stay cached so that every follow-up and every shared
  prefix still hits. Cache caps for the scarce-cache sweep are set as fractions of this.

Send times are nominal (turn 0 at its start, later turns after an estimated reply time plus the
think time), so these are estimates; the engine reports the real numbers per run.
"""
from __future__ import annotations

from bench.workload import ImagePool, RequestOptions, Trace, build_messages
from router.hashing import prompt_info


def nominal_sends(trace: Trace, decode_s_per_token: float = 0.03) -> list[tuple[float, int, int]]:
    """[(send time, conv_id, turn)] in nominal order."""
    out = []
    for c in trace.conversations:
        t = c.turns[0].at
        for k, tr in enumerate(c.turns):
            if k:
                prev = c.turns[k - 1]
                t = (t + prev.max_tokens * decode_s_per_token + tr.think_s) if tr.think_s is not None else tr.at
            out.append((t, c.conv_id, k))
    return sorted(out)


def workload_stats(trace: Trace, pool: ImagePool, block_size: int = 16,
                   hold_s: float | None = None) -> dict:
    """Reuse ceiling and peak live working set (in blocks). A conversation's blocks stay live
    from its first send until its last send + `hold_s` (default: its own mean think time)."""
    conv = {c.conv_id: c for c in trace.conversations}
    opts = RequestOptions(model="x")
    sends = nominal_sends(trace)
    seen: set[int] = set()
    reuse = total = 0
    blocks_of: dict[int, set[int]] = {}
    span: dict[int, list[float]] = {}
    for t, cid, k in sends:
        info = prompt_info(build_messages(trace, conv[cid], k, pool, opts), block_size)
        hit = 0
        for b in info.blocks:
            if b in seen:
                hit += 1
            else:
                break
        seen.update(info.blocks)
        reuse += hit * block_size
        total += info.n_tokens
        blocks_of.setdefault(cid, set()).update(info.blocks)
        s = span.setdefault(cid, [t, t])
        s[1] = t
    # peak live unique blocks over time (shared prefixes counted once)
    events = []
    for cid, (a, b) in span.items():
        thinks = [tr.think_s for tr in conv[cid].turns[1:] if tr.think_s is not None]
        hold = hold_s if hold_s is not None else (sum(thinks) / len(thinks) if thinks else 0.0)
        events.append((a, 1, cid))
        events.append((b + hold, -1, cid))
    events.sort()
    live: dict[int, int] = {}
    peak = 0
    for _, d, cid in events:
        for blk in blocks_of[cid]:
            live[blk] = live.get(blk, 0) + d
            if live[blk] <= 0:
                live.pop(blk)
        peak = max(peak, len(live))
    return {"requests": len(sends), "prompt_tokens_est": total,
            "reuse_ceiling": reuse / total if total else None,
            "peak_live_blocks": peak, "peak_live_tokens": peak * block_size}
