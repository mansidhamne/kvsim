"""Visualize the synthetic workload (Block 2).

Everything here is computed from the same generator the simulator uses, not drawn by hand.

  (a, b) anatomy of two conversations, turn by turn. Faded = tokens an ideal (infinite, global)
         cache would already hold at that moment; solid = tokens that must be prefilled.
  (c)    how often each of the 20 popular boards is opened (Zipf) vs private collections.
  (d)    average prompt per turn index: how much of it was already computed somewhere.

"Ideal" = longest prefix that any earlier request (this conversation or another) has already
computed, i.e. the hit rate of an unbounded cache on a single machine. It is an upper bound on
what any router + finite cache can achieve; the gap to it is what routing and eviction lose.

Run:  python experiments/00_workload.py
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from _common import RESULTS
from kvsim import CostModel, WorkloadConfig
from kvsim.workload import generate_requests

KIND_COLOR = {"sys": "#9aa0a6", "img": "#e60023", "u": "#4285f4", "r": "#34a853"}
KIND_NAME = {"sys": "system prompt (shared by all)", "img": "images (256 tokens each, 1.5x cost)",
             "u": "user message", "r": "assistant reply"}

cost = CostModel()
wl = replace(WorkloadConfig(), seed=0, horizon_s=1500.0)
BS = cost.block_size
reqs, trie = generate_requests(wl, cost, return_trie=True)

# ---------------------------------------------------------------- per-request analysis
def kind_of(node: int) -> str:
    return trie.key(node)[0]


first_conv: dict[int, int] = {}      # node -> first conversation that computed it
multi: set[int] = set()              # nodes computed by 2+ conversations
info = {}                            # rid -> (ideal_hit_blocks, cross_hit_blocks)
for r in reqs:                       # reqs are sorted by arrival
    ideal = cross = 0
    still_ideal = still_cross = True
    for n in r.blocks:
        seen = n in first_conv
        if still_ideal and seen:
            ideal += 1
        else:
            still_ideal = False
        if still_cross and seen and (first_conv[n] != r.conv_id or n in multi):
            cross += 1
        else:
            still_cross = False
        if not still_ideal and not still_cross:
            break
    info[r.rid] = (ideal, cross)
    for n in r.blocks:
        if n not in first_conv:
            first_conv[n] = r.conv_id
        elif first_conv[n] != r.conv_id:
            multi.add(n)

by_conv: dict[int, list] = defaultdict(list)
for r in reqs:
    by_conv[r.conv_id].append(r)


def conv_summary(cid: int):
    rs = by_conv[cid]
    kinds = [kind_of(n) for n in rs[0].blocks]
    first_img = next(i for i, k in enumerate(kinds) if k == "img")
    key = trie.key(rs[0].blocks[first_img])           # ("img", "pop"|"own", id, i, j)
    n_img = sum(1 for k in kinds if k == "img") * BS // wl.tokens_per_image
    return key[1], key[2], n_img


def pick(popular: bool):
    for cid, rs in by_conv.items():
        if len(rs) != 4 or rs[0].arrival < wl.warmup_s:
            continue
        typ, board, n_img = conv_summary(cid)
        if (typ == "pop") != popular or not (5 <= n_img <= 8):
            continue
        if popular and info[rs[0].rid][0] * BS <= wl.system_tokens:
            continue                                    # board not yet seen by anyone
        return cid
    raise RuntimeError("no suitable conversation found")


def segments(r):
    """Run-length encode a prompt into (label_key, kind, n_blocks), in order."""
    segs, turn_u, turn_r = [], 0, 0
    prev = None
    for n in r.blocks:
        key = trie.key(n)
        k = key[0]
        tag = (k, key[2]) if k in ("u", "r") else (k, None)   # ("u", turn) / ("r", turn)
        if tag == prev:
            segs[-1][2] += 1
        else:
            segs.append([tag, k, 1])
            prev = tag
    return segs


def draw_conv(ax, cid, title):
    rs = by_conv[cid]
    typ, board, n_img = conv_summary(cid)
    max_tok = max(r.n_blocks for r in rs) * BS
    for row, r in enumerate(rs):
        y = len(rs) - 1 - row
        ideal, cross = info[r.rid]
        x = 0
        for (tag, k, nb) in segments(r):
            w = nb * BS
            s, e = x, x + w
            cut = ideal * BS
            if s < cut:                                   # faded part (already computed)
                ax.barh(y, min(e, cut) - s, left=s, color=KIND_COLOR[k], alpha=0.30,
                        height=0.62, edgecolor="white", lw=0.6)
            if e > cut:                                   # solid part (must prefill)
                ax.barh(y, e - max(s, cut), left=max(s, cut), color=KIND_COLOR[k],
                        height=0.62, edgecolor="white", lw=0.6)
            if w >= 110:
                if k == "sys":
                    label = "system"
                elif k == "img":
                    label = f"{n_img} images"
                else:
                    label = f"{k}{tag[1] + 1}"
                ax.text(s + w / 2, y, label, ha="center", va="center", fontsize=7,
                        color="white" if e > cut else "#333333")
            x = e
        ax.plot([ideal * BS] * 2, [y - 0.38, y + 0.38], color="black", lw=1.2, ls="--")
        total = r.n_blocks * BS
        ax.text(max_tok * 1.02, y,
                f"turn {row + 1}: {total:,} tok, {ideal * BS / total:.0%} reusable",
                va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, max_tok * 1.36)
    ax.set_ylim(-0.6, len(rs) - 0.4)
    ax.set_xlabel("prompt position (tokens)")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.grid(False)
    ax.spines["left"].set_visible(False)


fig = plt.figure(figsize=(13.5, 11))
gs = GridSpec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.7, wspace=0.28)

ax_a = fig.add_subplot(gs[0, :])
draw_conv(ax_a, pick(False),
          "(a) A conversation with its own private images: only the system prompt is shared "
          "across users, so turn 1 reuses little")
ax_b = fig.add_subplot(gs[1, :], sharex=ax_a)
draw_conv(ax_b, pick(True),
          "(b) A conversation that opened a popular board someone else already used: "
          "the images are reusable at turn 1 too")

handles = [mpatches.Patch(color=c, label=KIND_NAME[k]) for k, c in KIND_COLOR.items()]
handles += [mpatches.Patch(facecolor="#777777", alpha=0.30, label="faded: already computed earlier"),
            mpatches.Patch(facecolor="#777777", label="solid: must be prefilled now"),
            plt.Line2D([], [], color="black", ls="--", lw=1.2, label="reusable-prefix boundary")]
ax_a.legend(handles=handles, ncol=3, fontsize=7.5, loc="upper center",
            bbox_to_anchor=(0.5, 1.42), frameon=False)

# ---------------------------------------------------------------- (c) popular boards
convs = [rs[0] for rs in by_conv.values()]
n_convs = len(convs)
board_counts = np.zeros(wl.n_popular_boards)
for rs in by_conv.values():
    typ, board, _ = conv_summary(rs[0].conv_id)
    if typ == "pop":
        board_counts[board] += 1
n_pop = int(board_counts.sum())
ranks = np.arange(1, wl.n_popular_boards + 1, dtype=float)
p = ranks ** (-wl.zipf_s)
p /= p.sum()

ax_c = fig.add_subplot(gs[2, 0])
ax_c.bar(np.arange(1, wl.n_popular_boards + 1), 100 * board_counts / n_convs,
         color="#e60023", alpha=0.85, label="generated (this trace)")
ax_c.plot(ranks, 100 * wl.popular_prob * p, "ko", ms=4, label="theory: 30% x Zipf(1.1)")
ax_c.set_xticks([1, 5, 10, 15, 20])
ax_c.set_xlabel("popular board (rank by popularity)")
ax_c.set_ylabel("% of all conversations")
ax_c.set_title(f"(c) 30% open a shared popular board (Zipf),\n70% bring private images "
               f"(top board = {100 * board_counts[0] / n_convs:.1f}% of {n_convs:,} convs)",
               loc="left", fontsize=10, fontweight="bold")
ax_c.legend(frameon=False, fontsize=8)

# ---------------------------------------------------------------- (d) reuse by turn
max_turn = wl.turns_max
sums = np.zeros((max_turn, 3))        # cross-conversation, own history, new
counts = np.zeros(max_turn)
for r in reqs:
    if r.arrival < wl.warmup_s:
        continue
    ideal, cross = info[r.rid]
    sums[r.turn] += [cross * BS, (ideal - cross) * BS, (r.n_blocks - ideal) * BS]
    counts[r.turn] += 1
means = sums / counts[:, None]
ax_d = fig.add_subplot(gs[2, 1])
xs = np.arange(1, max_turn + 1)
bottom = np.zeros(max_turn)
for j, (lab, col, al) in enumerate([("reusable: from OTHER conversations", "#4285f4", 0.9),
                                    ("reusable: this conversation's earlier turns", "#34a853", 0.9),
                                    ("new: must be prefilled", "#e60023", 0.9)]):
    ax_d.bar(xs, means[:, j], bottom=bottom, color=col, alpha=al, label=lab, width=0.7)
    bottom += means[:, j]
for t in range(max_turn):
    reuse = (means[t, 0] + means[t, 1]) / bottom[t]
    ax_d.text(xs[t], bottom[t] + 60, f"{reuse:.0%}", ha="center", fontsize=8)
ax_d.set_xticks(xs)
ax_d.set_xticklabels([f"{t}\n(n={int(c)})" for t, c in zip(xs, counts)], fontsize=8)
ax_d.set_xlabel("turn number  (n = requests, measured after warm-up)")
ax_d.set_ylabel("mean prompt length (tokens)")
ax_d.set_ylim(0, bottom.max() * 1.15)
ax_d.set_title("(d) Later turns are ~90% reusable (if routed well)\n"
               "label = share an unbounded single cache already holds",
               loc="left", fontsize=10, fontweight="bold")
ax_d.legend(frameon=False, fontsize=7.5, loc="upper left")

fig.suptitle("Synthetic workload: system prompt + images + multi-turn history "
             "(token counts exact; traffic is generated, not real)", fontsize=12, y=0.995)
out = RESULTS / "workload.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print(f"{len(reqs)} requests, {n_convs} conversations, {n_pop} popular ({n_pop / n_convs:.0%})")
for t in range(max_turn):
    print(f"turn {t + 1}: n={int(counts[t])}, mean prompt {bottom[t]:.0f} tok, "
          f"reusable {(means[t, 0] + means[t, 1]) / bottom[t]:.0%}")
