"""Block 2: synthetic multi-turn "Pinterest Assistant" traffic.

A conversation's prompt at turn k is:

    [system prompt][image_1 ... image_n][user_1][resp_1] ... [user_k]
     ^ shared by all  ^ the user's pin collection   ^ grows every turn

Turn k+1's prompt starts with turn k's entire prompt (+ the model's answer), so
the longest reusable prefix grows with every turn. That is exactly the situation
where *where* you send the request decides whether you pay for the prefix again.

Design decisions
* Open-loop arrivals. All arrival times are drawn up-front from a seeded RNG, so
  every routing policy sees the *identical* request sequence. (Closed-loop would
  make arrivals depend on the policy's latency, which breaks fair comparison.)
* "Concurrency" = number of simultaneously active conversations. Conversation
  starts are Poisson with rate chosen via Little's law so the steady-state number
  of active conversations is `active_convs`. Raising it raises both request rate
  AND cache pressure (more distinct contexts), like real traffic growth.
* Prefix identity comes from a trie: id(parent, content) -> int. Equal id means
  equal token sequence from position 0. Deterministic and collision-free (a
  real system uses chained hashes of token ids instead).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import CostModel, WorkloadConfig


@dataclass(frozen=True)
class Request:
    rid: int
    conv_id: int
    turn: int
    arrival: float
    blocks: tuple[int, ...]       # trie node id per KV block, in prompt order
    cum_weight: tuple[float, ...]  # cum_weight[i] = recompute cost (in text-blocks) of blocks[:i]

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def uncached_weight(self, hit_blocks: int) -> float:
        """Cost-weighted blocks left to prefill if the first `hit_blocks` are cached."""
        return self.cum_weight[-1] - self.cum_weight[hit_blocks]


class BlockTrie:
    """Interns (parent_id, content_key) -> int, i.e. a radix tree of prompt prefixes."""

    ROOT = 0

    def __init__(self, record_keys: bool = False) -> None:
        self._ids: dict[tuple, int] = {}
        # optional reverse map node id -> content key, only used for visualization
        self._keys: dict[int, tuple] | None = {} if record_keys else None

    def child(self, parent: int, key: tuple) -> int:
        k = (parent, key)
        node = self._ids.get(k)
        if node is None:
            node = len(self._ids) + 1
            self._ids[k] = node
            if self._keys is not None:
                self._keys[node] = key
        return node

    def key(self, node: int) -> tuple:
        """Content key of a node (requires record_keys=True)."""
        assert self._keys is not None, "BlockTrie was created without record_keys"
        return self._keys[node]

    def __len__(self) -> int:
        return len(self._ids)


def _blocks_for(tokens: int, block_size: int) -> int:
    return max(1, math.ceil(tokens / block_size))


class _Prompt:
    """Mutable prompt under construction: parallel lists of node ids and weights."""

    def __init__(self, trie: BlockTrie):
        self.trie = trie
        self.nodes: list[int] = []
        self.weights: list[float] = []

    def extend(self, key_prefix: tuple, n_blocks: int, weight: float) -> None:
        parent = self.nodes[-1] if self.nodes else BlockTrie.ROOT
        for j in range(n_blocks):
            parent = self.trie.child(parent, key_prefix + (j,))
            self.nodes.append(parent)
            self.weights.append(weight)

    def snapshot(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        cum = [0.0]
        for w in self.weights:
            cum.append(cum[-1] + w)
        return tuple(self.nodes), tuple(cum)


def generate_requests(wl: WorkloadConfig, cost: CostModel, *, return_trie: bool = False):
    """Generate the request trace. With return_trie=True returns (requests, trie)
    where the trie can map block ids back to their content (for plotting)."""
    rng = np.random.default_rng(wl.seed)
    bs = cost.block_size
    trie = BlockTrie(record_keys=return_trie)

    sys_blocks = _blocks_for(wl.system_tokens, bs)
    img_blocks = _blocks_for(wl.tokens_per_image, bs)

    # Popular boards: fixed *ordered* image lists shared across users (Zipf popularity).
    # Order matters: the same images in a different order share no prefix.
    board_sizes = rng.integers(wl.images_min, wl.images_max + 1, size=wl.n_popular_boards)
    ranks = np.arange(1, wl.n_popular_boards + 1, dtype=float)
    board_p = ranks ** (-wl.zipf_s)
    board_p /= board_p.sum()

    n_convs = int(rng.poisson(wl.conv_start_rate * wl.horizon_s))
    starts = np.sort(rng.uniform(0.0, wl.horizon_s, size=n_convs))

    raw: list[tuple[float, int, int, tuple[int, ...], tuple[float, ...]]] = []
    for conv_id, t0 in enumerate(starts):
        prompt = _Prompt(trie)
        prompt.extend(("sys",), sys_blocks, 1.0)

        if rng.random() < wl.popular_prob:
            board = int(rng.choice(wl.n_popular_boards, p=board_p))
            image_keys = [("pop", board, i) for i in range(int(board_sizes[board]))]
        else:
            n_img = int(rng.integers(wl.images_min, wl.images_max + 1))
            image_keys = [("own", conv_id, i) for i in range(n_img)]
        for ik in image_keys:
            prompt.extend(("img",) + ik, img_blocks, cost.image_cost_multiplier)

        n_turns = int(rng.integers(wl.turns_min, wl.turns_max + 1))
        t = float(t0)
        for turn in range(n_turns):
            if turn > 0:
                t += wl.think_min_s + rng.exponential(max(1e-9, wl.think_mean_s - wl.think_min_s))
            user_blocks = _blocks_for(int(rng.integers(*wl.user_tokens)), bs)
            # snapshot = prompt-so-far + this turn's user message; this is what gets prefilled
            work = _Prompt(trie)
            work.nodes, work.weights = list(prompt.nodes), list(prompt.weights)
            work.extend(("u", conv_id, turn), user_blocks, 1.0)
            blocks, cum = work.snapshot()
            raw.append((t, conv_id, turn, blocks, cum))

            # the assistant's reply becomes history for the next turn
            resp_blocks = _blocks_for(int(rng.integers(*wl.response_tokens)), bs)
            work.extend(("r", conv_id, turn), resp_blocks, 1.0)
            prompt.nodes, prompt.weights = work.nodes, work.weights

    raw.sort(key=lambda r: (r[0], r[1]))
    requests = [Request(i, c, k, t, b, cw) for i, (t, c, k, b, cw) in enumerate(raw)]
    return (requests, trie) if return_trie else requests
