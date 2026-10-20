"""Block 0: every assumption of the simulation lives here, in one place.

Nothing in this file is measured. These are *modeled* numbers. If you ever
validate against a real server (e.g. vLLM with prefix caching), this is the only
file you need to recalibrate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """How long a prefill takes on one simulated GPU worker.

    prefill_time = fixed_overhead + (cost-weighted uncached tokens) / rate

    Only *uncached* tokens cost compute. A cache hit costs ~0 (the KV tensors
    are already in HBM) -- that asymmetry is the whole reason routing matters.
    """

    block_size: int = 64                  # tokens per KV block (vLLM default is 16)
    prefill_tokens_per_s: float = 25_000  # per-worker prefill throughput (ASSUMPTION)
    image_cost_multiplier: float = 1.5    # vision tower + projector make image tokens
                                          # costlier to recompute than text (ASSUMPTION)
    fixed_overhead_s: float = 0.015       # scheduling + first-token + network floor
    # Optional attention term: new tokens also attend to the cached prefix, so a hit is not
    # entirely free. seconds per (new token x cached token). 0 = hits cost nothing extra.
    attn_s_per_token_pair: float = 0.0

    def prefill_time(self, weighted_blocks: float, cached_blocks: float = 0.0) -> float:
        new_tokens = weighted_blocks * self.block_size
        t = self.fixed_overhead_s + new_tokens / self.prefill_tokens_per_s
        if self.attn_s_per_token_pair:
            t += self.attn_s_per_token_pair * new_tokens * cached_blocks * self.block_size
        return t

    @classmethod
    def from_calibration(cls, path, **overrides) -> "CostModel":
        """Load the `cost_model` section written by calibration/01_prefill_and_cache.py."""
        import json
        fitted = json.load(open(path))["cost_model"]
        return cls(**{**fitted, **overrides})


@dataclass(frozen=True)
class ClusterConfig:
    n_workers: int = 3
    cache_capacity_tokens: int = 150_000  # KV budget per worker (HBM reserved for prefix cache)
    cache_kind: str = "block_lru"         # "block_lru" (vLLM-style) | "sequence_lru" (llama.cpp-style)

    def cache_capacity_blocks(self, cost: CostModel) -> int:
        return self.cache_capacity_tokens // cost.block_size


@dataclass(frozen=True)
class WorkloadConfig:
    """Synthetic Pinterest-Assistant-style multi-turn conversations."""

    seed: int = 0
    horizon_s: float = 420.0              # conversations *start* during [0, horizon]
    warmup_s: float = 120.0               # requests arriving before this are not measured
    active_convs: int = 100               # target steady-state concurrent conversations

    think_mean_s: float = 12.0            # gap between a user's turns
    think_min_s: float = 1.0
    turns_min: int = 3
    turns_max: int = 8

    # context every conversation starts with
    system_tokens: int = 1024             # shared by ALL conversations
    images_min: int = 4                   # size of the pin collection a user attaches
    images_max: int = 16
    tokens_per_image: int = 256           # visual tokens per image (after projector)

    # per-turn text
    user_tokens: tuple[int, int] = (64, 192)
    response_tokens: tuple[int, int] = (128, 384)

    # cross-conversation sharing: some users open the same popular board
    n_popular_boards: int = 20
    popular_prob: float = 0.3             # P(conversation starts from a popular board)
    zipf_s: float = 1.1                   # skew over popular boards (higher = hotter head)

    @property
    def mean_turns(self) -> float:
        return (self.turns_min + self.turns_max) / 2

    # Experiments that vary conversation length need a fixed *request* rate instead of a
    # fixed concurrency (otherwise 1-turn conversations imply infinite starts). If set,
    # it overrides the Little's-law rate below.
    conv_start_rate_override: float | None = None

    @property
    def conv_start_rate(self) -> float:
        """Little's law: concurrency L = lambda * W  ->  lambda = L / W."""
        if self.conv_start_rate_override is not None:
            return self.conv_start_rate_override
        duration = max(1e-9, (self.mean_turns - 1) * self.think_mean_s)
        return self.active_convs / duration
