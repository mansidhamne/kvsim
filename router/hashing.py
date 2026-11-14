"""Chat messages -> prompt "tokens" -> chained KV block hashes.

Phase 3 version (shadow index): the router does NOT run the model's tokenizer. It splits text on
whitespace (1 word ~ 1 token; measured ~1.05 tokens/word for the benchmark's text), adds a role
marker per message, and expands each image into (w/28)*(h/28) tokens keyed by its uuid, like
Qwen2-VL. Block identity is a chained hash over full blocks, exactly like vLLM's structure, so
"same prefix" means the same thing to the router and the engine even though the hash VALUES
differ. Phase 4 replaces this with the engine's own token ids and hash function.

The mock worker uses the same functions, so on the mock the router's view is exact.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from typing import Sequence

IMAGE_PATCH = 28          # Qwen2-VL: one visual token per 28x28 pixels


def image_size_from_part(part: dict) -> tuple[int, int] | None:
    """Image size for token counting, read from the image header in a data URL."""
    url = (part.get("image_url") or {}).get("url", "")
    if url.startswith("data:"):
        from PIL import Image
        raw = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as im:
            return im.size
    return None


def image_id(part: dict) -> str:
    url = (part.get("image_url") or {}).get("url", "")
    return part.get("uuid") or hashlib.sha1(url.encode()).hexdigest()[:16]


def image_tokens(size: tuple[int, int], patch: int = IMAGE_PATCH) -> int:
    return max(1, math.ceil(size[0] / patch) * math.ceil(size[1] / patch))


def tokenize(messages: list[dict], cfg=None, _img_cache: dict | None = None) -> list:
    """Messages -> list of hashable token keys: ("role", r), ("w", word), ("img", id, i).
    `cfg` may carry `image_patch`; `_img_cache` maps image id -> token count (avoids re-reading
    the image header for every request)."""
    patch = getattr(cfg, "image_patch", IMAGE_PATCH)
    toks: list = []
    for m in messages:
        toks.append(("role", m.get("role", "")))
        content = m.get("content", "")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for p in parts:
            if p.get("type") == "text":
                toks.extend(("w", w) for w in p.get("text", "").split())
            elif p.get("type") == "image_url":
                ident = image_id(p)
                n = None if _img_cache is None else _img_cache.get(ident)
                if n is None:
                    n = image_tokens(image_size_from_part(p) or (448, 448), patch)
                    if _img_cache is not None:
                        _img_cache[ident] = n
                toks.extend(("img", ident, i) for i in range(n))
    toks.append(("role", "assistant-start"))
    return toks


def block_hashes(tokens: Sequence, block_size: int) -> list[int]:
    """Chained hashes over FULL blocks only, like vLLM: h_i = H(h_{i-1}, tokens of block i)."""
    out, parent = [], b"none"
    for i in range(0, len(tokens) - block_size + 1, block_size):
        d = hashlib.sha256(parent + repr(tokens[i:i + block_size]).encode()).digest()
        out.append(int.from_bytes(d[:8], "big"))
        parent = d
    return out


@dataclass(frozen=True)
class PromptInfo:
    """What a router needs about one request's prompt."""
    blocks: tuple[int, ...]             # chained hash per full block
    cum_weight: tuple[float, ...]       # cum_weight[i] = cost-weighted blocks of blocks[:i]
    tail_weight: float                  # cost-weighted partial last block (never cached)
    n_tokens: int
    n_image_tokens: int
    block_size: int = 16

    def uncached_tokens(self, hit_blocks: int) -> int:
        """Raw tokens still to prefill (for the attention term of the cost model)."""
        return self.n_tokens - hit_blocks * self.block_size

    def uncached_weight(self, hit_blocks: int) -> float:
        """Cost-weighted blocks still to prefill if the first `hit_blocks` are cached
        (same contract as kvsim.workload.Request, so kvsim's CostModel applies)."""
        return self.cum_weight[-1] - self.cum_weight[hit_blocks] + self.tail_weight


def prompt_info(messages: list[dict], block_size: int = 16, image_cost_multiplier: float = 1.0,
                img_cache: dict | None = None, patch: int = IMAGE_PATCH) -> PromptInfo:
    class _C:
        image_patch = patch
    toks = tokenize(messages, _C, img_cache)
    blocks = block_hashes(toks, block_size)
    cum = [0.0]
    n_img = 0
    for i in range(len(blocks)):
        chunk = toks[i * block_size:(i + 1) * block_size]
        k = sum(1 for t in chunk if t[0] == "img")
        n_img += k
        cum.append(cum[-1] + (len(chunk) - k + k * image_cost_multiplier) / block_size)
    tail = toks[len(blocks) * block_size:]
    k = sum(1 for t in tail if t[0] == "img")
    n_img += k
    tail_w = (len(tail) - k + k * image_cost_multiplier) / block_size
    return PromptInfo(tuple(blocks), tuple(cum), tail_w, len(toks), n_img, block_size)
