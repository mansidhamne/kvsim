"""Phase 1: multi-turn VLM conversations as real OpenAI-style chat requests with real images.

Same traffic shape as `kvsim.workload` (shared system prompt, a collection of images per
conversation, popular "boards" shared across conversations with Zipf popularity, 3-8 turns,
think time between turns, Poisson conversation starts sized by Little's law), but the output
is actual request bodies instead of block ids.

Design decisions
* Every policy must see byte-identical prompts, or the comparison is unfair. So the assistant
  turns in the history are CANNED text from the trace, not whatever the model replied, and each
  request asks for exactly that many output tokens. The model's real reply is only timed.
* An image is identified by (pool index, size). The same identity always produces the same
  bytes and the same UUID, so the engine's prefix cache can hit on it.
* "1 word = 1 token" for sizing text. Real tokenizers differ (~1.3 tokens/word for English);
  the mock worker uses the same convention, and Phase 2 measures the real ratio.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TraceConfig:
    seed: int = 0
    duration_s: float = 300.0             # conversations START during [0, duration]
    active_convs: int = 20                # target steady-state concurrent conversations
    think_mean_s: float = 12.0
    think_min_s: float = 1.0
    turns: tuple[int, int] = (3, 8)       # inclusive range

    system_words: int = 300               # shared by all conversations
    images_per_conv: tuple[int, int] = (2, 6)
    image_sizes: tuple[tuple[int, int], ...] = ((448, 448), (672, 448))  # (width, height)
    user_words: tuple[int, int] = (20, 60)
    reply_words: tuple[int, int] = (60, 160)

    n_boards: int = 20                    # popular boards shared across conversations
    board_prob: float = 0.3
    zipf_s: float = 1.1
    pool_size: int = 64                   # number of distinct source images

    @property
    def conv_start_rate(self) -> float:
        """Little's law: active = rate * mean conversation duration."""
        mean_turns = (self.turns[0] + self.turns[1]) / 2
        duration = max(1e-9, (mean_turns - 1) * self.think_mean_s)
        return self.active_convs / duration


# ---------------------------------------------------------------------------------- trace


@dataclass(frozen=True)
class ImageRef:
    pool_index: int
    width: int
    height: int

    @property
    def uuid(self) -> str:
        return f"img-{self.pool_index:05d}-{self.width}x{self.height}"


@dataclass
class Turn:
    at: float                 # scheduled send time (s from trace start)
    user_text: str
    reply_text: str           # canned assistant reply that goes into the NEXT turn's history
    max_tokens: int


@dataclass
class Conversation:
    conv_id: int
    board: int | None         # popular board id, or None for a private collection
    images: list[ImageRef]
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Trace:
    config: TraceConfig
    system_text: str
    conversations: list[Conversation]

    def n_requests(self) -> int:
        return sum(len(c.turns) for c in self.conversations)

    # -- (de)serialization: the trace file holds no image bytes, only image identities
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Trace":
        d = json.loads(s)
        cfg = d["config"]
        cfg["turns"] = tuple(cfg["turns"])
        for k in ("images_per_conv", "user_words", "reply_words"):
            cfg[k] = tuple(cfg[k])
        cfg["image_sizes"] = tuple(tuple(x) for x in cfg["image_sizes"])
        convs = [Conversation(c["conv_id"], c["board"], [ImageRef(**i) for i in c["images"]],
                              [Turn(**t) for t in c["turns"]]) for c in d["conversations"]]
        return cls(TraceConfig(**cfg), d["system_text"], convs)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        return cls.from_json(Path(path).read_text())


_WORDS = (
    "the a of and to in is it that for on with as was at by this from be are or an which "
    "image photo picture color light shadow left right top bottom center background foreground "
    "person people dog cat car tree sky water building street table chair window door road "
    "red blue green yellow white black bright dark small large old new wooden metal glass "
    "looks shows appears seems contains features includes describe compare explain notice "
    "style mood scene detail texture shape pattern edge corner object item group pair "
    "first second third last other same different similar both each every some many few "
    "because while where when how why what which there here near far above below between"
).split()


def _text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def generate_trace(cfg: TraceConfig) -> Trace:
    rng = np.random.default_rng(cfg.seed)
    txt = random.Random(cfg.seed)

    system_text = "You are a helpful visual assistant. " + _text(txt, cfg.system_words)

    def random_image() -> ImageRef:
        w, h = cfg.image_sizes[int(rng.integers(len(cfg.image_sizes)))]
        return ImageRef(int(rng.integers(cfg.pool_size)), w, h)

    lo, hi = cfg.images_per_conv
    boards = [[random_image() for _ in range(int(rng.integers(lo, hi + 1)))]
              for _ in range(cfg.n_boards)]
    ranks = np.arange(1, cfg.n_boards + 1, dtype=float)
    board_p = ranks ** (-cfg.zipf_s)
    board_p /= board_p.sum()

    n_convs = int(rng.poisson(cfg.conv_start_rate * cfg.duration_s))
    starts = np.sort(rng.uniform(0.0, cfg.duration_s, size=n_convs))

    convs: list[Conversation] = []
    for cid, t0 in enumerate(starts):
        if cfg.n_boards and rng.random() < cfg.board_prob:
            b = int(rng.choice(cfg.n_boards, p=board_p))
            board, images = b, list(boards[b])
        else:
            board, images = None, [random_image() for _ in range(int(rng.integers(lo, hi + 1)))]
        conv = Conversation(cid, board, images)
        t = float(t0)
        for k in range(int(rng.integers(cfg.turns[0], cfg.turns[1] + 1))):
            if k:
                t += cfg.think_min_s + rng.exponential(max(1e-9, cfg.think_mean_s - cfg.think_min_s))
            n_user = int(rng.integers(cfg.user_words[0], cfg.user_words[1] + 1))
            n_reply = int(rng.integers(cfg.reply_words[0], cfg.reply_words[1] + 1))
            conv.turns.append(Turn(round(t, 4), _text(txt, n_user), _text(txt, n_reply), n_reply))
        convs.append(conv)
    return Trace(cfg, system_text, convs)


# ---------------------------------------------------------------------------------- images


class ImagePool:
    """Source images, indexed 0..N-1, rendered to JPEG bytes at a requested size.

    * `directory`: real photos (e.g. COCO val images). File i = i-th file in sorted order
      (modulo the number of files), so a trace is portable across machines with the same set.
    * no directory: deterministic synthetic images (enough for the mock worker, which never
      looks at pixels).

    Same (index, size) -> same bytes, always (cached in memory).
    """

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, directory: str | Path | None = None):
        self.files: list[Path] = []
        if directory is not None:
            self.files = sorted(p for p in Path(directory).iterdir()
                                if p.suffix.lower() in self.EXTS)
            if not self.files:
                raise ValueError(f"no images found in {directory}")
        self._cache: dict[tuple[int, int, int], bytes] = {}

    @property
    def synthetic(self) -> bool:
        return not self.files

    def jpeg(self, ref: ImageRef) -> bytes:
        key = (ref.pool_index, ref.width, ref.height)
        b = self._cache.get(key)
        if b is None:
            b = self._render(ref)
            self._cache[key] = b
        return b

    def data_url(self, ref: ImageRef) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg(ref)).decode()

    def _render(self, ref: ImageRef) -> bytes:
        from PIL import Image, ImageDraw, ImageOps
        if self.files:
            with Image.open(self.files[ref.pool_index % len(self.files)]) as src:
                img = ImageOps.fit(src.convert("RGB"), (ref.width, ref.height))
        else:
            r = random.Random(ref.pool_index)
            img = Image.new("RGB", (ref.width, ref.height),
                            tuple(r.randrange(256) for _ in range(3)))
            d = ImageDraw.Draw(img)
            for _ in range(12):
                x0, y0 = r.randrange(ref.width), r.randrange(ref.height)
                x1, y1 = x0 + r.randrange(20, 200), y0 + r.randrange(20, 200)
                d.rectangle((x0, y0, x1, y1), fill=tuple(r.randrange(256) for _ in range(3)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


# ---------------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class RequestOptions:
    model: str = "mock"
    send_image_uuids: bool = True   # attach a stable per-image uuid (see docs/vllm_api.md)
    uuid_field: str = "uuid"        # field name on the image content part [verify in Phase 2]
    temperature: float = 0.0


def build_messages(trace: Trace, conv: Conversation, turn: int, pool: ImagePool,
                   opts: RequestOptions = RequestOptions()) -> list[dict]:
    """Chat history for `turn` (0-based): system, then the image collection + first question,
    then alternating canned replies and later questions."""
    image_parts = []
    for ref in conv.images:
        part = {"type": "image_url", "image_url": {"url": pool.data_url(ref)}}
        if opts.send_image_uuids:
            part[opts.uuid_field] = ref.uuid
        image_parts.append(part)
    msgs: list[dict] = [{"role": "system", "content": trace.system_text}]
    msgs.append({"role": "user",
                 "content": image_parts + [{"type": "text", "text": conv.turns[0].user_text}]})
    for k in range(1, turn + 1):
        msgs.append({"role": "assistant", "content": conv.turns[k - 1].reply_text})
        msgs.append({"role": "user", "content": conv.turns[k].user_text})
    return msgs


def build_request(trace: Trace, conv: Conversation, turn: int, pool: ImagePool,
                  opts: RequestOptions = RequestOptions()) -> tuple[dict, dict]:
    """-> (JSON body, HTTP headers). Headers carry the conversation id for routers that need a
    session key (session affinity); the engine ignores them."""
    t = conv.turns[turn]
    body = {
        "model": opts.model,
        "messages": build_messages(trace, conv, turn, pool, opts),
        "max_tokens": t.max_tokens,
        "min_tokens": t.max_tokens,        # vLLM extension: fixed output length per turn
        "temperature": opts.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"x-kvroute-conv": str(conv.conv_id), "x-kvroute-turn": str(turn)}
    return body, headers


def iter_requests(trace: Trace) -> Iterable[tuple[float, Conversation, int]]:
    """All (scheduled time, conversation, turn) in time order."""
    items = [(t.at, c, k) for c in trace.conversations for k, t in enumerate(c.turns)]
    items.sort(key=lambda x: (x[0], x[1].conv_id, x[2]))
    return items


def fingerprint(trace: Trace) -> str:
    """Short hash of the trace content, recorded with results to prove runs used the same trace."""
    return hashlib.sha256(trace.to_json().encode()).hexdigest()[:12]


def summarize_trace(trace: Trace) -> dict:
    convs = trace.conversations
    n_img = [len(c.images) for c in convs]
    return {
        "conversations": len(convs),
        "requests": trace.n_requests(),
        "board_share": round(sum(c.board is not None for c in convs) / max(1, len(convs)), 3),
        "images_per_conv_mean": round(float(np.mean(n_img)) if n_img else 0.0, 2),
        "last_send_s": round(max((t.at for c in convs for t in c.turns), default=0.0), 1),
        "fingerprint": fingerprint(trace),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate a multi-turn VLM trace (no image bytes).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--active-convs", type=int, default=20)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = generate_trace(TraceConfig(seed=a.seed, active_convs=a.active_convs, duration_s=a.duration))
    tr.save(a.out)
    print(json.dumps(summarize_trace(tr)))


if __name__ == "__main__":
    main()
