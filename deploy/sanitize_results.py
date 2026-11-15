"""Copy a Phase 3b results folder into the repo without any dataset-derived text.

VisionArena's terms forbid redistributing its content. Results never contain the prompts or
images, but they do contain the MODEL's replies to them (`output_text`, `sample_outputs`), which
describe those images. This copies a results folder and:
  * drops `output_text` from every request record and `sample_outputs` from every summary
  * gzips per-request logs
  * skips the downloaded dataset folder (`va_data/`)
Latencies, token counts and aggregate statistics are kept unchanged.

    python -m deploy.sanitize_results ~/Downloads/p3b/probe results/phase3b/probe
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

DROP_KEYS = {"output_text", "sample_outputs"}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def sanitize(src: Path, dst: Path) -> dict:
    counts = {"json": 0, "jsonl": 0, "copied": 0}
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_dir() or "va_data" in rel.parts or f.name == ".DS_Store":
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".jsonl":
            lines = [json.dumps(_clean(json.loads(l))) for l in f.read_text().splitlines() if l.strip()]
            with gzip.open(str(out) + ".gz", "wt") as g:
                g.write("\n".join(lines) + "\n")
            counts["jsonl"] += 1
        elif f.suffix == ".json":
            out.write_text(json.dumps(_clean(json.loads(f.read_text())), indent=2))
            counts["json"] += 1
        else:
            shutil.copy2(f, out)
            counts["copied"] += 1
    return counts


def main() -> None:
    src, dst = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
    print(sanitize(src, dst))
    leftover = [p for p in dst.rglob("*.json") if any(k in p.read_text() for k in DROP_KEYS)]
    assert not leftover, f"text fields remain in {leftover[:3]}"


if __name__ == "__main__":
    main()
