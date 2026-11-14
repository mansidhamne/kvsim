# Benchmark tooling (`bench/`)

Everything here runs on a laptop. GPUs are only needed when the targets are real vLLM workers.

| module | what it does |
|---|---|
| `bench/workload.py` | Generates a multi-turn VLM trace (conversations, images, turn times) and turns each turn into an OpenAI chat request. Same seed = same trace; the fingerprint proves it. |
| `bench/mock_worker.py` | A fake vLLM server: same API, same metric names, an LRU block prefix cache, FIFO prefill, fixed-rate decode. |
| `bench/loadgen.py` | Replays a trace against one URL (the router) or several workers, streaming, and writes one JSON line per request. |

## Try it

```bash
pip install -e ".[dev,router]"
python -m bench.mock_worker --n 3 --base-port 8001 --time-scale 10      # leave running
python -m bench.loadgen --targets http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003 \
    --pick session --active-convs 10 --duration 120 --time-scale 10 --out results/phase1/mock_session.jsonl
```

`--pick` is a stand-in for the router until Phase 3: `round_robin`, `random`, or `session`
(pin each conversation to one worker). `--images DIR` uses real photos instead of synthetic ones.

## Choices that affect results

* **Canned history.** The assistant messages in the history come from the trace, not from the
  model, so every policy sends byte-identical prompts. Output length is fixed per turn
  (`max_tokens = min_tokens`).
* **Open-loop across conversations, closed within one.** Conversations start on schedule no
  matter how slow the system is. A follow-up turn waits for the previous answer; the extra wait is
  recorded as `late_s`.
* **Stable image identity.** An image is (pool index, size): always the same JPEG bytes and the
  same `uuid` (`img-00012-448x448`), which is sent on the image content part so vLLM can use it as
  the cache key (field name to verify in Phase 2; `--no-image-uuids` turns it off).
* **Mock tokens.** 1 word = 1 token; an image = (w/28)·(h/28) tokens, like Qwen2-VL. Real counts
  are measured in Phase 2.

## Record fields (JSONL)

The first line is `{"meta": {...}}` (trace summary and fingerprint, targets, options). Then one
line per request: `conv_id`, `turn`, `target`, `worker`, `scheduled_s`, `sent_s`, `late_s`,
`ttft_s`, `e2e_s`, `itl_s` (gaps between streamed chunks), `output_chunks`, `prompt_tokens`,
`cached_tokens` (from the server's usage report), `status`, `error`.
