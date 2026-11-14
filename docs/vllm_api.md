# vLLM reference for kvroute (pinned: vLLM 0.32.0)

Everything the router and benchmarks rely on, for **one pinned version**. Re-check this file
before changing the pin. Each item is tagged:

- **[source]**: read from the v0.32.0 source on GitHub (tag `v0.32.0`)
- **[docs]**: from the v0.32.0 docs
- **[verify]**: not confirmed yet; check on the first GPU run (Phase 2)

## Version pin

- **vLLM 0.32.0** (released 2026-10-18). 0.33.0 (2026-11-02) is too new to pin yet.
- Python 3.11-3.14; wheels built against CUDA 12.9 by default. **[docs]**
- Install: `pip install vllm==0.32.0`

## Running on a T4 (compute capability 7.5)

- Minimum compute capability is 7.5 and the T4 is listed by name. **[docs]**
- No bf16 below compute capability 8.0: vLLM raises an error for bf16 on a T4. Always pass
  `--dtype float16`. **[source: `platforms/cuda.py`]**
- FlashAttention needs 8.0+. For standard attention the backend priority list is
  FLASH_ATTN -> FLASHINFER -> TRITON_ATTN -> FLEX_ATTENTION, so a T4 should fall back to
  FlashInfer or Triton. **[source]** Confirm in the startup log which one is picked. **[verify]**
- If the auto choice fails, try `--attention-backend TRITON_ATTN`. **[verify the flag name]**

## Prefix caching

| setting | default | notes |
|---|---|---|
| `enable_prefix_caching` | **True** | on by default **[source: `config/cache.py`]** |
| `prefix_caching_hash_algo` | `sha256` | choices `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor` **[source]** |
| `block_size` | model/platform dependent | pass `--block-size 16` explicitly so the router knows it |
| `gpu_memory_utilization` | 0.92 | lower it to shrink the KV budget, or to fit 2 instances on one T4 |
| `num_gpu_blocks_override` | None | sets the KV budget in blocks exactly; best for the cache-size experiments |
| `kv_cache_memory_bytes` | None | alternative to the above; ignores `gpu_memory_utilization` |

CLI flags are the field names with dashes (`--prefix-caching-hash-algo`, `--num-gpu-blocks-override`, ...). **[verify]**

### How a block hash is computed **[source: `v1/core/kv_cache_utils.py`]**

```
hash(block_i) = H( (hash(block_{i-1}) or NONE_HASH,  tuple(token_ids of block_i),  extra_keys_i) )
```

- Only **full** blocks are hashed and cached.
- `H` = the chosen algorithm. `sha256` pickles the tuple first (Python-specific).
  `sha256_cbor` uses canonical CBOR, which is reproducible from another process or language.
- `NONE_HASH` (the "parent" of block 0) = `H(seed)`. For sha256 variants the seed is the fixed
  string `"vllm-none-hash"` unless `PYTHONHASHSEED` is set. For xxhash variants it is random per
  process unless `PYTHONHASHSEED` is set.
- `extra_keys_i` is `None` for plain text blocks. Otherwise it is a tuple of: LoRA name keys,
  image keys `("mm", identifier, offset_of_image_start_relative_to_block_start)` for every image
  overlapping the block, `("cache_salt", salt)` on block 0 only, and prompt-embedding digests.

**Router decision:** run every worker with `--prefix-caching-hash-algo sha256_cbor` and the same
`PYTHONHASHSEED`, and use `--block-size 16`. Then the router can compute identical hashes if it
knows the token IDs and image identifiers.

### Image identifiers **[source: `multimodal/processing/inputs.py`, `multimodal/hasher.py`]**

- By default an image's identifier = `H_mm(model_id, image content, processor kwargs)`, with
  `H_mm` = blake3 by default (sha256/sha512 optional). The content is the raw pixel array plus
  mode, or the original file bytes. A UUID in the image's EXIF `ImageID` field is used instead
  if present.
- **The client can supply its own UUID per image.** If it does and there are no extra processor
  kwargs, vLLM uses that UUID **verbatim** as the identifier. With processor kwargs, it hashes
  the UUID together with them.
- **Router decision:** the load generator sends a stable UUID per image (e.g. `coco-000000139`).
  The router then knows each image's identifier without hashing pixels. How the UUID is passed
  in an OpenAI chat request (a `uuid` field on the image content part) is **[verify]**.
- The router still needs each image's **token count and position** to place it in blocks; that
  comes from the model's processor (same image size -> same count). **[verify per model]**

## KV-cache events **[source: `config/kv_events.py`, docs example]**

Config (JSON passed to `--kv-events-config`):

| field | default | meaning |
|---|---|---|
| `enable_kv_cache_events` | False | must be True |
| `publisher` | `"zmq"` when enabled, else `"null"` | |
| `endpoint` | `"tcp://*:5557"` | PUB socket; use a different port per worker |
| `replay_endpoint` | None | ROUTER socket for replaying missed events, e.g. `"tcp://*:5558"` |
| `buffer_steps` | 10000 | steps kept for replay |
| `hwm` | 100000 | ZMQ high-water mark; events drop past it |
| `max_queue_size` | 100000 | |
| `topic` | `""` | subscribe with the same topic |

Example launch (two workers on one T4 for the 4-worker setup):

```bash
PYTHONHASHSEED=0 vllm serve Qwen/Qwen2-VL-2B-Instruct --port 8001 --dtype float16 \
  --block-size 16 --prefix-caching-hash-algo sha256_cbor --gpu-memory-utilization 0.45 \
  --kv-events-config '{"enable_kv_cache_events": true, "endpoint": "tcp://*:5557", "replay_endpoint": "tcp://*:5558", "topic": "kv-events"}'
```

Wire format (subscriber side):

- Multipart message: `[topic, seq (8-byte big-endian int), payload (msgpack)]`.
- Decode with `msgspec.msgpack.Decoder(type=KVEventBatch)`.
- On a sequence gap, send `(last_seq + 1).to_bytes(8, "big")` to the replay socket; it streams
  the missed messages and ends with an empty payload.

Event types:

- `KVEventBatch`: `ts: float`, `events: list[BlockStored | BlockRemoved | AllBlocksCleared]`
  (plus a data-parallel rank in newer versions **[verify]**)
- `BlockStored`: `block_hashes`, `parent_block_hash`, `token_ids`, `block_size`, `lora_id`
  (deprecated), `medium`, `lora_name`, `extra_keys`, `group_idx`, `kv_cache_spec_kind`,
  `kv_cache_spec_sliding_window`, `locality`, `session_id`
- `BlockRemoved`: `block_hashes`, `medium`, `group_idx`, `locality`
- `AllBlocksCleared`: no fields

Details that matter for the index:

- Hashes in events are raw bytes by default. With env `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1`
  they are the low 64 bits as an int. Pick one and use it everywhere.
- Image keys in events are published **untagged**: `(identifier, offset)` pairs, not
  `("mm", identifier, offset)`. Re-add the `"mm"` tag when recomputing hashes.
- Because `BlockStored` carries `token_ids`, `parent_block_hash` and `extra_keys`, the router can
  check its own hash function against real events in a test: recompute and compare.

## Metrics (`GET /metrics`, Prometheus text) **[docs]**

| use | metric |
|---|---|
| ground-truth hit rate | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` (counters, in tokens) |
| load | `vllm:num_requests_running` (gauge), `vllm:num_requests_waiting` (gauge) **[verify name]** |
| memory pressure | `vllm:kv_cache_usage_perc` (gauge) |
| server-side latency | `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds` (histograms) |
| prompt sizes | `vllm:request_prompt_tokens` (histogram) |

Counters are cumulative per process: take differences between scrapes, and restart workers
between experiment cells (or record the values at the start of each cell).

## Open questions for the first GPU run (Phase 2)

1. Which attention backend does a T4 pick, and is fp16 output from the chosen VLM sane?
2. Exact OpenAI request field for a per-image UUID.
3. Image token count per image size for the chosen model.
4. Does `vllm:num_requests_waiting` exist under that name?
5. Kaggle's Python and CUDA versions vs the wheel's requirements.
