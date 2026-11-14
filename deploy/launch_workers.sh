#!/usr/bin/env bash
# Start N vLLM workers by hand (one per GPU, ports 8001, 8002, ...), with the same flags
# deploy/workers.py uses. Mostly for debugging; experiments use the Python launcher.
#   N=2 MODEL=Qwen/Qwen2-VL-2B-Instruct deploy/launch_workers.sh
set -euo pipefail
N=${N:-1}; MODEL=${MODEL:-Qwen/Qwen2-VL-2B-Instruct}; BASE=${BASE:-8001}
mkdir -p logs
for i in $(seq 0 $((N - 1))); do
  port=$((BASE + i))
  CUDA_VISIBLE_DEVICES=$i PYTHONHASHSEED=0 nohup vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$port" --dtype float16 --block-size 16 \
    --prefix-caching-hash-algo sha256_cbor --enable-prompt-tokens-details \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 8}' --served-model-name model \
    > "logs/vllm_${port}.log" 2>&1 &
  echo "worker $i -> http://127.0.0.1:$port (log: logs/vllm_${port}.log)"
done
