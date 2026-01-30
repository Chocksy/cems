#!/bin/bash
# Start native llama.cpp reranker with Metal GPU
llama-server \
  --model ~/models/Qwen3-Reranker-0.6B.Q8_0.gguf \
  --rerank \
  --host 127.0.0.1 \
  --port 8083 \
  --ctx-size 512 \
  --batch-size 256 \
  --n-gpu-layers 99 \
  "$@"

