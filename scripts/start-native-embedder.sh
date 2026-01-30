#!/bin/bash
# Start native llama.cpp embedder with Metal GPU
# 210x faster than Docker on Mac Mini M4!
llama-server \
  --model ~/models/embeddinggemma-300M-Q8_0.gguf \
  --embeddings \
  --host 127.0.0.1 \
  --port 8084 \
  --ctx-size 8192 \
  --batch-size 4096 \
  --ubatch-size 4096 \
  --n-gpu-layers 99 \
  "$@"
