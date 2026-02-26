# CEMS Reranker Bottleneck: Solutions Council Report

## Problem Statement
The llama.cpp reranker (Qwen3-Reranker-0.6B) is taking **9-13 seconds per query**, making it the major bottleneck in the retrieval pipeline.

## SOLVED: Root Cause = Docker on Mac Cannot Access Metal GPU

**Docker on macOS runs containers on CPU only** - Metal GPU is not accessible from Docker containers.

### Benchmark Results (Native Metal M4 Pro vs Docker CPU)

| Documents | Docker CPU | Native Metal | Speedup |
|-----------|-----------|--------------|---------|
| 20 docs | ~9-13s | **0.35s** | 25-37x |
| 40 docs | ~13-18s | **0.40s** | 32-45x |

### Solution: Run llama-server natively

```bash
# Install llama.cpp via homebrew (includes Metal support)
brew install llama.cpp

# Copy model from Docker volume
docker cp cems-llama-rerank:/models/Qwen3-Reranker-0.6B.Q8_0.gguf ~/models/

# Start native server with Metal GPU
./scripts/start-native-reranker.sh

# Configure CEMS to use native server
export CEMS_LLAMACPP_RERANK_URL="http://localhost:8083"
```

### OpenRouter Reranking Status

**OpenRouter does NOT have a dedicated reranking API.** They only provide:
- Embeddings (text-embedding-3, Qwen3-Embedding, E5-large, etc.)
- LLMs that could be prompted for reranking (expensive, slow)

For cloud reranking, consider:
- **Cohere Rerank**: $2/1K queries, ~100ms latency
- **Mixedbread.ai**: $7.50/1K queries, ~102ms latency (BEIR leader)
- **Voyage AI Rerank**: $2.50/1K queries, ~600ms latency

---

| Stage | Current Time | Target |
|-------|-------------|--------|
| extract_query_intent (LLM) | 700-1500ms | ✓ OK |
| Batch embedding | 180-235ms | ✓ OK |
| DB search | 4-8ms | ✓ OK |
| **llama.cpp reranking** | **9000-13000ms** | **< 500ms** |

---

## Root Cause Analysis

### 1. Docker Configuration Issues (Critical)
- **No GPU support configured** - llama.cpp running on CPU only
- **No memory limits** - potential memory thrashing
- **No batch-size tuning** for reranker container
- Docker overhead itself is minimal (~10-25%), NOT the main issue

### 2. Model Size vs Speed Trade-off
- Qwen3-Reranker-0.6B is 600M parameters
- Q8_0 quantization is accurate but slower than Q4_K_M
- Processing 40 candidates sequentially

### 3. How QMD Handles This
QMD uses the **same model** (Qwen3-Reranker-0.6B) but with:
- **Position-aware blending**: Top-3 results trust retrieval 75%, reranker 25%
- **Top-30 candidate limit** (vs your 40)
- **Caching of LLM outputs** by operation hash
- **Strong-signal skip**: If BM25 has clear winner, skip reranking entirely

---

## Solutions Matrix (Ranked by Impact/Effort)

### Tier 1: Quick Wins (1-2 hours each)

| Solution | Speedup | Effort | Quality Impact |
|----------|---------|--------|----------------|
| **1. Disable reranking** (already default) | ∞ | None | -5% recall |
| **2. Q4_K_M quantization** | 1.5-2x | 5 min | < 1% loss |
| **3. Reduce candidates to 20** | ~2x | 5 min | -1-2% recall |
| **4. Add GPU support to Docker** | 5-10x | 30 min | None |
| **5. Confidence-based skip** | 2-3x avg | 1 hr | None |

### Tier 2: Alternative Models (Sub-500ms targets)

| Model | Size | Latency (40 docs) | Quality vs Qwen3 |
|-------|------|-------------------|------------------|
| **jina-reranker-v1-tiny** | 33M | ~150-250ms | 92.5% |
| **ms-marco-MiniLM-L-6-v2** | 22.7M | ~200-350ms | ~95% |
| **cross-encoder/TinyBERT-L-2** | 4.4M | ~25-50ms | ~85% |
| **Cohere Rerank API** | Cloud | ~100ms | Best |
| **Mixedbread.ai API** | Cloud | ~102ms | BEIR leader |

### Tier 3: Architectural Changes (1-2 days)

| Approach | Description | Speedup |
|----------|-------------|---------|
| **MLP on embeddings** | Train tiny neural net on query+doc embeddings | 50-100x |
| **XGBoost LTR** | Learning-to-rank with hand-crafted features | 100-1000x |
| **Tiered reranking** | Fast path → slow path based on confidence | 3-5x avg |
| **Query-level caching** | Cache reranking results for similar queries | 10x for hits |

---

## Recommended Implementation Plan

### Phase 1: Immediate (Today) - Target: <2s latency

**Option A: Disable reranking entirely**
```python
# Already the default - just verify it's disabled
reranker_backend: "disabled"
```
Result: ~400ms total latency, -5% recall acceptable for most queries

**Option B: GPU + Quantization optimizations**

1. **Update docker-compose.yml for GPU support:**
```yaml
llama-rerank:
  image: ghcr.io/ggml-org/llama.cpp:server-cu122
  command:
    - --model
    - /models/Qwen3-Reranker-0.6B.Q4_K_M.gguf  # Faster quantization
    - --rerank
    - --ctx-size
    - "512"
    - --batch-size
    - "256"
    - --ubatch-size
    - "256"
    - --n-gpu-layers
    - "40"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
      limits:
        memory: 2G
```

2. **Reduce candidate limit:**
```python
# In config.py
rerank_input_limit: int = 20  # Down from 40
```

Expected result: **500ms-1.5s** (5-10x improvement with GPU)

### Phase 2: This Week - Target: <500ms with reranking

**Add confidence-based reranking skip:**
```python
# In memory/retrieval.py
def should_rerank(candidates: list[SearchResult], threshold_gap: float = 0.15) -> bool:
    """Skip reranking if top result has high confidence."""
    if len(candidates) < 2:
        return False
    gap = candidates[0].score - candidates[1].score
    return gap < threshold_gap  # Only rerank if scores are close

# In retrieve_for_inference_async
if enable_rerank and len(candidates) > 3 and should_rerank(candidates):
    candidates = await rerank_with_llamacpp(...)
else:
    # Skip reranking - top result is confident
    pass
```

Expected: ~50% of queries skip reranking entirely

### Phase 3: Next Week - Lightweight Reranker Alternative

**Option A: Switch to jina-reranker-v1-tiny (Recommended)**
- Download: `jina-reranker-v1-tiny-en` GGUF
- 33M params → ~150-200ms for 20 candidates
- 92.5% quality of larger models

**Option B: MLP on cached embeddings (Most innovative)**
```python
# Train tiny MLP on embedding concatenation
class MLPReranker(nn.Module):
    def __init__(self, embedding_dim=1536):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, query_emb, doc_emb):
        combined = torch.cat([query_emb, doc_emb], dim=-1)
        return self.layers(combined)
```
- Train on LongMemEval gold rankings
- <5ms inference for all candidates
- ~95% quality with proper training

---

## Files to Modify

| File | Change |
|------|--------|
| `docker-compose.yml` | GPU support, Q4_K_M model, batch tuning |
| `src/cems/config.py` | Reduce `rerank_input_limit` to 20 |
| `src/cems/memory/retrieval.py` | Add `should_rerank()` confidence check |
| `src/cems/llamacpp_server.py` | (Optional) Add latency logging |

---

## Verification Plan

1. **Baseline measurement:**
   ```bash
   docker logs cems-llama-rerank 2>&1 | grep -i "rerank\|latency"
   ```

2. **After GPU/quantization changes:**
   - Run test query via MCP tool
   - Check docker logs for rerank timing
   - Target: <1.5s per rerank call

3. **After confidence skip:**
   - Run 10 test queries
   - Count how many skip reranking
   - Target: 50%+ skip rate

4. **Quality validation:**
   - Run LongMemEval benchmark
   - Compare recall@5 before/after
   - Acceptable regression: <3%

---

## Decision Points

Before implementing, I need to clarify:

1. **Do you have an NVIDIA GPU available?**
   - If yes → GPU acceleration is the fastest path to <1s
   - If no → Model replacement or disabling is better

2. **What's your quality tolerance?**
   - Accept 5% recall loss → Disable reranking (fastest)
   - Accept 2-3% recall loss → Use jina-tiny or MLP
   - No quality loss → GPU + confidence skip only

3. **Cloud APIs acceptable?**
   - Cohere/Mixedbread can achieve 100ms latency
   - Cost: ~$2-7.50 per 1000 queries
