# 05_lookup_db (placeholder)

This stage is intentionally lightweight for now.

Current reality:
- `06_eval/` and `07_web_scanner/` can load vector tables into memory and do fast exact nearest-neighbor lookup directly.
- For current dataset sizes, exact search is often simple and fast enough.

Typical exact-search approaches in use:
- **Matrix multiply / dot-product search** on L2-normalized vectors (cosine similarity)
- **Batched top-k selection** (`argpartition`/`topk`) over similarity scores
- Optional reranking over shortlisted candidates

When this folder becomes important:
- if corpus size, latency, or memory constraints outgrow exact search
- then we can add ANN indices (e.g., HNSW/IVF/PQ-style approaches) and build pipelines here

Compatibility note:
- Existing scripts remain here so old workflows continue to run.
