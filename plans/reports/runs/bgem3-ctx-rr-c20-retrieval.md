# Retrieval eval — `bgem3-ctx-rr-c20`

- Thời điểm chạy: `2026-09-08T03:32:10+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n20", "top_k": 20, "index_config": "configs\\indexing\\bgem3-contextual.yaml", "index_fingerprint": "9e3bc146392b3051ce126708cd41b0b77dad243e0a44103a1e14783a82d08a85", "collection": "rag_bgem3_ctx", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"k": 1, "candidate_k": 20, "base": "hybrid", "rerank_candidates": 20}, "chunking": {"strategy": "hybrid", "size_unit": "chars", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "parent_size_multiple": 4, "structure_merge_short_sections": true, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5933 |
| hit_rate@10 | 0.7703 |
| hit_rate@20 | 0.7703 |
| hit_rate@5 | 0.7560 |
| map@20 | 0.6186 |
| mrr | 0.6601 |
| ndcg@10 | 0.6576 |
| precision@1 | 0.5933 |
| precision@10 | 0.1014 |
| precision@20 | 0.0517 |
| precision@5 | 0.1962 |
| recall@1 | 0.4785 |
| recall@10 | 0.7305 |
| recall@20 | 0.7376 |
| recall@5 | 0.7145 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 303.1 |
| p50 | 305.4 |
| p95 | 341.7 |
| max | 366.5 |
| stdev | 24.9 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.4706 | 0.6765 | 0.6765 | 0.6765 | 0.5466 | 0.5539 | 0.5815 | 0.4706 | 0.0735 | 0.0368 | 0.1471 | 0.4412 | 0.6765 | 0.6765 | 0.6765 |
| aggregation | 26 | 0.6538 | 0.8846 | 0.8846 | 0.8462 | 0.5714 | 0.7491 | 0.6523 | 0.6538 | 0.1692 | 0.0904 | 0.3077 | 0.2949 | 0.7115 | 0.7564 | 0.6603 |
| cross_lingual | 43 | 0.3953 | 0.4884 | 0.4884 | 0.4884 | 0.4281 | 0.4264 | 0.4433 | 0.3953 | 0.0581 | 0.0291 | 0.1163 | 0.3605 | 0.4884 | 0.4884 | 0.4884 |
| factoid | 68 | 0.7059 | 0.9118 | 0.9118 | 0.8971 | 0.7836 | 0.7824 | 0.8156 | 0.7059 | 0.0926 | 0.0463 | 0.1824 | 0.7059 | 0.9118 | 0.9118 | 0.8971 |
| multi_hop | 34 | 0.7059 | 0.8824 | 0.8824 | 0.8529 | 0.6515 | 0.7680 | 0.7115 | 0.7059 | 0.1559 | 0.0794 | 0.3000 | 0.3480 | 0.7696 | 0.7794 | 0.7402 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.6341 | 0.8537 | 0.8537 | 0.8293 | 0.6688 | 0.7162 | 0.7185 | 0.6341 | 0.1244 | 0.0640 | 0.2366 | 0.4736 | 0.8150 | 0.8272 | 0.7846 |
| vi | 127 | 0.5669 | 0.7165 | 0.7165 | 0.7087 | 0.5862 | 0.6238 | 0.6183 | 0.5669 | 0.0866 | 0.0437 | 0.1701 | 0.4816 | 0.6759 | 0.6798 | 0.6693 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
