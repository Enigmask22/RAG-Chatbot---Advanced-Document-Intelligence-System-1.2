# Retrieval eval — `bgem3-ctx-rr-c20-w025`

- Thời điểm chạy: `2026-09-08T03:34:47+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20-w1:0.25]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n20", "top_k": 20, "index_config": "configs\\indexing\\bgem3-contextual.yaml", "index_fingerprint": "9e3bc146392b3051ce126708cd41b0b77dad243e0a44103a1e14783a82d08a85", "collection": "rag_bgem3_ctx", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"k": 1, "candidate_k": 20, "weights": [1.0, 0.25], "base": "hybrid", "rerank_candidates": 20}, "chunking": {"strategy": "hybrid", "size_unit": "chars", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "parent_size_multiple": 4, "structure_merge_short_sections": true, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5789 |
| hit_rate@10 | 0.7656 |
| hit_rate@20 | 0.7703 |
| hit_rate@5 | 0.7560 |
| map@20 | 0.6081 |
| mrr | 0.6526 |
| ndcg@10 | 0.6493 |
| precision@1 | 0.5789 |
| precision@10 | 0.1000 |
| precision@20 | 0.0510 |
| precision@5 | 0.1933 |
| recall@1 | 0.4641 |
| recall@10 | 0.7265 |
| recall@20 | 0.7337 |
| recall@5 | 0.7057 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 311.2 |
| p50 | 312.6 |
| p95 | 355.9 |
| max | 427.3 |
| stdev | 29.6 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.4706 | 0.6765 | 0.6765 | 0.6765 | 0.5490 | 0.5539 | 0.5827 | 0.4706 | 0.0735 | 0.0368 | 0.1471 | 0.4412 | 0.6765 | 0.6765 | 0.6765 |
| aggregation | 26 | 0.6538 | 0.8462 | 0.8846 | 0.8462 | 0.5552 | 0.7462 | 0.6296 | 0.6538 | 0.1538 | 0.0846 | 0.3000 | 0.2949 | 0.6667 | 0.7244 | 0.6474 |
| cross_lingual | 43 | 0.3721 | 0.5116 | 0.5116 | 0.4884 | 0.4190 | 0.4181 | 0.4421 | 0.3721 | 0.0605 | 0.0302 | 0.1116 | 0.3372 | 0.5116 | 0.5116 | 0.4767 |
| factoid | 68 | 0.6765 | 0.8971 | 0.8971 | 0.8824 | 0.7612 | 0.7599 | 0.7951 | 0.6765 | 0.0912 | 0.0456 | 0.1794 | 0.6765 | 0.8971 | 0.8971 | 0.8824 |
| multi_hop | 34 | 0.7059 | 0.8824 | 0.8824 | 0.8824 | 0.6536 | 0.7794 | 0.7189 | 0.7059 | 0.1588 | 0.0794 | 0.3000 | 0.3480 | 0.7794 | 0.7794 | 0.7402 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5976 | 0.8293 | 0.8415 | 0.8171 | 0.6404 | 0.6906 | 0.6920 | 0.5976 | 0.1183 | 0.0604 | 0.2268 | 0.4431 | 0.7907 | 0.7988 | 0.7622 |
| vi | 127 | 0.5669 | 0.7244 | 0.7244 | 0.7165 | 0.5873 | 0.6280 | 0.6217 | 0.5669 | 0.0882 | 0.0449 | 0.1717 | 0.4777 | 0.6850 | 0.6916 | 0.6693 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
