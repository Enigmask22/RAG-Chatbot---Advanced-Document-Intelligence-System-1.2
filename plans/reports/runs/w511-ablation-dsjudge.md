# Ablation tầng sinh — `w511-glm` so với `w511-deepseek`

Mốc: `deepseek-v4-flash` · Ứng viên: `glm-5.3-flash` · Judge: `deepseek-v4-flash`

Cột **macro** là trung bình theo truy vấn (đơn vị lấy mẫu lại của bootstrap), khác micro-average mà `generation_metrics` báo. `CI95` là khoảng tin cậy của **hiệu**; chứa 0 nghĩa là tập này **không phân biệt được** hai nhánh — một phát biểu về cỡ mẫu ít nhất ngang bằng một phát biểu về hệ thống.

| metric | n | mốc | ứng viên | hiệu | CI95 | tốt hơn |
|---|---:|---:|---:|---:|:---:|:---:|
| `answer_relevancy` | 242 | 0.7686 | 0.7769 | +0.0083 | [-0.0207, +0.0372] | — |
| `citation_coverage` | 241 | 0.6160 | 0.8414 | +0.2255 | [+0.1788, +0.2747] | ứng viên |
| `citation_validity` | 211 | 0.8749 | 0.8455 | -0.0294 | [-0.0690, +0.0100] | — |
| `context_precision@5` | 209 | 0.2096 | 0.2096 | +0.0000 | [+0.0000, +0.0000] | — |
| `context_recall@5` | 209 | 0.7887 | 0.7887 | +0.0000 | [+0.0000, +0.0000] | — |
| `faithfulness` | 176 | 0.9725 | 0.9696 | -0.0029 | [-0.0295, +0.0266] | — |
| `false_refusal_rate` | 209 | 0.0861 | 0.0718 | -0.0144 | [-0.0383, +0.0096] | — |
| `misattribution` | 176 | 0.0057 | 0.0179 | +0.0122 | [-0.0053, +0.0281] | — |
| `refusal_accuracy` | 242 | 0.9050 | 0.9174 | +0.0124 | [-0.0124, +0.0372] | — |
| `refusal_recall` | 33 | 0.8485 | 0.8485 | +0.0000 | [-0.0909, +0.0909] | — |
| `uncited_grounding` | 28 | 0.7982 | 0.7024 | -0.0958 | [-0.2429, +0.0470] | — |

## Cảnh báo

* `citation_coverage`: mốc 242 câu, ứng viên 241 câu, so trên 241 câu giao nhau
* `citation_validity`: mốc 212 câu, ứng viên 231 câu, so trên 211 câu giao nhau
* `faithfulness`: mốc 179 câu, ứng viên 223 câu, so trên 176 câu giao nhau
* `misattribution`: mốc 179 câu, ứng viên 223 câu, so trên 176 câu giao nhau
* `uncited_grounding`: mốc 91 câu, ứng viên 46 câu, so trên 28 câu giao nhau
