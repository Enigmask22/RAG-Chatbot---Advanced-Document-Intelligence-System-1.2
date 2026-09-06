# Ablation tầng sinh — `w511-glm` so với `w511-deepseek`

Mốc: `deepseek-v4-flash` · Ứng viên: `glm-5.3-flash` · Judge: `glm-5.3-flash`

Cột **macro** là trung bình theo truy vấn (đơn vị lấy mẫu lại của bootstrap), khác micro-average mà `generation_metrics` báo. `CI95` là khoảng tin cậy của **hiệu**; chứa 0 nghĩa là tập này **không phân biệt được** hai nhánh — một phát biểu về cỡ mẫu ít nhất ngang bằng một phát biểu về hệ thống.

| metric | n | mốc | ứng viên | hiệu | CI95 | tốt hơn |
|---|---:|---:|---:|---:|:---:|:---:|
| `answer_relevancy` | 242 | 0.7686 | 0.7851 | +0.0165 | [-0.0083, +0.0413] | — |
| `citation_coverage` | 241 | 0.6160 | 0.8414 | +0.2255 | [+0.1788, +0.2747] | ứng viên |
| `citation_validity` | 211 | 0.8749 | 0.8455 | -0.0294 | [-0.0690, +0.0100] | — |
| `context_precision@5` | 209 | 0.2096 | 0.2096 | +0.0000 | [+0.0000, +0.0000] | — |
| `context_recall@5` | 209 | 0.7887 | 0.7887 | +0.0000 | [+0.0000, +0.0000] | — |
| `faithfulness` | 181 | 0.9571 | 0.9683 | +0.0112 | [-0.0222, +0.0449] | — |
| `false_refusal_rate` | 209 | 0.0909 | 0.0766 | -0.0144 | [-0.0431, +0.0144] | — |
| `misattribution` | 179 | 0.0301 | 0.0246 | -0.0055 | [-0.0347, +0.0230] | — |
| `refusal_accuracy` | 242 | 0.9008 | 0.9132 | +0.0124 | [-0.0124, +0.0413] | — |
| `refusal_recall` | 33 | 0.8485 | 0.8485 | +0.0000 | [-0.0909, +0.0909] | — |
| `uncited_grounding` | 31 | 0.9032 | 0.8387 | -0.0645 | [-0.2258, +0.0968] | — |

## Cảnh báo

* `citation_coverage`: mốc 242 câu, ứng viên 241 câu, so trên 241 câu giao nhau
* `citation_validity`: mốc 212 câu, ứng viên 231 câu, so trên 211 câu giao nhau
* `faithfulness`: mốc 186 câu, ứng viên 228 câu, so trên 181 câu giao nhau
* `misattribution`: mốc 185 câu, ứng viên 227 câu, so trên 179 câu giao nhau
* `uncited_grounding`: mốc 101 câu, ứng viên 47 câu, so trên 31 câu giao nhau
