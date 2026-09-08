> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-ctx-rr-c50-w025 | bgem3-ctx-rr-c20-w025 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6220 | 0.5789 | -0.0431 | p=0.01172 · 10↔1 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.8325 | 0.7656 | -0.0670 | p=0.0001221 · 14↔0 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.8469 | 0.7703 | -0.0766 | p=3.052e-05 · 16↔0 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.8230 | 0.7560 | -0.0670 | p=0.0001221 · 14↔0 câu đổi chiều | khác biệt thật |
| `map@20` | 0.6636 | 0.6081 | -0.0554 | CI95 [-0.0861, -0.0285] · 27↔14 câu khác nhau (p dấu=0.05958) | khác biệt thật |
| `mrr` | 0.7047 | 0.6526 | -0.0521 | CI95 [-0.0841, -0.0238] · 17↔3 câu khác nhau (p dấu=0.002577) | khác biệt thật |
| `ndcg@10` | 0.7079 | 0.6493 | -0.0586 | CI95 [-0.0898, -0.0307] · 24↔12 câu khác nhau (p dấu=0.06525) | khác biệt thật |
| `precision@1` | 0.6220 | 0.5789 | -0.0431 | p=0.01172 · 10↔1 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.1105 | 0.1000 | -0.0105 | CI95 [-0.0158, -0.0057] · 24↔3 câu khác nhau (p dấu=4.923e-05) | khác biệt thật |
| `precision@20` | 0.0577 | 0.0510 | -0.0067 | CI95 [-0.0093, -0.0043] · 27↔0 câu khác nhau (p dấu=1.49e-08) | khác biệt thật |
| `precision@5` | 0.2153 | 0.1933 | -0.0220 | CI95 [-0.0316, -0.0134] · 22↔0 câu khác nhau (p dấu=4.768e-07) | khác biệt thật |
| `recall@1` | 0.5000 | 0.4641 | -0.0359 | CI95 [-0.0646, -0.0096] · 10↔1 câu khác nhau (p dấu=0.01172) | khác biệt thật |
| `recall@10` | 0.8022 | 0.7265 | -0.0758 | CI95 [-0.1116, -0.0431] · 24↔3 câu khác nhau (p dấu=4.923e-05) | khác biệt thật |
| `recall@20` | 0.8246 | 0.7337 | -0.0909 | CI95 [-0.1268, -0.0574] · 27↔0 câu khác nhau (p dấu=1.49e-08) | khác biệt thật |
| `recall@5` | 0.7847 | 0.7057 | -0.0789 | CI95 [-0.1132, -0.0478] · 22↔0 câu khác nhau (p dấu=4.768e-07) | khác biệt thật |
