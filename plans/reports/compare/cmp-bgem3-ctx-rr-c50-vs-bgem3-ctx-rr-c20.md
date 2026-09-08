> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-rr-c20 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6077 | 0.5933 | -0.0144 | p=0.5488 · 7↔4 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.8134 | 0.7703 | -0.0431 | p=0.01172 · 10↔1 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.8278 | 0.7703 | -0.0574 | p=0.0004883 · 12↔0 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.8086 | 0.7560 | -0.0526 | p=0.0009766 · 11↔0 câu đổi chiều | khác biệt thật |
| `map@20` | 0.6449 | 0.6186 | -0.0264 | CI95 [-0.0531, -0.0025] · 17↔15 câu khác nhau (p dấu=0.8601) | khác biệt thật |
| `mrr` | 0.6912 | 0.6601 | -0.0311 | CI95 [-0.0599, -0.0056] · 13↔6 câu khác nhau (p dấu=0.1671) | khác biệt thật |
| `ndcg@10` | 0.6888 | 0.6576 | -0.0312 | CI95 [-0.0588, -0.0072] · 15↔13 câu khác nhau (p dấu=0.8506) | khác biệt thật |
| `precision@1` | 0.6077 | 0.5933 | -0.0144 | p=0.5488 · 7↔4 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.1067 | 0.1014 | -0.0053 | CI95 [-0.0100, -0.0005] · 14↔4 câu khác nhau (p dấu=0.03088) | khác biệt thật |
| `precision@20` | 0.0555 | 0.0517 | -0.0038 | CI95 [-0.0065, -0.0014] · 17↔2 câu khác nhau (p dấu=0.0007286) | khác biệt thật |
| `precision@5` | 0.2105 | 0.1962 | -0.0144 | CI95 [-0.0230, -0.0067] · 14↔1 câu khác nhau (p dấu=0.0009766) | khác biệt thật |
| `recall@1` | 0.4856 | 0.4785 | -0.0072 | CI95 [-0.0359, +0.0215] · 7↔4 câu khác nhau (p dấu=0.5488) | trong ngưỡng nhiễu |
| `recall@10` | 0.7759 | 0.7305 | -0.0455 | CI95 [-0.0774, -0.0167] · 14↔4 câu khác nhau (p dấu=0.03088) | khác biệt thật |
| `recall@20` | 0.7967 | 0.7376 | -0.0590 | CI95 [-0.0925, -0.0287] · 17↔2 câu khác nhau (p dấu=0.0007286) | khác biệt thật |
| `recall@5` | 0.7663 | 0.7145 | -0.0518 | CI95 [-0.0821, -0.0247] · 14↔1 câu khác nhau (p dấu=0.0009766) | khác biệt thật |
