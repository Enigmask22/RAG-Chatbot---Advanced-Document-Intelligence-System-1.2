# `NEW-09` — `rerank_candidates` 50 → 20: đo xong, và **bác bỏ**

*2026-09-08 · 2 lượt eval mới (209 câu × 2) · 2 kiểm định · GPU cục bộ · chi phí **$0***

> **Nợ:** *"Bundle `rerank_candidates=20`: eval lại, qua `make gate`, đo lại trần
> thông lượng + TTFT."* `W6-05` gọi đây là **đòn bẩy kép duy nhất còn lại**.

---

## 0. Kết luận trước, số sau

**Không đổi.** `c=20` mua **2,21×** độ trễ truy hồi (691,6 → 312,6 ms p50) và
một trần thông lượng ~**3,1 req/s** — nhưng trả bằng **15/15 metric xấu đi, mọi
CI loại trừ 0** so với bundle đang phục vụ.

`W6-05` **đúng** về đòn bẩy: thông lượng dự báo ~3,3 req/s, tính lại được ~3,1.
Thứ **sai** là giả định chất lượng đi kèm — và nó không đến từ `W6-05` mà từ một
câu của `W2-08` được mang sang một cấu hình khác.

---

## 1. ⭐⭐ "`c=20` giữ 91% mức cải thiện" **không chuyển được** sang bundle đang chạy

Câu ấy là lý do nợ này tồn tại, và nó đo trên **chunk không ngữ cảnh**. Bundle
`0.2.1` phục vụ **chunk có ngữ cảnh + trọng số RRF (1 : 0,25)**. Tính lại phần
cải thiện của reranker còn giữ được, trên đúng cấu hình đang chạy (nền = dense
trên cùng index, `bgem3-ctx` nDCG@10 0,5019 / recall@10 0,6348):

| | nDCG@10 | recall@10 |
|---|---:|---:|
| nền (dense, không rerank) | 0,5019 | 0,6348 |
| `c=50` + trọng số — **đang phục vụ** | 0,7079 | 0,8022 |
| `c=20` + trọng số | 0,6493 | 0,7265 |
| **phần cải thiện còn giữ** | **71,6%** | **54,8%** |

Không phải 91%. Và trên recall@10 — metric mà `G6` đặt ngưỡng — nó mất **gần
một nửa**.

⚠️ Bài học không phải "`W2-08` sai". `W2-08` đúng **với thứ nó đo**. Lỗi là mang
một tỉ lệ đo trên cấu hình A sang làm cơ sở quyết định cho cấu hình B, và giữa
A và B đã đổi **hai** thứ (chunk có ngữ cảnh, trọng số RRF). Cùng họ với
`TD-56` và với chuyện `W6-08` bỏ *"80–90% cache hit"*: một con số đúng trong
điều kiện của nó, trích ra khỏi điều kiện ấy thì thành sai.

---

## 2. Bảng 2×2 — và nó có được là nhờ một lượt chạy sai

| nDCG@10 | `n=50` | `n=20` | Δ |
|---|---:|---:|---:|
| **không trọng số** | 0,6888 | 0,6576 | −0,0312 |
| **trọng số 1 : 0,25** *(đang phục vụ)* | **0,7079** | 0,6493 | **−0,0586** |

Lượt eval đầu của tôi chạy **sai nhánh**: tôi bỏ `--rrf-weights 1 0.25`, nên
chuỗi danh tính ra `rrf1-c20]:…n20` trong khi bundle phục vụ
`rrf1-c20-w1:0.25]:…n50`. Lệch **hai trục** cùng lúc ⇒ mọi Δ quy công sai.

⭐ **`TD-38` bắt được nó, đúng như nó được sinh ra để làm**: `retriever_name` gom
mọi thứ làm đổi con số, nên so hai chuỗi là thấy ngay. Không có nó thì tôi đã
báo cáo −0,0312 và gọi đó là giá của việc giảm pool.

⭐ Và lượt sai **trở thành ô chứng**: nó cô lập trục trọng số, cho ra bảng 2×2
đầy đủ thay vì một cặp.

### ⚠️ Nhưng giả thuyết tôi rút ra từ bảng ấy **không đứng được**

Nhìn bảng thì có vẻ hai đòn bẩy **tương tác**: trọng số lãi +0,0191 ở `n=50`
nhưng lỗ −0,0083 ở `n=20`, tức pool nông làm mất lợi thế của việc hạ trọng số
sparse. Câu chuyện nghe hợp lý.

Đo thì **không kết luận được**:

| | Δ nDCG@10 | CI95 |
|---|---:|---|
| không trọng số, `50 → 20` | −0,0312 | [−0,0588, −0,0072] |
| có trọng số, `50 → 20` | −0,0586 | [−0,0898, −0,0307] |

Hai khoảng **chồng nhau** trên đoạn [−0,0588, −0,0307]. Cùng kết quả ở recall@10.
Nên tôi **không** tuyên bố có tương tác — chỉ ghi rằng ước lượng điểm gợi ý thế
và bằng chứng chưa đủ.

💡 Và chồng CI vốn là một phép kiểm **yếu** cho sự khác biệt; thứ đúng phải là
một kiểm định ghép trên *hiệu của hai hiệu*. Repo chưa có công cụ ấy. Nói ra chỗ
mình không đo được thay vì để khoảng chồng nhau làm cái cớ.

---

## 3. Kiểm định so với bundle đang phục vụ: **15/15 xấu đi**

`cmp-bgem3-ctx-rr-c50-w025-vs-bgem3-ctx-rr-c20-w025.md`. Trích bốn hàng nặng
nhất; **cả 15 hàng đều kết luận "khác biệt thật"**, không hàng nào đi ngược:

| metric | `c=50`+w | `c=20`+w | Δ | kiểm định |
|---|---:|---:|---:|---|
| `recall@20` | 0,8246 | 0,7337 | −0,0909 | CI95 [−0,1268, −0,0574] · 27↔0 |
| `recall@5` | 0,7847 | 0,7057 | −0,0789 | CI95 [−0,1132, −0,0478] · 22↔0 |
| `recall@10` | 0,8022 | 0,7265 | −0,0758 | CI95 [−0,1116, −0,0431] · 24↔3 |
| `ndcg@10` | 0,7079 | 0,6493 | −0,0586 | CI95 [−0,0898, −0,0307] · 24↔12 |

⚠️ Bảng ấy tự mang cảnh báo **không hiệu chỉnh đa so sánh** (`W2-09`: ~7 phép
kiểm hiệu dụng, số hàng "có ý nghĩa" do ngẫu nhiên ≈ 0,35). Ở đây điều đó
**không cứu được** `c=20`: nó không đi ngược chiều ở một hàng nào, và ba hàng
`recall` có `27↔0`, `22↔0` — tức **không một câu hỏi nào** tốt lên.

---

## 4. Nửa thông lượng: `W6-05` dự báo đúng

`W6-05` tính trần = **91% của `1 / thời-gian-rerank`**, và `W5-06` đo rerank
chiếm **92,8%** chặng truy hồi.

| | p50 truy hồi | rerank ước tính | trần dự báo |
|---|---:|---:|---:|
| `c=50` + w *(đang phục vụ)* | 691,6 ms | ~642 ms | **1,42 req/s** |
| `c=20` + w | 312,6 ms | ~290 ms | **3,14 req/s** |

**2,21×**, so với ~2,5× mà `W6-05` dự báo (1,33 → 3,3). Sai số nhỏ và cùng
chiều: **đòn bẩy có thật, và nó là đòn bẩy duy nhất còn lại**.

Nên nợ này không đóng bằng "hoá ra chẳng có gì". Nó đóng bằng: **cái giá đã
biết, và giá ấy quá đắt cho chỗ hôm nay**. Khi nào recall@10 đạt ngưỡng `G6`
0,90 và còn dư, cân nhắc lại — lúc ấy đổi 8 điểm recall lấy 2,2× thông lượng có
thể là mua được.

---

## 5. ⚠️ Cố ý **không** đúc bundle và **không** chạy `make gate`

Nợ ghi *"qua `make gate`"*. Tôi dừng trước bước ấy, có chủ đích:

* Gate so ứng viên với **champion đang phục vụ**, và ứng viên thua **15/15 với
  mọi CI loại trừ 0**. Kết quả `FAIL` đã biết trước khi chạy.
* Đúc một bundle cho cấu hình vừa bị bác bỏ là bỏ một artifact rác vào
  `bundles/` — và `W5-10` đã dạy rằng `save_bundle` một RC **là một lần deploy**.

⚠️ Thứ **không** được kiểm nhờ quyết định này: hành vi của chính `make gate`
trên đầu vào ấy. `W5-05` đã chạy nhánh `FAIL` thật (`citation_accuracy` 0,8308 <
0,85 · `p95` 4706 > 3500, exit 1), nên cơ chế có bằng chứng — chỉ không phải
bằng chứng từ lượt này. Ghi ra thay vì để trống.

---

## 6. Việc sinh ra từ lượt này

* ~~`NEW-09`~~ **đóng** — đo xong, bác bỏ. `c=50` + trọng số giữ nguyên.
* `TD-63` (trần 1,33 req/s vì khoá tuần tự hoá GPU) **vẫn mở, và giờ hẹp hơn**:
  đường (a) — giảm `rerank_candidates` — đã **đo và loại**. Còn lại đúng đường
  (b): **thêm container + thêm GPU**. Không còn phương án phần mềm nào.
* 💡 `bgem3-ctx-dense-rr-c50` cho nDCG@10 **0,6937**, cao hơn `bgem3-ctx-rr-c50`
  **không trọng số** (0,6888) và chỉ kém bản có trọng số (0,7079) — tức nhánh
  hybrid **chỉ có lãi khi sparse bị hạ trọng số**. Quan sát rơi ra từ bảng, chưa
  kiểm định, chưa mở nợ; ghi để lần sau không phải nhìn lại từ đầu.
