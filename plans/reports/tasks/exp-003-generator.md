# `exp-003` / `W5-11` — Ablation bộ sinh: hai model, và một golden set không phân biệt được chúng

*2026-09-06 · `pipeline/eval/ablation_generation.py`, `pipeline/eval/refusal_calibration.py` ·
484 câu trả lời thật, 4 lượt chấm, 21 test mới · chi phí **$1,54***

---

## 0. Kết luận trước, số sau

**Giữ `deepseek-v4-flash` cho production.** Không phải vì nó thắng về chất
lượng — trên 11 metric, **10 có khoảng tin cậy chứa 0**, và điều đó đúng dưới
**cả hai** judge. Nó thắng vì thứ duy nhất đo được rõ ràng là ràng buộc đang
chặn hệ thống: **p95 4.842 ms so với 10.879 ms**.

| | `deepseek-v4-flash` | `glm-5.3-flash` |
|---|---:|---:|
| p50 / **p95** / p99 end-to-end | 2.598 / **4.842** / 11.142 ms | 4.717 / **10.879** / 14.483 ms |
| cost/query (không cache) | $0,0010701 | **$0,0004823** |
| citation accuracy (quote) | **0,8681** ✅ | 0,8504 ✅ |
| faithfulness (judge DS) | 0,9825 | 0,9630 |
| citation coverage | 0,6126 | **0,7901** ⬅ *khác biệt duy nhất có ý nghĩa* |
| misattribution | **0,0025** | 0,0206 |
| refusal accuracy | 0,9050 | 0,9174 |
| F1 bộ dò từ chối (`TD-77`) | **0,882** | 0,841 |

GLM rẻ hơn **2,2×** và chậm hơn **2,2×**. Ngân sách p95 là 3.500 ms và hệ thống
đã vượt ở 4.842; đẩy lên 10.879 là vượt **3,1×**. Khoản tiết kiệm $0,0006/câu —
ở 10.000 câu/tháng là **$6** — không mua nổi sáu giây.

**Nhưng ablation vẫn trả về hai thứ có giá trị**, và cả hai đều không phải "đổi
model": nó chứng minh bản đang chạy **không bỏ phí chất lượng** nào đo được, và
nó cho nhánh failover (`chat_fallback_provider=glm`, cấu hình từ `W4-08` nhưng
chưa bao giờ có bằng chứng) một nền tảng: cùng chất lượng, rẻ hơn, và độ trễ ít
quan trọng hơn hẳn trong lúc nhà cung cấp chính đang chết.

---

## 1. ⭐⭐ Lượt đo tìm ra một lỗi production trước khi in được con số nào

Lượt chạy nhánh A báo **4 cache hit**, và ba trong số đó nằm ở **đầu** file —
chúng không thể do lượt chạy ấy sinh ra. Tức Redis còn entry từ phiên trước. Câu
hỏi tiếp theo là câu làm hỏng cả thí nghiệm:

> Nếu tôi đổi `CHAT_PROVIDER` sang GLM, nhánh GLM có nhận lại câu trả lời của
> DeepSeek không?

```python
def cache_namespace(bundle_version: str, top_k: int) -> str:
    return f"{bundle_version}+{CHAT_SYSTEM.spec}+k{top_k}"   # ← không có model sinh
```

Có. Và bảng ablation sẽ so một model với **chính nó** trong khi mọi con số trông
hoàn toàn bình thường.

Cám dỗ là bảo `bundle_version` đã phủ rồi. Nó **không** phủ: `app.py` dựng nhánh
sinh từ `Settings` (`chat_provider`/`chat_model`), không từ bundle. Nên đây
không chỉ là một cái bẫy của thí nghiệm mà là một lỗi production thật — **một
lần đổi biến môi trường** (nâng model, đổi nhà cung cấp, sửa cấu hình failover)
vẫn phát lại câu trả lời của model cũ tới hết **TTL 24 giờ**, với
`bundle_version` không đổi và không có gì kêu.

Cùng họ với `AU-02` (`NEW-08`), ở trục thứ ba: *một khoá cache phải chứa mọi đầu
vào làm đổi câu trả lời*, và danh tính bộ sinh là đầu vào lớn nhất trong số đó.

**Vá ba lớp:**

1. `cache_namespace(bundle_version, top_k, generator)`.
2. `ChatService.generator` rỗng ⇒ **cache tắt**, không phải "dùng chung một ô".
   Thà mất cache còn hơn phát lại lời của một model khác.
3. Câu trả lời do **failover** sinh ra không được ghi vào namespace của nhánh
   chính — một sự cố năm phút không được biến thành 24 giờ phát lại lời nhà
   cung cấp dự phòng. Cùng lối xử lý với `filters` ở `AU-02`: ca hiếm, dùng
   **điều kiện loại**.

### ⭐ Và phép so của bản vá đầu tiên sai

Bản đầu viết `served_model == self.generator`. Sai: provider **phân giải bí
danh** (`deepseek-chat` → `deepseek-v4-flash`), nên hai giá trị ấy lệch nhau một
cách hoàn toàn hợp lệ ở mọi lượt bình thường — phép so ấy tắt cache oan. Tín
hiệu failover đúng là model **được yêu cầu** (`chunk.final.model_requested`),
thứ chỉ đổi khi router thật sự chuyển nhánh.

Sau khi vá, nhánh A được **chạy lại từ đầu với cache tắt** ($0,2547 bỏ đi) để
hai nhánh được đo ở đúng cùng điều kiện: 242/242 sinh thật, 0 cache hit, cả hai.

---

## 2. ⭐⭐ 10/11 metric không phân biệt được hai model — và đó là kết quả, không phải thất bại

Bảng đầy đủ ở [`runs/w511-ablation-dsjudge.md`](../runs/w511-ablation-dsjudge.md).
Bootstrap cặp 10.000 vòng, ghép theo `query_id`, cùng hàm `paired_bootstrap` mà
mọi con số đã công bố từ `W2-01` đi qua.

| metric | n | DS | GLM | hiệu | CI95 |
|---|---:|---:|---:|---:|:---:|
| `citation_coverage` | 241 | 0,6160 | 0,8414 | **+0,2255** | **[+0,1788, +0,2747]** |
| `faithfulness` | 176 | 0,9725 | 0,9696 | −0,0029 | [−0,0295, +0,0266] |
| `answer_relevancy` | 242 | 0,7686 | 0,7769 | +0,0083 | [−0,0207, +0,0372] |
| `citation_validity` | 211 | 0,8749 | 0,8455 | −0,0294 | [−0,0690, +0,0100] |
| `misattribution` | 176 | 0,0057 | 0,0179 | +0,0122 | [−0,0053, +0,0281] |
| `refusal_accuracy` | 242 | 0,9050 | 0,9174 | +0,0124 | [−0,0124, +0,0372] |
| `uncited_grounding` | **28** | 0,7982 | 0,7024 | −0,0958 | [−0,2429, +0,0470] |

Đọc đúng: **"tập này không phân biệt được hai nhánh"**, không phải "hai hệ thống
như nhau". Với n vài trăm câu, đó là một phát biểu về **cỡ mẫu** ít nhất ngang
bằng một phát biểu về hệ thống.

Chỗ duy nhất có ý nghĩa — `citation_coverage` +0,23 nghiêng về GLM — cũng không
đọc thành "GLM trích dẫn tốt hơn": GLM trích **nhiều hơn** (468 citation vs 379)
nhưng tỉ lệ hợp lệ **thấp hơn** (0,850 vs 0,868) và misattribution **cao hơn 8×**
(0,0206 vs 0,0025). Trích nhiều mà sai nhiều hơn không phải là một cải thiện, nó
là một đánh đổi — và hai vế của đánh đổi ấy đều không đạt ngưỡng ý nghĩa.

### ⚠️ Mẫu ghép cặp co lại, và con số ấy phải được in ra

`uncited_grounding` chỉ còn **28/242** câu chung: hai model sinh ra số mệnh đề
và số citation khác nhau, nên tập câu **có gì để chấm** khác nhau (DS 91 câu,
GLM 46 câu). Một bảng in `n=242` cho mọi dòng sẽ nói dối ở đúng dòng yếu nhất,
nên `compare_branches` in cả tập lệch lẫn tập giao. Đây là giới hạn thật của
phép so ghép cặp trên metric do judge sinh, và nó không khắc phục được bằng thêm
mẫu — nó là hệ quả của việc hai hệ thống có **hình dạng đầu ra** khác nhau.

---

## 3. ⭐⭐ Self-preference bias: từ một câu cảnh báo thành một con số

DoD chỉ đòi *"ghi cảnh báo self-preference bias"*. Chạy judge kép biến nó thành
bảng 2×2 — và bảng ấy nói nhiều hơn một câu cảnh báo:

**`faithfulness` (micro)**

| | judge = DeepSeek | judge = GLM | chênh giữa hai judge |
|---|---:|---:|---:|
| câu trả lời của **DeepSeek** | 0,9825 | 0,9659 | **+0,0166** |
| câu trả lời của **GLM** | 0,9630 | 0,9676 | −0,0046 |
| **khoảng cách giữa hai model** | **+0,0195** | **−0,0017** | |

Dòng cuối là điểm chính: **khoảng cách giữa hai model đổi dấu tuỳ theo model nào
chấm.** Mỗi judge cho họ nhà mình điểm cao hơn. Và toàn bộ khoảng cách ấy
(0,0195) **nhỏ hơn** mức bất đồng giữa hai judge trên cùng một tập câu trả lời.

`misattribution` cũng đổi dấu (+0,0122 → −0,0055). `uncited_grounding` giữ dấu
nhưng độ lớn lệch **6×** (+0,1882 dưới judge DS, +0,0294 dưới judge GLM).

Điều **không** đổi: dưới cả hai judge, đúng một metric đạt ý nghĩa thống kê, và
đó là `citation_coverage` — một metric **tất định**, không do judge chấm. Kết
luận của báo cáo này đứng trên đúng phần không phụ thuộc vào ai cầm bút chấm.

> 💡 Nếu chỉ chạy một judge, tôi đã báo "DeepSeek faithful hơn GLM 0,0195" và
> con số ấy sẽ vào bundle. Nó không sai về số học — nó chỉ không tồn tại độc lập
> với người chấm.

`ablation_generation` **từ chối** so hai nhánh chấm bởi hai judge khác nhau
(`ValueError`, có test): hiệu số khi ấy mang lẫn cả phần dịch của `TD-66` và
không tách ra được sau khi đã trộn.

---

## 4. `TD-77` trả xong — bộ dò từ chối **có** suy giảm khi đổi model

`W5-07` đo F1 **0,889** cho bộ dò từ khoá trên nửa giữ ngoài, với
`deepseek-v4-flash`. `TD-77` ghi nghi vấn: cách nói *"tôi không tìm thấy"* là
một thói quen văn phong, và nó đổi theo model.

| nhánh | precision | recall | **F1** | tỉ lệ judge | tỉ lệ dò | lệch |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-flash` | 0,872 | 0,891 | **0,882** | 19,0 % | 19,4 % | **+2,2 %** |
| `glm-5.3-flash` | 0,822 | 0,860 | **0,841** | 17,8 % | 18,6 % | **+4,7 %** |

Nghi vấn đúng, mức độ vừa phải: F1 tụt 0,041 (−4,6 % tương đối) và độ chệch
**tăng gấp đôi**. Bộ dò chuyển sang model khác vẫn dùng được, nhưng con số F1
là của **một cặp (model, corpus)** — nên nó là một cột trong bảng này, không
phải một hằng số trong `W5-07`.

Dấu vẫn dương ở cả hai (báo **thừa**), tức nó không tái lập lỗi tệ nhất của bản
đầu — báo **thiếu** 24,5 %, hướng làm hệ thống trông tốt hơn thực tế.

Phép đo tốn **$0**: nhãn judge đã nằm trong cache đóng băng, và
`refusal_calibration` chạy `frozen_cache=True` nên nó **không thể** lặng lẽ biến
thành một phép đo tốn tiền.

---

## 5. Hai lỗi kiến trúc mà chính bộ test bắt được

**`pipeline` import `serving`.** `refusal_calibration.py` lấy
`looks_like_refusal` từ `serving/core/metrics.py`, và
`test_pipeline_does_not_import_serving` đỏ ngay. Bài test đúng: Pipeline Plane
phải chạy độc lập trên máy GPU thuê, nơi không có serving stack. Cách sai là
chép danh sách từ khoá sang chỗ thứ hai — khi ấy bảng Grafana và bảng hiệu chỉnh
đo **hai bộ dò khác nhau**, đúng họ lỗi với `AU-12`. Bộ dò chuyển xuống
`rag_core/generation/refusal.py`; `serving.core.metrics` xuất lại tên cũ.

**`generation_metrics` hardcode judge DeepSeek.** Tức một bảng ablation có
DeepSeek trong danh sách ứng viên sẽ do **chính một ứng viên** chấm, và không có
cờ nào để kiểm chéo. `calibration.py` đã có sẵn đoạn rẽ nhánh theo họ model từ
`W5-04`; gộp cả hai vào `judge.build_judge` thay vì chép lần thứ hai — hai bản
sao sẽ lệch, và cách chúng lệch là im lặng (`reasoning_effort` gửi sang DeepSeek
được **nhận rồi bỏ qua**, đo ở `W3-04`).

---

## 6. Tiêm lỗi — 19 phép, và 5 phép sống sót đều đáng giá

Lượt một **14/19 đỏ**. Năm phép sống sót chia làm hai loại, cả hai đều là phát
hiện thật:

**Hai điều kiện chết trong mã** (`M2`, `M8`) — chúng không bao giờ đổi được kết
quả:
* `and self.generator` ở **đầu ghi** cache: điều kiện failover ngay dưới đã bao
  nó (`requested_model` không bao giờ rỗng, nên `generator=""` tự chặn).
* nhánh `if settings.chat_provider == "none"` trong `primary_generator`: `"none"`
  không có trong bảng model, nên `.get` đã trả `""`.

Cả hai bị **xoá**. Dự án này đã học đúng bài ấy một lần ở `W4-06`, ghi ngay
trong `chat.py`: *một điều kiện không thể thay đổi hành vi là một chú thích viết
bằng cú pháp `if`*.

**Ba lỗ trong test** (`M6`, `M17`, và `M3`):
* `M17` — `f1` đổi thành `precision` **sống sót** vì bài test dựng ví dụ có
  P = R = 0,5 = F1. Một bài test chọn ví dụ cân đối không phân biệt được ba công
  thức. Dựng lại với P = 2/3, R = 0,4, F1 = 0,5.
* `M6` — xoá `requested_model = chunk.final.model_requested` sống sót vì bài test
  failover đổi `service.generator`, nên giá trị **khởi tạo** đã tự chặn. Tức bài
  test canh được một điều kiện nhưng mù với chính đường mà failover đi: chỉ
  **chunk cuối** biết nhánh dự phòng đã trả lời. Dựng lại cảnh đúng.
* `M3` — **sống sót có chủ đích**, và ghi lý do ngay tại chỗ: `and self.generator`
  ở **đầu đọc** là một hàng rào **hiệu năng**, không phải hàng rào đúng đắn. Bỏ
  nó không sinh ra câu trả lời sai — namespace nó đọc (`…+g`) là namespace mà
  không đường ghi nào chạm tới được — chỉ sinh ra một lần embed + một lượt Redis
  mỗi lượt chat, vĩnh viễn miss. Giữ lại và nói rõ, thay vì thêm một bài test
  service-level nặng để canh một thứ không thể sai.

Sau khi bịt: tiêm lại `M6`/`M17` → **cả hai đỏ**.

---

## 7. Cái **không** làm, và vì sao

**Nhánh Qwen3-8B (vLLM) và nhánh OpenRouter không chạy.** Bạn chốt phạm vi hai
nhánh API trước khi bắt đầu:

* **OpenRouter** — không có `OPENROUTER_API_KEY`. GLM thay chỗ một cách trung
  thực: key có sẵn, giá đã ghim trong `GLM_PRICING`, hành vi suy luận đã đo
  (`reasoning_effort=low` cho 0 token suy luận). Nó **không** phải một
  "OpenRouter pinned slug", và báo cáo không giả vờ là vậy.
* **vLLM/Qwen3** — cần GPU thuê. `TD-30` giữ nguyên mức, với ba ẩn số chưa đo
  y như cũ. `W0-05` (dựng RunPod) vẫn hoãn.

Hệ quả cho `G6`: câu *"chọn model nào cho production và vì sao"* trả lời được,
nhưng nó là lựa chọn giữa **hai** nhà cung cấp API, không phải giữa API và
self-hosted. Câu hỏi *"tự host có rẻ hơn không"* vẫn chưa có số đo.

`TD-83` (smoke tầng sinh trong nightly) **cũng chưa trả**: máy kiểm định giờ đã
có (`ablation_generation`), nhưng nối nó vào `nightly.yml` cần một nhánh model
thứ hai chạy được **trong CI**, mà cả hai nhánh hôm nay đều cần secret. Chuyển
sang `W6-06` cùng phần quản lý secret.

---

## 8. Số đo có thẩm quyền — thay hai con số cũ

| | cũ | **mới** | vì sao mới đúng hơn |
|---|---|---|---|
| p95 end-to-end | 4.706 ms (`W5-05`) | **4.842 ms** | cùng 242 request, nhưng **cache tắt** — con số cũ có 4 lượt trúng cache kéo xuống |
| cost/query | $0,0016828 (`W5-07`) | **$0,0010701** | 242 lượt thay vì 9, và không lượt nào trúng cache |

Cả hai đều là số của `deepseek-v4-flash`. `p95` vẫn **trượt** ngân sách 3.500 ms
(138 %), và giờ đã biết đòn bẩy **không** nằm ở việc đổi model: nhánh rẻ hơn còn
chậm hơn 2,2×. Chỗ còn lại là `DEFAULT_RERANK_CANDIDATES` (rerank chiếm 92,8 %
ngân sách truy hồi, `W5-06`) → `W6-05`.

## 9. Chi phí

| khoản | USD |
|---|---:|
| sinh 242 câu × DeepSeek | 0,2590 |
| sinh 242 câu × GLM | 0,1167 |
| judge DeepSeek × 2 nhánh | 0,5206 |
| judge GLM × 2 nhánh | 0,3858 |
| ⚠️ lượt DeepSeek đầu (có cache) — **bỏ đi** | 0,2547 |
| smoke 2 × 3 câu | 0,0058 |
| **Tổng** | **1,5426** |

$0,2547 bỏ đi là giá của việc phát hiện lỗi cache **sau** khi đã chạy một nhánh
chứ không phải trước. Rẻ hơn nhiều so với giá của việc không phát hiện.

## 10. Nợ

**Trả xong**: `TD-77` (F1 bộ dò theo từng model, và nó **có** suy giảm).

**Giữ nguyên mức**: `TD-30` (đường vLLM chưa chạy lần nào) · `TD-70` (bí danh
generator trong `0.1.0`/`0.2.0`) · `TD-66` (judge identity — giờ có thêm số đo
self-preference).

**Chuyển chỗ trả**: `TD-83` → `W6-06`.

**Mới**: `TD-86` — `citation_coverage` là metric duy nhất phân biệt được hai
model, và nó thưởng cho việc **trích nhiều** mà không phạt việc trích sai. GLM
+0,23 coverage đi kèm misattribution ×8. Cần một metric hợp nhất (coverage ×
validity) trước khi ai đó tối ưu vào đúng cái ô này.
