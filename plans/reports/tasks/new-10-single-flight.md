# `NEW-10` — single-flight: 8 lời gọi trả tiền thành 1, và một cái giá phải nói ra

*2026-09-08 · `serving/core/single_flight.py` + 3 mối nối · **34 test mới** · tiêm **17/18 đỏ** · chi phí **$0***

> **Nợ:** *"Single-flight cho lượt semantic cache TRƯỢT (`AU-11`). N request
> trùng nhau đồng thời = N lời gọi trả tiền + N lượt rerank trên đúng tài
> nguyên là trần của hệ thống."*

---

## 0. Đo được, trên hệ đang phục vụ

Cùng probe, cùng máy, cùng stub hiệu chỉnh theo DeepSeek (`W6-05`) — nên hai
cột so được trực tiếp:

| | trước (`W6-05`) | sau | |
|---|---:|---:|---|
| **lời gọi provider** | **8** | **1** | 8× |
| đỉnh lượt chồng nhau ở provider | 6 | **1** | |
| người theo sau được phục vụ | 0 | **7** | |
| đồng hồ tường cả đợt | 9.010,8 ms | **4.811,8 ms** | 1,87× |
| `total_ms` p50 | 6.745 | **4.802** | 1,41× |
| `total_ms` max | 9.006 | **4.812** | 1,87× |
| **`ttft_ms` p50** | **3.808** | **4.801** | ⚠️ **xấu đi 1,26×** |
| `ttft_ms` max | 6.074 | **4.811** | 1,26× |

`probes/new10-singleflight-after.json` · lệnh:
`python -m loadtest.singleflight_probe --n 8`

### ⚠️ Cái giá: TTFT p50 xấu đi, và nó là **hệ quả trực tiếp của thiết kế**

Người theo sau nhận câu trả lời khi leader **xong hẳn**, nên TTFT của họ bằng
`total_ms` của leader. Không có single-flight thì request đầu tiên bắt đầu chảy
chữ sau 1,5–3,5 s vì nó không xếp hàng sau ai; có single-flight thì bảy người
kia không thấy ký tự nào cho tới giây thứ 4,8.

Đổi lại: **đuôi co lại** (max 6.074 → 4.811 ms), **tổng thời gian cả đợt gần
gấp đôi tốc độ**, và hoá đơn chia 8. Với một trang demo công khai — nơi hình
dạng thật của `AU-11` là *reload trang* và *bấm gửi hai lần* — đó là đánh đổi
đúng chiều.

⭐ Có một thiết kế **không** phải trả giá ấy: fan-out từng mẩu của leader sang
follower theo thời gian thực. Nó đòi phát một async generator cho N người tiêu
thụ, mỗi người có thể ngắt kết nối bất kỳ lúc nào — một cơ chế đồng bộ **thứ
hai** trên đường request nóng. Cố ý không làm; ghi lại ở §6 kèm con số vừa đo,
để lần sau ai mở lại thì mở bằng dữ liệu chứ không bằng cảm giác.

---

## 1. Năm quyết định, và cái thứ nhất là cái dễ làm sai nhất

1. **⭐⭐ Khoá phải chính xác, không được mờ.** Semantic cache khớp cosine ≥ 0,96
   — và nó *được phép* mờ vì mỗi hit khai `matched_question` + `similarity` ra
   khung `meta`, người đọc thấy được. Single-flight **không có** bước ấy:
   follower nhận thẳng câu trả lời như thể của mình. Nên khoá là
   `(tenant, cache_namespace, câu hỏi nguyên văn)`. Hai *paraphrase* đồng thời
   **không** được gộp — chấp nhận, vì `AU-11` đo trên tám câu **giống hệt**.
2. **⭐⭐ Follower nhận câu trả lời từ bộ nhớ, không qua Redis.** Đường ghi cache
   là task nền (`_PENDING`), nên một follower thức dậy rồi `lookup()` lại sẽ
   **đua với chính đường ghi ấy** và thường thua.
3. **⭐⭐ Leader hỏng ⇒ follower không cùng chết.** `resolve(None)` = *"tự đi mà
   làm"*. Tệ nhất là N request đầy đủ — **đúng bằng hôm nay**.
4. **Hạn giờ 15 s** (> p99 11.142 ms của `exp-003`). Nó **không** phải van an
   toàn rẻ tiền — quá hạn nghĩa là follower đã tiêu 15 s *rồi mới* bắt đầu tự
   làm. Nó là hàng rào cho ca leader **biến mất** mà không chạy `finally`.
5. **⚠️ Trong tiến trình, và nói ra.** Cùng lời khai với `CostBudget` (`W4-08`)
   và hạn mức nhịp (`TD-39`): 4 replica ⇒ 4 lượt sinh. Không dựng khoá phân tán
   Redis cho một triển khai chưa tồn tại — và `NEW-09` (08/09/2026) vừa chứng
   minh lối ra multi-container còn chưa tới.

---

## 2. ⭐⭐ Bài test tìm ra một lỗ mà tôi vừa tự tay tạo

`resolve()` sống trong `finally` của `stream_turn`. Nhưng `try` lớn ấy bắt đầu
**sau** hai khung đầu (`meta`, `sources`) — nên một client ngắt kết nối ở đúng
khoảng ấy làm `finally` không bao giờ chạy, và **bảy người theo sau chờ đủ 15
giây cho một request đã chết từ mili giây thứ nhất**.

Tôi không suy ra điều đó. Tôi viết
`test_a_client_that_disconnects_mid_stream_still_releases_the_follower` vì
"ba đường thoát" là danh sách phải kiểm, và nó **đỏ ngay lần đầu**.

Vá bằng một lớp bọc mỏng quanh `_stream_turn`. Và lớp bọc ấy **làm đỏ hai bài
huỷ có sẵn từ `W4-06`**: đóng generator ngoài **không** đóng generator trong
một cách dứt khoát (nó chờ GC), nên `_schedule_save` và `trace.finish` chạy
muộn hoặc không chạy. `contextlib.aclosing` đóng lại. Hai lỗi liên tiếp trong
cùng một bản vá, cả hai do test có sẵn hoặc test mới bắt — không cái nào do
đọc lại mã.

⚠️ Còn một cửa sổ thứ hai, cùng hình dạng, ở **khúc khác**: vé được nhận trong
`_prepare` nhưng chỉ giải phóng ở `stream_turn`. Nếu truy hồi ném lỗi ở giữa,
`ChatTurn` không bao giờ ra đời ⇒ future nằm lại trong sổ **và không bao giờ
xong** ⇒ khoá bị **đầu độc vĩnh viễn**, mọi lượt hỏi cùng câu về sau đều thành
follower và đều chờ hết hạn giờ. Đó là chế độ hỏng tệ nhất của cả hạng mục, và
nó **không** ở chỗ dễ nhìn.

---

## 3. Tiêm lỗi: 17/18 đỏ, và ba lượt chấm mới đủ

Lượt một **8/14**. Ba phép sống sót, và hai trong ba là **mã chết của tôi**:

| phép sống | hoá ra |
|---|---|
| `_done` bỏ đi vẫn xanh | Cờ ấy **thừa** — `future.done()` mới là chốt thật. Đã xoá. |
| `_retire` dùng `pop()` trần vẫn xanh | Phép so `is` **không thể** cứu ca nào. Đã xoá, cùng với **một kịch bản tôi bịa ra để biện minh cho nó** (§4). |
| `_prepare` bỏ đường cứu vé vẫn xanh | Lỗ test thật — không bài nào nhìn tới cửa sổ ở §2. Đã thêm. |

Lượt hai **12/15**. Ba phép sống nữa:

| phép sống | hoá ra |
|---|---|
| đổi `\x00` thành `:` | ⭐⭐ Bài test của tôi **không thể đỏ**: nó so hai khoá vốn khác nhau dưới *mọi* dấu ngăn cách. Luật thật là **không nhập nhằng**, và kiểm nó cần đúng một cặp **va nhau**: với `:` thì `("a","b:c","q")` và `("a","b","c:q")` cho cùng một chuỗi. |
| `if ticket.is_leader:` → `if True:` (ai cũng leader) | ⭐⭐ **Mệnh đề trung tâm của hạng mục chưa có bài test nào** — mọi bài trước dựng vé **bằng tay** thay vì đi qua `_prepare`. Thêm bài đo thứ `AU-11` đo: **số lượt truy hồi**. |
| khoá bỏ `cache_namespace` | ⭐ `AU-02` ở trục thứ ba: cùng câu hỏi khác `top_k` là **hai** câu hỏi. Thêm bài. |

Lượt ba **17/18**. Phép còn lại là **mutant tương đương**, không phải lỗ test:

> **`M8`** — đảo thứ tự `_retire` / `set_result`. `resolve()` đồng bộ từ đầu tới
> cuối, nên **không coroutine nào chen vào giữa hai lệnh**; không quan sát viên
> nào phân biệt được hai thứ tự. Giữ thứ tự hiện tại vì nó đọc ra bất biến rõ
> hơn, và **không** viết bài test canh nó — bài ấy sẽ canh một thứ không quan
> sát được.

---

## 4. ⭐⭐ Tôi viết một hàng rào, rồi bịa một câu chuyện để biện minh cho nó

Bản đầu của `_retire`:

```python
if self._inflight.get(key) is future:      # "phép so `is` không thừa"
    del self._inflight[key]
```

kèm docstring giải thích: *"Leader A hết hạn, leader B nhận cùng khoá, rồi
`finally` của A mới chạy — một `pop(key)` trần sẽ xoá vé của B."*

Nghe rất hợp lý. **Phép tiêm bác bỏ nó**: thay bằng `pop()` trần thì không bài
nào đỏ. Truy ra thì kịch bản ấy **không thể xảy ra** — `join()` chỉ cấp vé
leader khi future đang giữ khoá là `None` hoặc `done()`, mà `resolve()` đồng bộ
nên không ai quan sát được trạng thái trung gian; suy ra một khoá **không bao
giờ** có hai leader sống cùng lúc.

Hai dòng mã chết, cộng một câu chuyện làm chúng trông như đã có người nghĩ tới
ca ấy. Đó nguy hiểm hơn mã chết trần: người sửa sau sẽ **tin** vào lời giải
thích. Cùng bài học `W5-11`/`M2` — *một điều kiện không đổi được hành vi là một
chú thích viết bằng cú pháp `if`* — nhưng ở dạng tệ hơn một bậc.

Bộ test giờ ghim **bất biến** (`TestMotKhoaKhongBaoGIOCoHAILeaderSong`) thay vì
ghim hàng rào.

---

## 5. ⭐ Bảng sẽ nói dối nếu không tách nhãn

Follower đi qua **cùng** nhánh phát lại của `W4-10`, nên span `cache.replay` —
thứ `MetricsSink` dùng để đếm *"phục vụ mà không gọi provider"* — quy **cả**
công của single-flight cho semantic cache. Con số không sai; **cái tên** nói dối
về cơ chế, và bảng RAG Health sẽ báo tỉ lệ trúng cache tăng vọt sau một bản vá
**không đụng gì tới cache**.

Tách: `result="single_flight"` vs `result="replay"`, và follower vẫn đếm
`cache.lookup → miss` như đúng sự thật (nó *đã* trượt cache). Cùng lý lẽ với
`refusals_suspected` của `W5-07`: đặt tên đúng **thứ đang đếm**.

---

## 6. Việc sinh ra từ lượt này

* ~~`NEW-10`~~ **đóng**. `AU-11` đóng cùng: 8 → 1 lời gọi, đo trên hệ thật.
* 💡 **Fan-out theo thời gian thực** — chưa mở nợ, có số: nó xoá được khoản
  TTFT p50 +26% ở §0, giá là một cơ chế đồng bộ **thứ hai** trên đường nóng
  (phát một generator cho N người tiêu thụ ngắt kết nối độc lập). Mở lại khi
  TTFT trở thành SLO thật (`G2` còn ⏳ chờ chốt) — không phải trước đó.
* ⚠️ **Trần vẫn theo tiến trình.** Khi `TD-63` đi đường (b) — thêm container —
  thì single-flight, `CostBudget` và hạn mức nhịp **cùng lúc** mất hiệu lực
  theo tỉ lệ số replica. Ba nợ một chỗ trả, và chỗ ấy là Redis.
