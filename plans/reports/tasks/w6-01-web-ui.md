# `W6-01` — Web UI, và một lỗi cache tìm ra trong lúc chụp ảnh màn hình

> Ngày 06/09/2026 · chi phí **~$0,002** (một lượt sinh thật để chụp ảnh) ·
> `serving/ui/index.html` · `tests/e2e/test_ui_smoke.py` ·
> ảnh: `plans/reports/probes/w601-ui-{1,2,3}-*.png`

DoD: *streaming · citation click → highlight chunk gốc · feedback 👍/👎 · upload
progress* · **người lạ dùng được không cần hướng dẫn** · Test: Playwright ·
Evidence: GIF.

---

## 0. Ba câu trả lời

1. **Giao diện chạy**: một tệp tĩnh, không bước build, phục vụ ở `GET /`.
   Streaming, bấm `[n]` mở đúng nguồn và **tô đoạn được trích**, 👍/👎 ghi được.
   13/13 bài Playwright xanh, và 8 phép tiêm vào chính trang để chứng minh 13
   bài ấy đo được cái gì.
2. **`AU-10`/upload**: DoD viết "upload progress" nhưng hệ thống **không có
   upload**, và thêm nó vào sẽ phá đúng quy tắc cứng về license corpus. Cái dựng
   được một cách trung thực là **tiến độ job ingest**, đi qua một proxy có auth.
3. **Một lỗi production tìm ra trong lúc chụp ảnh**: server trỏ vào DeepSeek
   **thật** phát lại nguyên văn câu trả lời do stub của `W6-05` sinh ra — và
   khung `done` khai `model: "deepseek-v4-flash"`. Đã vá.

---

## 1. ⭐⭐ Trang này dựng DOM từ hai nguồn không tin được

Đó là câu quyết định gần như mọi lựa chọn kỹ thuật bên dưới.

* **Văn bản chunk corpus.** Chính khung `sources` gắn cờ `flags` cho nó
  (`W4-12`) — tức máy chủ đã nói thẳng *"cái này có thể chứa payload"*.
* **Câu trả lời của model**, thứ được sinh **ra từ** nội dung ấy.

Nên luật số một của `serving/ui/index.html` là **không bao giờ `innerHTML`**.
Mọi văn bản đi qua `createTextNode` / `textContent`; bộ dựng markdown cũng tạo
phần tử bằng `createElement` chứ không ghép chuỗi HTML — nghĩa là kể cả khi luật
parse có lỗi, kết quả tệ nhất là **hiển thị xấu**, không phải thực thi mã.

### Ba hàng rào, và chúng hỏng theo ba cách khác nhau

| hàng rào | ai cưỡng chế | hỏng khi nào |
|---|---|---|
| Không `innerHTML` trong mã | người viết | lần sửa thứ ba, khi không ai nhớ luật |
| `tests/unit/test_ui.py` quét mã | CI | không — nó đỏ ngay dòng vi phạm đầu tiên |
| CSP (`default-src 'none'`) | trình duyệt | chỉ khi trình duyệt bỏ qua CSP |

Bài test grep thô, nhưng nó chạy trong 3 ms và không cần trình duyệt. ⚠️ Nó phải
bóc chú thích trước khi quét: docstring của chính trang **nhắc tên** những API bị
cấm để giải thích vì sao chúng bị cấm, và một phép grep thô sẽ đỏ vì đúng đoạn
văn nói rằng chúng không được dùng.

⭐ Bài test âm ấy đi kèm một bài **dương** (`createTextNode` và `createElement`
phải có mặt, `textContent` xuất hiện > 10 lần). Không có nó thì "xoá hết
JavaScript" cũng là một cách làm bài test xanh.

### ⭐ Không markdown đầy đủ, và đó là một quyết định chứ không phải lười

Câu trả lời thật có `**đậm**`, tiêu đề, gạch đầu dòng — bỏ hết thì trang khó
đọc. Một bộ render markdown đầy đủ thì cần một **sanitiser**, và một sanitiser là
một phụ thuộc bên ngoài nằm đúng trên đường đi của nội dung không tin được.

Tập con viết tay (tiêu đề, đậm, bullet, `[n]`) **không parse HTML** nên không có
gì để khử. Không ảnh, không link, không HTML thô.

---

## 2. ⭐⭐ Bấm `[n]` → tô đoạn được trích: nó cần một trường mới trong SSE

Khung `sources` cũ chở `chunk_id`, `title`, `source_url`, `score`, `flags` —
**không có nội dung**. Với chừng đó, UI chỉ hiện được *tiêu đề* nguồn, tức người
đọc vẫn phải **tin** lời model rằng quote có thật. Đó đúng là thứ `W4-09` sinh ra
để không phải tin.

Nên `ChatTurn.sources()` mang thêm `content`. Không phải một khoản lộ mới: chunk
ấy **đã** đi tới model trong prompt của chính người dùng đang hỏi, và
`tenant_filter()` đã chạy trước đó. Thứ thêm vào là *ai nhìn thấy nó* — client
hỏi, thay vì chỉ nhà cung cấp LLM.

### ⭐⭐ Nhưng hàng Postgres thì **không** được mang nó

Cùng một danh sách nguồn phục vụ hai mục đích khác nhau, nên nó phải là **hai
payload khác nhau**:

| | khung SSE | hàng Postgres |
|---|---|---|
| mục đích | để người đọc kiểm được quote | lịch sử hội thoại, và **ứng viên golden set** (`W5-08`) |
| `content` | có | **không** |
| kích thước mỗi lượt | ~8 KB | ~1 KB |

Không lọc thì hàng lịch sử thành **bản sao thứ hai của index**, và nó đi tiếp
vào file ứng viên golden set. `persisted_sources()` lọc đúng một khoá, và có một
bài test ghim rằng nó lọc **đúng một khoá** — một bản vá cắt nhầm `chunk_id` sẽ
tái lập chính lỗi mà `W5-08` vừa đóng.

### ⭐⭐ Trang có thể **không tìm thấy** chỗ tô, và nó không được phép nói gì thêm

Máy chủ đối chiếu quote bằng `_quote_matches`: tách theo dấu lược (`...`, `…`,
`[...]`), mọi mảnh phải khớp nguyên văn **đúng thứ tự, không chồng lấn**. Trang
chỉ tìm chuỗi con sau khi chuẩn hoá whitespace.

Hai luật khác nhau ⇒ có những quote **hợp lệ** mà trang không định vị được. Cám
dỗ là cho trang tự quyết. Nó không được:

* phán quyết `verified` đến từ **máy chủ** (`W4-09`), trang chỉ vẽ lại nó;
* không tô được thì trang **in nguyên văn quote** ra cạnh nguồn, không im lặng.

Bài Playwright ghim đúng lựa chọn ấy: `mark` **hoặc** `.quote` phải có ít nhất
một cái. Một bài test chỉ đòi `mark` sẽ ép người sửa sau này đi chép
`_quote_matches` sang JavaScript — tức tạo ra bản thứ hai của một phép kiểm bảo
mật, đúng họ lỗi `AU-12`.

---

## 3. ⭐⭐ Lỗi tìm ra trong lúc chụp ảnh màn hình

Ảnh đầu chụp với stub của `W6-05` (rẻ, tất định). Ảnh sau muốn chụp câu trả lời
**thật**, nên đổi server sang DeepSeek. Và server thật trả về:

```
event: done
data: {"finish_reason": "cache", "model": "deepseek-v4-flash", ...}
```

kèm nguyên văn đoạn text do **stub** sinh ra (`"Theo nguồn [1], nội dung liên
quan được nêu trong tài liệu."` lặp 15 lần). Khung `done` gọi tên một model chưa
từng viết đoạn ấy.

`W5-11` đã đưa `provider:model` vào khoá cache. `DEEPSEEK_BASE_URL` thì không
nằm trong cả hai. **Cùng một cặp provider+slug trỏ vào hai máy chủ khác nhau là
hai bộ sinh khác nhau** — và với một vLLM tự dựng thì slug còn do người dựng tự
đặt, nên hai hệ thống hoàn toàn khác nhau có thể khai cùng một chuỗi.

Hẹp hơn `W5-11` ở production (base URL ít khi đổi), nhưng luật không đổi: *một
khoá cache phải chứa mọi đầu vào làm đổi câu trả lời*. Endpoint là một trong số
đó, và nó là trục **thứ tư** sau `bundle` (`W4-10`), `prompt` (`W4-11`),
`top_k` (`AU-02`), `generator` (`W5-11`).

⚠️ **Để ngoài `generator`, không nhét vào nó.** `generator` còn là tín hiệu
failover (`requested_model == generator.split(":", 1)[-1]`), và một URL có dấu
`:` sẽ làm phép tách ấy trả về `"8199"`. Đó đúng là lỗi mà **bản vá đầu của
`W5-11` đã mắc một lần** — chỉ khác chỗ hỏng. Hai thứ khác nhau ⇒ hai tham số
khác nhau, không phải một chuỗi khéo léo.

---

## 4. ⭐ "Upload progress": DoD hỏi một thứ hệ thống cố ý không có

`pipeline/ingest/schemas.py:IngestRequest` nhận một **tên config**, và docstring
ở đó nói vì sao nó không nhận đường dẫn: nhận đường dẫn thì `../../.env` hay một
YAML bất kỳ trên đĩa đều đi qua được. **Không có endpoint nào nhận file.**

Và đó không phải một thiếu sót cần vá ở đây. Quy tắc cứng của dự án là *corpus
phải công khai, license cho phép redistribute*, và `pipeline/corpus/` cưỡng chế
nó bằng manifest + license + DVC. Một nút "tải tài liệu lên" mở đúng con đường
mà luật ấy sinh ra để đóng: tài liệu không rõ nguồn → index → prompt → một câu
trả lời **có trích dẫn**.

Nên phần dựng được: **tiến độ job ingest**. Chạy lại một config đã có, xem
`documents_done / documents_total` và `chunks_embedded` chạy. Phần "upload" cần
một quyết định về license mà không hạng mục nào của `W6` đã lấy → ghi vào sổ,
không tự quyết.

### ⭐ Proxy, không phải gọi thẳng cổng 8001

`AU-10`: dịch vụ ingest **không có auth**, biện pháp giảm nhẹ hiện tại là bind
`127.0.0.1`. Cho trình duyệt gọi thẳng sẽ (a) buộc mở CORS trên một dịch vụ
không xác thực, (b) buộc nó rời khỏi loopback. Đi vòng qua `/admin/ingest` thì
tầng auth của `W4-04` áp dụng nguyên vẹn — `ADMIN_PREFIX` che theo **tiền tố
đường dẫn**, nên route này được bảo vệ vì nó *ở trong* `/admin`.

⚠️ Mặc định **tắt** (`INGEST_API_URL` rỗng ⇒ 503 kèm lời giải thích). Một bề
mặt điều khiển pipeline mở sẵn ở mọi lần deploy là thứ không ai xin. Và 503 phải
**nói ra** là đang tắt: một 503 câm sẽ bị đọc là "dịch vụ chết" và ai đó sẽ đi
khởi động lại một thứ đang khoẻ.

Proxy cũng không dội thân lỗi upstream ra client (`AU-03`, `NEW-08`): mã trạng
thái là thông tin **cần** (404 = job không tồn tại, khác hẳn 502), thân lỗi thì
mang host, cổng, đường dẫn đĩa.

---

## 5. Những chỗ giao diện phải nói ra sự thật khó chịu

| tình huống | trang làm gì | vì sao |
|---|---|---|
| câu trả lời rỗng (`finish_reason="empty"`) | **tắt** nút 👍/👎, ghi "(không chấm được lượt rỗng — TD-78)" | `_save()` bỏ qua text rỗng ⇒ `answer_message_id` trỏ vào hàng không tồn tại ⇒ `POST /feedback` trả 404, ở đúng lượt đáng nhận 👎 nhất |
| luồng đứt trước khung `done` | báo đỏ "câu trả lời có thể thiếu" | docstring `POST /chat`: một dòng `delta` dừng lại **giống hệt nhau** khi model nói xong, khi kết nối đứt, và khi provider hết hạn mức |
| chunk bị gắn cờ nghi tiêm | phù hiệu vàng ngay trên nguồn | `W4-12` cho cờ ra tới client chứ không chỉ vào log: người đọc là người duy nhất biết nó có bất thường hay không |
| citation không xác minh được | `[n]` viền đỏ + phù hiệu `x/y trích dẫn xác minh được` | `W4-09` là phép kiểm duy nhất chỉ báo được bằng một khung SSE |
| trả từ cache | phù hiệu "trả từ cache" | một câu trả lời 33 ms và một câu 6 giây là hai chuyện khác nhau |
| khác ngôn ngữ câu hỏi | phù hiệu vàng | `W4-07` đo được tỉ lệ nền là *tất cả*, không phải *thỉnh thoảng* |

---

## 6. `tests/e2e/test_ui_smoke.py` — và phép kiểm cho chính nó

10 bài Playwright trên Chromium thật, chạy với stub của `W6-05` (không tốn tiền,
tất định — `TD-41`: `temp=0` ở DeepSeek không tất định).

Một bộ e2e dễ trông như đang kiểm mọi thứ trong khi nó chỉ kiểm rằng trang tải
được. Nên **8 phép tiêm vào chính HTML/JS**, chấm bằng bộ Playwright ấy: xoá
`onclick` của `[n]`, làm regex `[n]` không khớp, `d.open = false`, bỏ nhánh in
quote nguyên văn, để nội dung nguồn rỗng, gửi `message_id` sai, in câu trả lời
thô, bỏ dòng khai bundle. *(Kết quả ở §8.)*

⚠️ Mỗi phép tiêm cần **khởi động lại server**: `serving/api/ui.py` đọc trang một
lần lúc import. Đó là lựa chọn đúng cho production (không I/O đồng bộ trên vòng
lặp sự kiện) và nó làm lượt tiêm chậm — chấp nhận được cho một thứ chạy tay.

---

## 7. ⚠️ Một va chạm cổng do `W6-05` để lại

`tests/integration/test_chat_stream.py` cấp cổng cho các tiến trình uvicorn của
nó từ dải **8091–8119**. Stub của `W6-05` mặc định **8099**. Một stub đang chạy
làm đúng một bài trong dải ấy đỏ, với thông báo `"uvicorn chết lúc khởi động"` —
không nhắc gì tới cổng.

Mất một lượt chẩn đoán (và một lần suýt đổ lỗi cho thay đổi của chính hạng mục
này). Stub dời sang **8199**, kèm chú thích nói vì sao.

---

## 8. Tiêm lỗi

**Python: 15/15 đỏ** (lượt một 13/15). Hai phép sống sót, cả hai là lỗ test thật:

* **M5** — xoá `"/"` khỏi `PUBLIC_PATHS`. Sống sót vì fixture của
  `tests/unit/test_ui.py` dựng một FastAPI **trần không có `AuthMiddleware`**:
  nó kiểm được *nội dung* trang, không kiểm được *ai vào được* trang. Bịt bằng
  một bài trên app thật, cộng một bài ngược lại (trang mở **không** kéo theo
  endpoint dữ liệu nào).
* **M15** — `primary_endpoint` luôn trả URL của DeepSeek bất kể nhà cung cấp.
  Sống sót vì hàm ấy **chưa có bài test nào** — nó vừa được thêm ở §3 và tôi đã
  kiểm nó bằng mắt trên hệ chạy thay vì bằng một bài test.

**HTML/JS: 8/8 đỏ** (lượt một 5/8). Ba phép sống sót, và cả ba nói một điều
khác nhau về bộ e2e:

* **U4** — xoá nhánh "không tô được thì in nguyên văn". Sống sót vì với stub,
  quote **luôn** nguyên văn nên trang luôn tô được: nhánh dự phòng chưa từng
  chạy trong bộ test. Bịt bằng một bài gắn thẳng một quote không có trong nguồn
  (`page.evaluate`) rồi kiểm rằng nó được in ra.
* **U5** — để nội dung nguồn rỗng. Sống sót vì **nhánh dự phòng cứu nó**: không
  tô được thì in quote, và bài test cũ chấp nhận cả hai. Đúng về mặt logic, sai
  về mặt sản phẩm — một nguồn rỗng là một nguồn không kiểm được. Bịt bằng một
  bài đòi thân nguồn có nội dung thật.
* **U7** — in câu trả lời thô ở nhánh `delta`. Sống sót vì nhánh `citations`
  **gọi lại** `renderMarkdown` ở cuối, nên trạng thái sau cùng giống hệt. Thứ
  khác nhau là quãng ở giữa: với câu trả lời 6 giây, người dùng hoặc thấy `[1]`
  bấm được ngay khi nó hiện ra, hoặc nhìn text trơ suốt sáu giây. *Một mutation
  chỉ đổi trạng thái tạm thời vẫn đáng giết: quãng tạm thời ấy là toàn bộ trải
  nghiệm của một API stream.*

⚠️ **Và một lượt tiêm giả.** Lần chấm lại đầu tiên cho **0/3 đỏ** — cả ba phép
tiêm "sống sót" trong khi hai bài test mới rõ ràng phải bắt được chúng. Nguyên
nhân: một server **cũ** vẫn giữ cổng 8000, nên hàm `_serve()` của bộ tiêm thấy
`/ready` xanh, trả về vui vẻ, và cả lượt chấm chạy trên trang **chưa bị tiêm**.

Cùng họ với "đỏ giả" của `NEW-08`, chỉ ngược chiều — và nguy hiểm hơn, vì một
lượt tiêm toàn màu xanh trông giống hệt một lượt tiêm không tìm ra gì. Bộ tiêm
giờ **từ chối chạy** khi cổng đã có người.

---

## 9. DoD: đạt gì, chưa đạt gì

| yêu cầu | trạng thái |
|---|---|
| streaming | ✅ |
| citation click → highlight chunk gốc | ✅ (tô, hoặc in nguyên văn khi không định vị được — §2) |
| feedback 👍/👎 | ✅, kèm nhánh `TD-78` |
| upload progress | 🟡 **tiến độ ingest**; *upload* không tồn tại và không nên tồn tại — §4 |
| người lạ dùng được không cần hướng dẫn | 🟡 **có điều kiện**: người lạ **có khoá** dùng được không cần hướng dẫn. Người lạ không có khoá thấy một bảng giải thích cần gì. API này đòi khoá ở mọi endpoint dữ liệu, nên "không cần gì cả" là việc của `W6-02` (demo công khai có rate limit) |
| Test: Playwright | ✅ 13 bài, và **8 phép tiêm vào chính trang** để chứng minh 13 bài ấy đo được cái gì |
| Evidence: GIF | 🟡 **ba ảnh chụp** thay vì GIF: `w601-ui-1-empty.png`, `-2-answer-real.png`, `-3-citation-real.png`. Một GIF cần `ffmpeg` — thêm một phụ thuộc hệ thống chỉ để làm tài liệu. Ảnh chụp từ lượt **thật** (DeepSeek, 4/4 citation xác minh được, 6.491 ms), không phải từ stub |

---

## 10. Đo được

* **37 test mới**: 21 tĩnh (`test_ui.py`) + 7 proxy (`test_ingest_proxy.py`) +
  4 nguồn/namespace + 5 bịt lỗ tiêm — cộng **13** bài Playwright
* 2 323 xanh bộ mặc định (2 skip) · ruff/mypy sạch
* chi phí **~$0,002**

## 11. Việc sinh ra từ lượt này

| ID | việc |
|---|---|
| `NEW-11` | **Cho phép tải tài liệu lên, hoặc chốt là không bao giờ.** Cần một quyết định về license trước khi cần một endpoint: ai chịu trách nhiệm cho tài liệu người dùng đẩy vào, và nó có được trộn vào cùng collection với corpus công khai không |
| `W6-02` | Khoá API là hàng rào cuối giữa "người lạ dùng được" và "người lạ dùng được **không cần gì cả**" |
