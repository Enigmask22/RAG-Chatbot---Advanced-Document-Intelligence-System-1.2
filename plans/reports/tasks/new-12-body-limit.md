# `NEW-12` — trần thân request: **852 MB → 55 MB**, và tiền đề của nợ này sai

*2026-09-08 · `serving/api/body_limit.py` + 2 mối nối · **20 test mới** · tiêm **16/16 đỏ** · chi phí **$0***

> **Nợ:** *"Trần kích thước thân request, ở tầng trước ứng dụng. `W6-06` đóng
> trục `filters` (`SEC-04`) nhưng **lỗ gốc không đóng được trong mã**: Pydantic
> chỉ thấy thân request **sau** khi nó đã được đọc trọn vào bộ nhớ. Chỗ đúng là
> reverse proxy (`client_max_body_size`)."*

---

## 0. Đo trước, kết luận sau

`POST` 200 MB, cùng ngăn xếp với API thật (uvicorn + starlette + pydantic), RSS
lấy mẫu liên tục 5 ms/lần:

| | status | RSS đỉnh |
|---|---|---:|
| `/chat` — **có** `max_length=8000` của Pydantic | 422 | **852,3 MB** |
| `/raw` — không Pydantic, chunked | 200 | 421,4 MB |
| `/raw` — không Pydantic, khai `content-length` | 200 | 406,3 MB |
| **`/raw` + trần ASGI**, chunked | **413** | **55,0 MB** |
| **`/raw` + trần ASGI**, khai `content-length` | **413** | **54,8 MB** |

Nền của tiến trình là 54,5–54,9 MB. Nên **trần đưa mức tăng về đúng 0**.

`probes/new12-body-limit.md` · `scratchpad/probe_body.py`

---

## 1. ⭐⭐ Tiền đề của nợ **sai**, và phép đo là thứ chỉ ra điều đó

`W6-06` §8 viết: *"lỗ duy nhất không đóng được **trong mã**: Pydantic chỉ thấy
thân request sau khi nó đã được đọc trọn vào bộ nhớ."*

Vế sau **đúng**. Vế trước **sai** — nó đúng với *Pydantic*, không đúng với
*ASGI*. Middleware ASGI thuần chạy **dưới** tầng ấy: nó thấy `content-length`
trước khi một byte thân nào được đọc, và nó thấy từng khung `http.request` khi
chúng tới. 852 MB → 55 MB, không cần một dòng cấu hình proxy nào.

⚠️ Cùng họ với `NEW-14` sáng nay: một dòng nợ mang sẵn **kết luận** về chỗ phải
sửa, và kết luận ấy sai. Ở `NEW-14` là nghi phạm (*"phân trang theo con trỏ"*),
ở đây là tầng (*"chỉ proxy mới làm được"*). Bài học chung: **một dòng nợ được
phép ghi triệu chứng; nó không được phép ghi chẩn đoán như thể đã đo.**

### Và hàng đầu bảng là hàng đáng đọc kỹ nhất

`security-final` §8 ghi *"không trần (**ngoài trần từng trường**)"* — đọc như
một giảm nhẹ một phần. Đo được: trần từng trường không giảm nhẹ gì, nó
**khuếch đại**. 852 MB là **4,3× payload** và **cao hơn** đường không có
Pydantic (421 MB), vì đường 422 còn dựng thêm bản giải mã JSON và thông điệp
lỗi. Trên trục kích thước request, `max_length=8000` là một **số âm**.

---

## 2. ⭐⭐ Và đây mới là thứ thật sự đóng `SEC-04`

`W6-06` chặn `filters` ở `MAX_FILTER_VALUES = 100` **giá trị mỗi trường** —
nhưng không chặn **độ dài từng giá trị**. Đếm:

| thân request | byte | hợp lệ với mọi phép kiểm hôm nay? |
|---|---:|---|
| `/chat` message ASCII đầy | 8.083 | ✅ |
| `/chat` message tiếng Việt đầy | 24.083 | ✅ |
| `/ingest` 1000 × 200 ký tự | 204.090 | ✅ **lớn nhất hợp lệ** |
| `/chat` + `filters` 1 KB/mục × 100 × 5 trường | 550.161 | ✅ |
| **`/chat` + `filters` 1 MB/mục × 100** | **100.048.510** | ✅ ⚠️ |

Một thân **100 MB** đi qua **mọi** phép kiểm hệ thống có. `W6-06` đóng đúng
trục nó nhìn vào và để hở trục ngay bên cạnh — đó là chế độ hỏng cố hữu của mọi
hàng rào đếm-theo-trường: **nó buộc người viết phải liệt kê đúng và đủ mọi
trục**. Một trần tính bằng **byte của cả thân** không phải liệt kê gì; nó chặn
mọi trục cùng lúc, kể cả những trục chưa ai nghĩ ra.

⭐ Nên trần mặc định **1 MiB** không phải một con số tròn chọn cho đẹp: nó là
**5,1×** thân hợp lệ lớn nhất **đo được** (204.090 byte). Có test ghim cả hai
đầu — `/ingest` 204.090 byte **không** bị chặn, và `DEFAULT_MAX_BODY_BYTES`
phải lớn hơn con số ấy.

---

## 3. ⚠️ Ba lần trong một ngày tôi bịa lý do cho một lựa chọn chưa nghĩ hết

`NEW-10` §4 ghi bài học ấy sáng nay. `NEW-14` tái phạm hai lần. Hạng mục này
thêm **ba** lần nữa, và cả ba đều bị **phép đo** bắt, không phải đọc lại mã:

| tôi viết | đo ra |
|---|---|
| `connection: close` khi từ chối — "việc đúng phải làm" | client nhận **`ReadError`**, không đọc được 413. Bỏ header thì **cả hai** ca trả 413 thật, RSS không đổi. |
| trần phải nằm **ngoài** auth vì *"thân 200 MB không khoá không được chờ xác thực"* | `AuthMiddleware` quyết định **hoàn toàn bằng header**, không chạm `receive` — nó từ chối sau **0 byte**, còn trần phải đếm tới `max_bytes`. **Phép từ chối rẻ hơn phải ra ngoài hơn.** Đảo lại. |
| bọc scope không-http *"là cách làm chết `startup`"* | Tiêm bỏ hẳn nhánh ấy: `lifespan` **chạy trót lọt**, không bài nào đỏ. |

Ca thứ ba đáng nói riêng, vì cách xử nó **khác** với `_retire` của `NEW-10`. Ở
đó tôi **xoá** hàng rào, vì nó bảo vệ một kịch bản không thể xảy ra. Ở đây phép
kiểm `scope["type"] != "http"` là **hợp đồng ASGI** — mọi middleware phải có
nó, và ngày có route WebSocket đầu tiên thì phép đếm byte của HTTP không được
áp lên nó. Nên nó **ở lại**, nhưng:

* docstring nói thẳng rằng nó **hôm nay không quan sát được bằng hành vi**;
* bài test ghim nó bằng phép so **danh tính** (`receive` và `send` mà ứng dụng
  nhận phải **đúng là** hai object gốc) — thứ duy nhất quan sát được — thay vì
  bằng một câu chuyện.

Ranh giới giữa hai ca: *"kịch bản này không thể xảy ra"* ⇒ xoá. *"hợp đồng đòi
thế, và hôm nay chưa ai gọi tới"* ⇒ giữ, nhưng **khai là inert** và ghim bằng
thứ đo được.

---

## 4. Tiêm lỗi: **16/16 đỏ**, sau ba lượt — và lượt hai tìm ra một lỗ **trong mã**

Lượt một **13/14**. `M10` (bỏ nhánh không-http) sống — **lỗ test**, không phải
mutant tương đương: bài của tôi khẳng định `app.duoc_goi`, thứ vẫn đúng khi
middleware bọc cả `lifespan`. Viết lại theo danh tính ⇒ 14/14.

⭐⭐ Lượt hai không sinh ra từ một phép tiêm mà từ việc **soát lại ba đường
thoát** — đúng danh sách đã dùng ở `NEW-10`. Đường thứ ba hở: ứng dụng thấy
`http.disconnect` rồi **`return` mà không phản hồi gì** — không ném, không gửi
khung nào. Phép cứu 413 của tôi nằm **trong nhánh `except`**, nên đường ấy
không có phản hồi nào cả và **request treo tới lúc client bỏ cuộc**. Cùng hình
dạng với `NEW-10` §2: một phép dọn đặt sau chỗ nó cần phủ. Chuyển ra ngoài
`try` ⇒ phủ cả hai đường thoát; `M15` ghim.

Lượt ba **15/16**. `M16` (không chặn khung thân sau khi vượt trần) sống — lỗ
test lần nữa, và nó lộ ra một thói quen: mọi bài của tôi đọc `bat.status`, thứ
chỉ nhìn khung `start` **đầu tiên**. Với `M16`, ứng dụng nhả nốt cặp
start+body 200 của nó **sau** phản hồi 413 — hai phản hồi chồng lên nhau trên
một kết nối, tức lỗi giao thức — mà `status` vẫn trả 413 và mọi bài vẫn xanh.
Sửa bằng cách **đếm khung**, không đọc trường ⇒ 16/16.

| phép | vì sao nó phải đỏ |
|---|---|
| `M1`/`M2` bỏ hoặc nới phép kiểm `content-length` | đường rẻ nhất, và ranh giới `>` chứ không `>=` |
| `M3`/`M4` bỏ hoặc nới phép đếm byte | chunked không có lời khai nào để tin |
| `M5` không trả `http.disconnect` | thân vẫn chảy vào RAM, chỉ status đổi — đúng bản vá giả nguy hiểm nhất |
| `M6` để nguyên 400 của Starlette | client không biết lý do thật |
| **`M7` nuốt luôn lỗi thật của ứng dụng** | mọi 500 thành 413 ⇒ bảng báo "người dùng gửi thân quá to" cho một sự cố máy chủ |
| `M8`/`M9` sai status / không nói ra con số trần | người bị chặn phải sửa được request của mình |
| `M11` trần 1 MiB → 64 KiB | chặn nhầm `/ingest` |
| **`M15` đẩy phép cứu 413 lùi vào `except`** | đường thoát thứ ba: ứng dụng return mà không phản hồi ⇒ treo |
| **`M16` không chặn khung thân sau khi từ chối** | hai phản hồi chồng nhau trên một kết nối |
| `M12` gỡ middleware khỏi `app.py` | mối nối cũng phải được canh |
| **`M13` đảo thứ tự với auth** | ghim đúng quyết định ở §3 |
| **`M14` đẩy ra ngoài `RequestContext`** | 413 mất `X-Request-ID` |

---

## 5. ⚠️ Cái này **không** thay được reverse proxy

Nói rõ để không ai đọc nhầm dòng đóng nợ:

* Byte vẫn **đi hết qua socket vào tiến trình ứng dụng** trước khi bị đếm.
  Proxy hấp thụ chúng sớm hơn một tầng.
* Chỉ proxy mới chặn được một client mở **trăm kết nối** rồi nhỏ giọt từng byte
  — trần này tính theo **một** request.
* `TD-39` (chặn theo IP) vẫn cần đúng chỗ ấy.

Cái đã có là hàng rào **tồn tại ngay hôm nay, ở mọi chỗ hệ chạy** — kể cả Space
HF, nơi không có proxy nào thuộc quyền ta, và kể cả `make up-api`, nơi compose
không có proxy nào cả.

---

## 6. Việc sinh ra từ lượt này

* ~~`NEW-12`~~ **đóng** ở tầng ứng dụng. Dòng *"Kích thước thân request"* của
  `security-final` §8 chuyển từ **không có hàng rào** sang **1 MiB, có test**.
* ⚠️ **`SEC-04` giờ mới thật sự đóng.** `MAX_FILTER_VALUES` ở lại — nó vẫn là
  hàng rào đúng cho *số lượng* `MatchAny` gửi sang Qdrant — nhưng trục **độ dài
  từng giá trị** thì trần byte đóng, và §2 ghi con số 100 MB để lần sau không
  ai phải phát hiện lại.
* 💡 **Reverse proxy vẫn là việc còn lại**, giờ với phạm vi hẹp hơn và nói được
  ra: nhỏ giọt nhiều kết nối, và chặn theo IP (`TD-39`). Không còn là *"chỗ duy
  nhất làm được"*.
