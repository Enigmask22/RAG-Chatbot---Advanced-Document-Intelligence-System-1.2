# `TD-78` — lượt model im lặng chấm được 👎, vì hàng trợ lý có mặt trước cả stream

*2026-09-08 · `serving/core/chat.py` + UI + 4 mối đọc/ghi · **6 test mới** (+2 bài đổi theo hợp đồng) · tiêm **10/10 đỏ** (lượt chạy được; lượt đầu tự xoá bản vá — §4b) · chi phí **$0***

> **Nợ `TD-78` (đã thu hẹp 06/09 bởi `W6-01`):** *"Vẫn không chấm được một câu
> trả lời rỗng: `_save()` cố ý bỏ qua text rỗng nên `answer_message_id` trỏ vào
> một hàng **không tồn tại** và `POST /feedback` trả 404 — đúng lượt (model im
> lặng, `finish_reason="empty"`) đáng nhận 👎 nhất. […] **Còn lại**: ghi hàng
> trợ lý **luôn** […] rồi để đường đọc lịch sử lọc nó ra — lúc ấy 👎 mới ghi
> được. Cửa sổ đua nhỏ (hàng ghi ở task nền) cũng đóng cùng lúc."*

---

## 0. Đo được, trên Postgres thật

| mệnh đề | trước | sau |
|---|---|---|
| 👎 một lượt model im lặng | **404** | **201**, và mục review mang đúng câu hỏi |
| hàng trợ lý tồn tại lúc client cầm khung `meta` | không (task nền) | **có** — đọc thẳng Postgres **giữa stream**: `("assistant", "", "pending")` |
| lịch sử sau một lượt rỗng | (hàng không tồn tại) | chỉ hàng `user` — hàng rỗng **vô hình với người đọc** |
| trang `limit=3` trên hội thoại có 1 hàng rỗng xen giữa | — | **đúng 3 hàng nhìn thấy**, `next_after` đúng |

---

## 1. ⭐⭐ Một nước đi đóng HAI lỗ — vì hai lỗ là một lỗ

Dòng nợ liệt kê hai việc: ghi hàng cho lượt rỗng, **và** đóng cửa sổ đua
(feedback tới trước khi task nền ghi xong). Chúng là **cùng một lỗ** nhìn từ hai
phía: `answer_message_id` phát ra ở khung `meta` là một **lời hứa**, và cả hai
ca chỉ là hai cách lời hứa ấy chưa/không được giữ.

Nên bản vá không phải "sửa `_save` cho nó ghi cả text rỗng" (chỉ đóng lỗ một):
**`_open_turn` ghi placeholder** (`content=""`, `finish_reason="pending"`)
trong **cùng transaction** với câu hỏi, và `_save` đổi từ INSERT thành
**UPDATE**. Id là hàng thật từ trước khung SSE đầu tiên — lời hứa trở thành sự
thật tại thời điểm hứa, không phải tại một thời điểm nền nào đó sau này.

⚠️ Crash giữa stream để lại `pending` vĩnh viễn — **cố ý**: đó là sự thật
("lượt chưa từng kết thúc"), nó vô hình với người đọc nhờ bộ lọc §2, và một
job dọn dẹp là máy móc cho một ca chỉ xảy ra khi tiến trình chết giữa chừng.

## 2. ⭐ Bộ lọc phải nằm TRONG SQL, và lý do là một hợp đồng của client

`W4-06` quyết *"một hàng rỗng trong lịch sử tệ hơn không có hàng nào"* — giữ
nguyên, nhưng chuyển từ **lúc ghi** sang **lúc đọc**: hàng tồn tại cho
feedback, không cho hiển thị.

Chỗ dễ sai: lọc ở Python **sau** `limit`. `GET /conversations/{id}` suy
`next_after` từ `len(messages) == limit`, tức client đọc *"trang ngắn"* là
*"hết lịch sử"* — một hàng rỗng lọt vào trang làm trang hụt đi một, và client
**dừng phân trang giữa một lịch sử vẫn còn**, không dấu vết. Có bài test dựng
đúng ca ấy (`u1, a1(rỗng), u2, a2, u3` với `limit=3` phải ra `[u1, u2, a2]` và
`next_after="a2"`).

`_history` (ngân sách prompt) cùng lý do, khác hậu quả: nó lấy
`MAX_HISTORY_MESSAGES` hàng **mới nhất** rồi mới lọc Python — một chuỗi lượt
rỗng liên tiếp (provider trục trặc một lúc) lấp đầy cửa sổ bằng hàng vô hình và
prompt mất sạch hội thoại thật. Test quan sát qua **đường thật**: nhét
`MAX_HISTORY_MESSAGES` hàng rỗng đè lên một cặp hỏi–đáp thật, `POST /chat`, và
đọc prompt mà một LLM-ghi-âm nhận được.

## 3. ⭐ Bài test giữa stream — vì "trước khi stream kết thúc" là chính mệnh đề

Cửa sổ đua không kiểm được bằng "gọi xong rồi feedback thật nhanh": nhanh bao
nhiêu vẫn là *sau*. Bài test mở `client.stream`, đọc **đúng tới khung `meta`**,
rồi trong lúc stream còn treo giữa chừng, đọc thẳng Postgres bằng engine chủ:
hàng phải là `("assistant", "", "pending")` **ngay lúc ấy**. Không drain,
không chờ — drain xong thì mệnh đề đã đổi thành mệnh đề khác.

## 4. ⚠️ Hai lỗi của chính lượt này

**(a) Gọi `chat._history` tay từ test là một `InterfaceError` cross-loop.**
`sessions` của app sống trong vòng lặp của portal `TestClient`;
`pytest.mark.asyncio` chạy vòng lặp khác. Chữa bằng cách đi đường thật
(`POST /chat` + `_RecordingLLM`) — và bài test **tốt lên** vì thế: nó kiểm
prompt thật sự gửi đi, không kiểm một hàm nội bộ.

**(b) ⭐⭐ Bộ hoàn tác của lượt tiêm lỗi XOÁ CHÍNH BẢN VÁ.** Script tiêm dùng
`git checkout -- <file>` để hoàn tác từng mutant — nhưng `chat.py` lúc ấy
**chưa commit**, nên checkout khôi phục về HEAD, tức xoá sạch TD-78 sau mutant
đầu tiên. Tám mutant sau báo `PATCH FAIL (count=0)` — tiêm vào một file đã
không còn là file định tiêm. Hai bài học:

* Công cụ hoàn tác phải hoàn tác về **bản đang kiểm** (giữ nội dung trong bộ
  nhớ, ghi lại), không về một mốc git nào đó — hai thứ chỉ trùng nhau khi mọi
  thứ đã commit, và lượt tiêm lỗi theo định nghĩa chạy trên mã chưa commit.
* `1 chạy + 8 PATCH FAIL` là **tín hiệu phép đo hỏng**, không phải 8 lỗ test —
  cùng họ với "một phép đo cho cùng một con số ở mọi cấu hình" của probe RSS.

Mọi thay đổi khôi phục từ ngữ cảnh phiên làm việc (từng chuỗi thay thế còn
nguyên), xác minh lại bằng 29/29 xanh trước khi tiêm lại.

## 4b. ⭐⭐ Và lượt chạy ĐỦ hai tầng bắt được một lỗi thật: hai hàng, MỘT đồng hồ

Hai bài của `test_chat_stream` đỏ: lịch sử trả `["assistant", "user"]`. Vì
placeholder chèn **cùng transaction** với câu hỏi, và `now()` của Postgres là
**thời điểm bắt đầu transaction** — hai hàng nhận CÙNG một `created_at`, thứ
tự `(created_at, id)` rơi xuống so **id ngẫu nhiên**: tung đồng xu mỗi lượt.

Chữa đúng ngữ nghĩa cũ thay vì chữa triệu chứng: `_save` đặt lại
`row.created_at = func.now()` **lúc điền** — chính là mốc mà kiến trúc cũ
(INSERT ở task nền) vẫn ghi, nên người đọc lịch sử không thấy gì đổi; còn lúc
chưa điền thì hàng vô hình nên mốc của nó không có người xem.

⚠️ Cái ghim sẵn có (thứ tự role trong lịch sử) giết mutant này với xác suất
~50% — nó so một **thứ tự suy ra** từ đồng xu uuid. Thêm một bài ghim **tất
định**: so thẳng hai mốc `created_at` trong Postgres (`M10`).

## 5. Tiêm lỗi: 10/10 đỏ — và một phép được đón trước

| phép tiêm | bài giết nó |
|---|---|
| M1 bỏ ghi placeholder | happy-path 👎 (mọi feedback mất chỗ trỏ) |
| M2 placeholder sinh `finish_reason="stop"` | bài đọc giữa stream |
| M3 placeholder sinh content khác rỗng | bài đọc giữa stream |
| M4 `_save` quay về nết cũ (return sớm khi rỗng) | bài lượt im lặng — **nhờ phép khẳng định đọc thẳng Postgres thêm TRƯỚC lượt tiêm**: mọi phép khẳng định khác đều qua với nết cũ (placeholder tồn tại nên 👎 vẫn 201), chỉ khác hàng kẹt ở `pending` |
| M5 `_save` không điền content | happy-path (hàng bị lọc khỏi lịch sử) |
| M6 `load_history` bỏ lọc | bài lịch-sử-giấu-hàng-rỗng |
| M7 lọc ở Python SAU `limit` | bài trang-`limit=3` |
| M8 `_history` bỏ lọc SQL | bài ngân-sách-prompt |
| M9 placeholder không mang `user_message_id` | bài ghép câu hỏi của `NEW-08` — placeholder phải mang khoá nối **từ lúc sinh** |
| M10 `_save` không đặt lại `created_at` lúc điền | bài so-thẳng-hai-mốc (§4b) — ghim tất định, thay cho cái ghim tung-đồng-xu |

`M4` là phép đáng tiền nhất: nó được **đoán trước khi chạy** ("mệnh đề nào chưa
bài nào ghim?") thay vì để lượt tiêm chỉ ra — rẻ hơn một vòng chạy lại, và là
đúng thói quen mà `M13` của `TD-39` (chỗ nối không có bài test nào) dạy.

## 6. UI và các mối còn lại

* `serving/ui/index.html`: bỏ nhánh tắt nút khi `finish_reason === "empty"` —
  nút chỉ còn ẩn khi **không có** `message_id` (chế độ không trạng thái). Tham
  số `finishReason` của `feedbackButtons` biến mất cùng nhánh ấy: **đếm** ra
  đúng một chỗ gọi (bài học `NEW-14`: đếm, đừng ước lượng).
* Thông điệp 404 của `record_feedback` thôi nói *"hoặc nó chưa được ghi xong"*
  — lời giải thích ấy mô tả kiến trúc cũ.
* ⚠️ Kèm theo lượt này nhưng là nợ của lượt trước: **7 fixture** integration
  dựng `Settings(...)` không khai `quota_shared` — tức bộ đếm hạn mức của mọi
  bài ấy đổi giữa cục bộ và Redis theo việc **Docker đang bật hay tắt** (lỗi
  phụ-thuộc-môi-trường thứ năm trong ngày, quét một lần cho hết thay vì đợi
  từng bài đỏ). Tất cả khai `quota_shared=False` tường minh, trỏ về chú thích
  chung ở `chat_app.make`.
