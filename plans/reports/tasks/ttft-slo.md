# TTFT SLO — quyết định đã chốt, và một SLO chưa quan sát được thì chưa phải SLO

*2026-09-08 · `serving/core/metrics.py` + `serving/core/chat.py` + `rag-health.json` · 7 test mới · tiêm **7/7 đỏ** · chi phí **$0***

> **Bối cảnh:** `W6-05` đo được p95 end-to-end 4.842 ms **không đạt được bằng
> tối ưu** (bỏ toàn bộ truy hồi + rerank vẫn còn 4.055 ms) và đề xuất: giữ dòng
> 3.500 ms ở ❌, **thêm** một SLO TTFT. Người dùng chốt 08/09/2026:
> **TTFT p95 ≤ 2.000 ms ở tải thiết kế** — đạt hôm nay với u ≤ 2
> (1.400 ms @ u=1 · 2.200 @ u=8 thì trượt, đúng nghĩa "tải thiết kế").

## 1. Chốt một SLO là nhận một nghĩa vụ quan sát

Con số 1.400 ms của `W6-05` đo bằng **probe ngoài** — nó trả lời "hôm nay có
đạt không", không trả lời "tuần sau còn đạt không". Một SLO chỉ có nghĩa khi
người vận hành đọc được nó từ hệ đang chạy, nên phần việc thật của lượt này
không phải sửa bảng CHECKLIST mà là: `rag_ttft_seconds` trên `/metrics`, và
hai panel trên `rag-health.json` (p50·p95, và "% lượt vượt SLO" **đếm** bằng
bucket `le="2.0"` thay vì nội suy — bucket ấy có sẵn trong `_DURATION_BUCKETS`
từ ngày đầu, đúng triết lý "ngưỡng của bảng mục tiêu phải là một cạnh bucket").

## 2. Phép đo đã tồn tại — việc còn lại là không đo HAI lần

`chat.py` gán `ttfb_ms` tại token đầu từ trước (khung `done` khai nó cho
client). Lượt này chỉ nối nó tới Prometheus, và chỗ dễ sai là **đếm trùng**:
quét mọi span có khoá `ttfb_ms` thì một span tương lai mượn tên khoá ấy đếm
một lượt thành hai. Nên sink chỉ đọc từ `_ANSWER_SPANS` — danh sách đóng hai
cái tên (`completion`, `cache.replay`), mỗi lượt đúng một span như vậy. Mutant
`M2` (thay guard bằng `True`) chết vì đúng bài test dựng một `ttfb_ms` lạc
trên span `citations`.

Hai quy tắc đếm phải nói ra:

* **Lượt cache replay VÀO histogram** — SLO là của người dùng, không phải của
  model; bỏ những lượt nhanh nhất ra là tự làm xấu p95. `replay.end` giờ mang
  `ttfb_ms=elapsed`, **cùng con số** khung `done` khai — bảng và client không
  được kể hai chuyện về một lượt.
* **Lượt không phát được byte nào KHÔNG vào** — `finish_reason="empty"` không
  có "thời gian tới byte đầu"; ghi 0 là khai một lượt tức thời chưa từng xảy
  ra và p50 tụt theo tỉ lệ lượt rỗng.

## 3. Tiêm lỗi: 7/7 đỏ

`M7` (bỏ `ttfb_ms` khỏi `completion.end`) chỉ chết bởi bài **integration** đọc
`/metrics` thật sau một lượt `/chat` thật — unit test không thấy được việc nối
dây, cùng bài học `M1` của `TD-74`. `M1` (quên chia 1000) chết vì bài khẳng
định `_sum` bằng **giây**; `M5` ghim cạnh bucket 2.0 — mất nó thì panel "%
vượt SLO" chuyển từ phép đếm sang phép nội suy mà không ai báo.

## 4. Hệ quả trên sổ sách

* `G2` **✅ 4/4** — tiêu chí "không tổ hợp nào vượt 3.500 ms" tick với nghĩa
  *đã có kết luận cuối* (đo được, trượt, không đạt được bằng tối ưu, và người
  dùng quyết giữ ❌ + chốt SLO thay thế), không phải nghĩa *đã đạt ngưỡng*.
* Bảng §1: dòng TTFT hết ⏳, cột mục tiêu ghi ≤ 2.000 ms @ tải thiết kế.
* Dòng end-to-end 3.500 ms giữ nguyên ❌ — thay nó mới là dời cột gôn.

⚠️ **Giới hạn phải nói ra**: số 1.400/2.200/4.300 ms đo bằng stub hiệu chỉnh
(`W6-05`), không phải provider thật; và `rag_ttft_seconds` bắt đầu đếm từ
commit này — chưa có lịch sử. Lần đo tải kế tiếp trên hệ thật nên đối chiếu
probe ngoài với chính histogram này (hai đường đo một thứ là một phép kiểm
chéo miễn phí).
