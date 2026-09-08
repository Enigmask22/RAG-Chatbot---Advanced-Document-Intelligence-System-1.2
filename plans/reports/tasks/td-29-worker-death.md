# `TD-29` — giết một worker thật, và tiền đề "cần đường code riêng cho test" là sai

*2026-09-08 · `tests/integration/test_worker_death.py` (chỉ test, **0 dòng mã production**) · 18,5 s/lượt, ổn định 3/3 · chi phí **$0***

> **Nợ `TD-29`:** *"Đường 'worker chết hẳn' chưa được kiểm bằng cách giết tiến
> trình thật. […] Vướng: job phải đủ chậm để còn đang chạy lúc bị giết […] nên
> cần một job cố ý chậm, tức **một đường code chỉ tồn tại cho test**."*

## 0. Kịch bản, chạy thật

`enqueue` → worker arq ở **tiến trình con** nhặt job (job ngủ 60 s ở lần thử
đầu) → `TerminateProcess`/`SIGKILL` → khoá `in-progress` **không được nhả, chỉ
hết hạn** (đo được: TTL = `job_timeout + 10 s`, đúng công thức
`arq/worker.py:277`) → worker thứ hai nhặt lại với `job_try = 2`, **không đặt
trước khoá nào** → job hoàn tất. Chuỗi `tries` trong Redis: `[1, 2]`.

## 1. ⭐⭐ Tiền đề của dòng nợ SAI — lần thứ tư trong hai ngày

Job chậm sống **trong module test**, đăng ký vào một worker của test, trên hàng
đợi riêng (`arq:td29`). Thứ đang kiểm — khoá hết hạn, `INCR` ở lần nhặt sau —
là hành vi của **arq**, giống hệt cho mọi hàm được đăng ký; production không
cần thêm nhánh nào. Cùng họ với `NEW-14`/`NEW-12`/`TD-39`: dòng nợ ghi sẵn một
chẩn đoán chưa đo, và người làm sau phải **kiểm tiền đề trước khi kiểm việc**.

## 2. ⭐ Ba phép khẳng định chống "xanh sai kịch bản" — viết TRƯỚC khi chạy

Bài này không có mã production để tiêm lỗi, nên phần "tiêm" đổi dạng: hỏi *"bài
này xanh được bằng những con đường sai nào?"* và chặn từng đường **bằng phép
khẳng định trong chính bài test**:

| đường xanh sai | phép chặn |
|---|---|
| `proc.kill()` không giết được ai | `proc.wait(timeout=10)` — treo là đỏ |
| chậm tay: arq tự huỷ job vì quá giờ TRƯỚC khi kịp giết, job đi đường retry-vì-lỗi (arq mặc định `retry_jobs=True`) và `tries` vẫn ra `[1, 2]` | ngay sau khi giết: khoá `in-progress` phải **còn sống** và `tries == [1]` — job phải đang bay đúng lúc chết |
| khoá bị nhả chủ động thay vì hết hạn | TTL đo tại chỗ phải nằm trong `(job_timeout+9 s, job_timeout+10 s]` — ghim công thức của arq bằng phép đo, đổi là đỏ |

## 3. ⚠️ Cái giá nói ra: bài test 18,5 giây, và con số ấy KHÔNG giảm được

Hằng `+10 s` trong TTL khoá của arq không cấu hình được, và phục hồi **chỉ**
xảy ra khi khoá hết hạn — bài test phải chờ trọn. `job_timeout = 5 s` là nhỏ
nhất còn giữ khoảng an toàn giữa "nhặt" và "giết". Đây cũng chính là cảnh báo
vận hành của `TD-29` thu nhỏ: production để `job_timeout = 2 giờ`, nên một
worker chết ngoài đời làm job nằm im **tới 2 giờ**. Quyết định giữ nguyên
(không thêm heartbeat): build corpus hiện tại 397 s, worker chết là sự cố hiếm,
và giá của một heartbeat là một vòng đời tiến trình mới phải nuôi — mở lại khi
nào SLA ingest có thật.
