# `TD-49` — "mọi route đều hỏng" giờ nói được VÌ SAO, theo kiểu

*2026-09-08 · `packages/rag_core/llm/router.py` · 5 test mới · tiêm **7/7 đỏ** · chi phí **$0***

> **Nợ `TD-49`:** *"Lỗi 'mọi route đều không phục vụ được' gộp mọi lý do thành
> một `LLMError` phẳng. Kiểu `PermanentLLMError` không còn đọc được ở đó, nên
> người gọi không phân biệt được 'cả hai nhà đều sập' với 'request của mình sai
> ở cả hai nhà'."*

## 1. `AllRoutesFailed(LLMError)` — kiểu là cho người gọi, log là cho người vận hành

Hai chỗ ném cuối của `complete()` và `astream()` giờ ném `AllRoutesFailed`
mang `failures` (lỗi từng route **đã được hỏi**, đúng thứ tự thử), `skipped`
(số route bị cầu dao bỏ qua), và `__cause__` = lỗi route cuối — traceback mặc
định chỉ thẳng vào lần thử sau chót. Subclass của `LLMError` nên **mọi
`except LLMError` hiện có tiếp tục bắt được** — có test ghim riêng cho hợp
đồng tương thích ấy, vì tương thích lùi là một hợp đồng chứ không phải một sự
tình cờ.

## 2. ⭐⭐ `permanent` và cái bẫy của route bị bỏ qua

Câu người gọi thật sự hỏi là: *"thử lại có nghĩa không?"* —
`permanent == True` nghĩa là request của MÌNH bị mọi nhà từ chối, thử lại là
trả tiền cho cùng một lỗi.

Định nghĩa ngây thơ `all(isinstance(f, PermanentLLMError))` có một lỗ: một
route bị cầu dao bỏ qua là một route **chưa được hỏi**, không phải đã từ chối.
Thiếu vế `skipped == 0`, một cầu dao đang mở biến một 4xx lẻ ở route còn lại
thành "request sai ở mọi nhà" — và người gọi thôi thử lại **đúng lúc nên thử**
(route kia sắp nguội). `M1` của lượt tiêm chết vì đúng bài test dựng ca này:
mở mạch route đầu bằng một lỗi 500, rồi khẳng định `permanent == False` dù lỗi
duy nhất trong `failures` là permanent.

## 3. Phạm vi dừng lại ở KIỂU — người dùng đầu tiên đến sau

Nhánh *đứt-giữa-stream* (`emitted=True`) giữ nguyên `LLMError` phẳng: đó là
một sự cố khác (đã phát chữ đi rồi, không chuyển route được) với thông điệp tự
đứng vững. Việc serving map `permanent` thành mã HTTP khác (4xx thay vì SSE
`error` chung) là một quyết định UI/contract riêng — kiểu đã mang đủ dữ liệu
cho người sau làm việc ấy mà không phải đào lại router.

## 4. Tiêm lỗi: 7/7 đỏ

Đáng kể: `M3`/`M4` (quên gom lỗi ở một trong hai nhánh `except`) — mỗi nhánh
một mutant vì "thêm một danh sách là *hai* chỗ append", đúng dạng lỗi
`single_flight`/`_declare_zero` của `NEW-10`; `M6` ghim rằng `astream` không
được nhận ít thông tin hơn `complete`; `M7` ghim `skipped` được đếm thật chứ
không phải một trường luôn 0.
