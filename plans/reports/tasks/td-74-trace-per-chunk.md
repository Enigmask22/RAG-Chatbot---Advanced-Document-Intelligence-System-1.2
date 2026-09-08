# `TD-74` — trace cắt theo KHỐI, và một bất biến giữ cho hai đường không trôi

*2026-09-08 · `serving/core/chat.py` · 5 test mới + 1 chốt integration · tiêm **7/7 đỏ** · chi phí **$0***

> **Nợ `TD-74`:** *"Prompt trong trace cắt ở 4 000 ký tự trong khi prompt thật
> đo được 9 392. […] người gỡ lỗi một câu trả lời sai không đọc được nửa sau
> ngữ cảnh, và đó thường là nửa chứa lỗi. Quyết: cắt theo **chunk** (giữ đủ 5
> khối, mỗi khối cắt riêng và nói rõ đã cắt) chứ không cắt chuỗi đã ghép."*

## 1. Cái sửa không nằm ở `redact()` — nó nằm ở NGƯỜI GỌI

`redact()` vốn đã đệ quy qua list/dict và cắt **từng chuỗi** ở `_MAX_TEXT`.
Vấn đề là span `prompt` đưa vào **một chuỗi đã ghép**: message user cuối mang
cả khối `NGỮ CẢNH` ~10k ký tự. Nên bản vá không thêm cơ chế cắt nào — nó đổi
**hình của dữ liệu đi vào**: message ngữ cảnh phát `content_parts` (mỗi chunk
một mảnh, header dán vào mảnh đầu, câu hỏi là mảnh cuối), và trần 4.000 sẵn có
tự áp cho từng mảnh. Trần giữ nguyên; điều dòng nợ gọi là *"chỗ cắt thì sai"*
được sửa mà không mở rộng ngân sách bộ nhớ nào.

## 2. ⭐⭐ Bất biến tái dựng: view và chuỗi gửi đi đến từ CÙNG các mảnh

Rủi ro thật của một "cách trình bày cho trace" là nó thành **nguồn thứ hai**:
sửa `prompt()` mà quên view (hoặc ngược lại) thì trace khai một prompt không ai
gửi — tệ hơn cả bị cắt, vì nó sai mà trông đủ. Chặn bằng cấu trúc + một phép
khẳng định:

* `prompt()` và `prompt_trace_view()` cùng gọi `user_content_parts()` — một
  nguồn, hai cách ghép;
* test ghim `"\n\n".join(parts) == prompt()[-1].content` — **bằng đúng từng
  ký tự**, kể cả header `NGỮ CẢNH:` dán bằng `\n` vào khối đầu. Hai mutant
  (đổi ký tự nối, tách header thành mảnh riêng) chết vì đúng phép này.

## 3. Tiêm lỗi: 7/7 đỏ

`M1` (callsite quay về chuỗi ghép) chỉ chết bởi **chốt integration** trên span
thật — unit test không thấy được việc nối dây; cùng bài học `M13` của `TD-39`.
`M6` (parts bị ghép lại trước khi vào trace) chết bởi bài "đầu của MỌI khối
phải sống sót sau `redact`" — 5 khối × 6.000 ký tự, khối 2–5 từng biến mất với
hành vi cũ. `M7` (mất guard `retrieves`) chết bởi bài nhánh no-retrieval.

⚠️ Bộ hoàn tác của lượt tiêm giữ nội dung trong bộ nhớ — bài học `git checkout
-- trên file chưa commit` của `TD-78` sáng nay, áp dụng ngay trong ngày.
