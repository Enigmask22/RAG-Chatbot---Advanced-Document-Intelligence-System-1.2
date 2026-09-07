---
title: RAG Platform Demo
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12.12"
app_file: app.py
pinned: false
license: mit
short_description: RAG hỏi đáp trên 60 báo cáo World Bank, trích dẫn máy chủ tự xác minh
---

# RAG platform — demo công khai

Hỏi đáp trên **60 tài liệu World Bank về Việt Nam** (40 EN + 20 VI, 15.814
chunk). Truy hồi hybrid BGE-M3 (dense + sparse, hợp nhất RRF) → xếp lại bằng
cross-encoder `bge-reranker-v2-m3` trên ZeroGPU → sinh bằng `deepseek-v4-flash`.

**Mã nguồn, số đo eval và báo cáo từng hạng mục:**
[github.com/Enigmask22/RAG-Chatbot](https://github.com/Enigmask22/RAG-Chatbot)

## Điều đáng xem: trích dẫn do **máy chủ** xác minh

Model bắt buộc phải trích dẫn `[n]` kèm nguyên văn. Máy chủ đối chiếu từng
trích dẫn với **đúng chunk mà `n` chỉ vào** — trích dẫn đúng nguyên văn nhưng
gán sai số nguồn vẫn là `verified: false`. Dấu ✅ ở bảng nguồn là phán quyết
của máy chủ, không phải lời của model.

## Đây có phải hệ thống thật không

Có, và điều đó kiểm được: `requirements.txt` cài `rag-platform` từ
`git+…@<commit>`, nên toàn bộ truy hồi, prompt và xác minh trích dẫn là mã
trong repo, không phải bản viết lại cho demo. Bundle là `manifest.json` đúng
bản đã eval, và phép kiểm danh tính runtime (`TD-38`) chạy lúc khởi động —
Space **không lên được** nếu môi trường dựng ra một retriever khác chuỗi đã ký.

Index chạy trên `qdrant-client` local mode thay cho Qdrant server. Xếp hạng đo
được là **trùng khớp hoàn toàn** với server trên 30 truy vấn golden
(overlap@20 = 1,0; top-1 giống 30/30).

## Ba thứ bản demo này không có

| Thiếu | Hệ quả |
|---|---|
| Postgres | Không lưu lịch sử. Cũng là lựa chọn riêng tư: không giữ câu hỏi của người lạ |
| Redis | Không có cache — mỗi câu là một lượt sinh **thật**, không phải bản phát lại |
| Xác thực | Hàng rào duy nhất là hạn mức: trần tổng theo ngày + trần theo khách |

Trần theo khách dựa trên `x-forwarded-for`, mà header đó **người gọi ghi được**
— nên con số thật sự bảo vệ hoá đơn là trần tổng theo ngày. Viết ra thay vì để
người đọc tự đoán.

## Giấy phép

Mã: MIT. **Tài liệu trong corpus: [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/)**,
bản quyền thuộc **World Bank** — Space này chỉ phân phối lại và không sở hữu
chúng. World Bank không xác nhận nội dung do bản demo này sinh ra. Mỗi nguồn
trong giao diện đều dẫn về URL gốc của tài liệu.

Câu trả lời do LLM sinh và **có thể sai**. Đây là bản demo kỹ thuật, không phải
nguồn tư vấn.
