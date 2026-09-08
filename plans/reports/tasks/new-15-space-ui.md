# `NEW-15` — UI Space demo: thiết kế là thứ CSS làm, không phải thứ emoji làm hộ

*2026-09-08 · `space/app.py` (tầng trình bày) + 4 test mới trong `test_space_assets.py` · tiêm **6/6 đỏ** · ruff + mypy sạch 278 file · chi phí **~$0.001** (một lượt sinh thật để chụp màn hình lượt hỏi-đáp)*

> Yêu cầu người dùng (kèm ảnh chụp UI cũ): *"Nâng cấp UI lên đi (tùy bạn thiết kế),
> tránh sử dụng các icon tự sinh, design sao cho trông chuyên nghiệp, non-AI
> Generate, gây ấn tượng với client và nhà tuyển dụng."*

## 1. Chẩn đoán UI cũ: ba dấu vết "AI-generate"

1. **Emoji làm việc của đồ hoạ**: 🟢/🔴 làm đèn trạng thái, ✅ làm nhãn xác minh,
   ⛔/⚠️ mở đầu thông điệp lỗi. Emoji trong giao diện là chữ ký của một trang
   được sinh ra, không được thiết kế.
2. **Dấu vết template Gradio nguyên khối**: footer "Use via API · Built with
   Gradio · Settings", nhãn component lộ ("Chatbot", "Examples"), typography
   mặc định.
3. **Không có hệ thống thị giác**: tiêu đề là một dòng markdown `##`, không
   wordmark, không nhịp khoảng cách, panel nguồn là chữ trần không khung.

## 2. Cái sửa — và cái cố ý KHÔNG sửa

**Sửa (toàn bộ nằm ở tầng trình bày):** topbar thương hiệu (wordmark "RAG
Platform" + tag "demo công khai" + mô tả một dòng) · chip trạng thái quota là
**chấm CSS hai class `on`/`off`** với số liệu bằng chữ mono · chữ **IBM Plex
Sans** cho giao diện, **IBM Plex Mono** cho dòng số đo (`bundle 0.2.1 · model
deepseek-v4-flash · TTFB 1065 ms · $0.001701` giờ đọc như một dòng telemetry —
đúng bản chất của nó) · footer Gradio ẩn bằng CSS · nhãn xác minh trích dẫn
thành chữ đậm `trích dẫn đã xác minh` · panel nguồn đóng khung, nhãn viết hoa
giãn ký tự · câu hỏi mẫu thành chip bo tròn xếp hai cột.

**Không sửa:** mọi hàng rào an toàn của `W6-02`. `sanitize_html=True` tường
minh ở cả hai bồn chứa, nội dung chunk vẫn vào code fence, và chip quota là
`gr.HTML` **chỉ vì** đầu vào của nó là số đếm của `GUARD` — không một chữ nào
của người dùng hay của model đi qua đường raw HTML. Ranh giới "cái gì được
render thô" là quyết định an ninh, không phải quyết định thẩm mỹ, nên nó được
ghi thành chú thích ngay tại chỗ khai component.

## 3. Ba vòng Playwright — vì CSS viết chay là CSS chưa chạy

Không đoán DOM của Gradio 6: chạy app thật (`make space-run`, gradio local
6.26.0 = đúng bản Space) rồi chụp bằng Playwright cả hai theme, ba vòng:

| vòng | thấy gì | sửa gì |
|---|---|---|
| 1 | Chip câu hỏi mẫu xếp dọc; icon ≡ trước label; nghi vấn dải đen đầu trang | `.gallery` cần `flex-direction: row` (class gốc là column); ẩn SVG label; dải đen là artifact chụp full-page, không phải element — bỏ qua |
| 2 | Vẫn một chip mỗi hàng dù container đã flex-row | `elementFromPoint`/computed style chỉ ra **Gradio đóng cứng width nút 405px** — cột 665px chỉ lọt một |
| 3 | `max-width: calc(50% - 4px)` + ellipsis ⇒ lưới chip 2×2, cả hai theme đạt | — |

Vòng 2 là lý do quy trình này tồn tại: rule `#examples > .gallery { display:
flex; flex-direction: row }` **đúng và có hiệu lực** (computed style xác nhận
`display: flex, dir: row, wrap: wrap`) mà layout vẫn sai — vì thủ phạm là
thuộc tính của phần tử **con**. Một CSS "hợp lý" được commit không qua vòng
chụp sẽ mang đúng lỗi này lên Space.

Kiểm chứng cuối bằng một lượt hỏi thật (câu GDP mẫu, ~$0.001): trích dẫn
`[1]…[5]` hiển thị, nhãn xác minh bằng chữ ở nguồn `[1]`/`[2]`, dòng stats
mono, quota nhảy 0→1/500, theme sáng và tối đều sạch.

## 4. Test mới: cấm emoji phải là một bài test, không phải một dòng report

Emoji trong chuỗi giao diện là thứ dễ mọc lại nhất — mỗi chỗ hiển thị mới là
một cám dỗ gõ ✅. `TestGiaoDienKhongDungEmoji` (4 bài):

* **Cấm 10 emoji trong string literal của `app.py`** — soi **AST**, không soi
  văn bản file, và **trừ docstring** (quy ước ⚠️/⭐ của repo sống ở đó — nhà
  của chúng là người đọc mã, không phải màn hình). Cùng bài học `W6-02` đã trả
  học phí: *"chuỗi này có xuất hiện ở đâu trong file không"* không bao giờ là
  câu hỏi đang cần hỏi.
* **Nhóm chứng** kiểu `W6-07`: ghim `"_Chưa có lượt nào._"` và mảnh f-string
  `"```text"` phải **nhìn thấy được** trong danh sách literal — bài cấm xanh vì
  sạch emoji, không phải vì bộ thu thập trả về rỗng.
* Đèn trạng thái phải có **cả hai** class `on`/`off` trong CSS — thiếu một là
  hai trạng thái trông y nhau.
* CSS phải có luật ẩn `footer` — dấu vết template rõ nhất của trang.

Tiêm 6/6 đỏ: ✅ mọc lại trong nhãn xác minh · bỏ luật ẩn footer · đổi
placeholder bảng nguồn · mất class `off` · đèn quay về emoji 🟢 · Chatbot mất
`sanitize_html` (bài cũ của `W6-02` vẫn còn răng trên layout mới).

## 5. Giới hạn nói ra

* ⚠️ **Ảnh chụp không vào repo** — bằng chứng thị giác nằm ở quy trình (ba vòng
  §3), không có file .png nào được commit. Test chỉ ghim được các bất biến
  văn bản (emoji, CSS rule, sanitize); "trông chuyên nghiệp" là phán quyết
  của người nhìn, và người dùng là trọng tài cuối.
* ⚠️ Selector `.gallery`/`.label` là DOM nội bộ của Gradio 6.26.0 — một lần
  nâng `sdk_version` có thể làm chip trở về xếp dọc (suy giảm thẩm mỹ, không
  suy giảm chức năng). Frontmatter ghim `sdk_version: 6.26.0` nên điều này chỉ
  xảy ra khi có người chủ động nâng.
* Google Fonts (`fonts.googleapis.com`) tải được trên HF Spaces; nếu bị chặn ở
  môi trường nào đó, stack dự phòng `'Segoe UI', system-ui` vẫn giữ layout.
