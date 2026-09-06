"""Phục vụ trang giao diện — `W6-01`.

## ⭐ Một file tĩnh, không bước build

Không React, không bundler, không `node_modules`. Ba lý do đo được, không phải
sở thích:

1. **`W6-02` (demo HF Spaces) chỉ có một container.** Một bước build nghĩa là
   thêm một tầng image, thêm một toolchain, và một image đã 7,35 GB (`TD-57`).
2. `serving/Dockerfile` `COPY` cây nguồn Python. Một file HTML đi cùng nó miễn
   phí; một thư mục `dist/` cần một stage riêng và một chỗ để hỏng lặng lẽ.
3. Toàn bộ giao diện là **một** màn hình chat. Chi phí của framework ở đây không
   đổi lại được gì.

## ⭐⭐ Vì sao đọc file mỗi lần thay vì `StaticFiles`

`StaticFiles` mount cả một thư mục. Ở đây có đúng một tệp, và mount một thư mục
để phục vụ một tệp là mở sẵn chỗ cho tệp thứ hai vô tình lọt vào ảnh (ghi chú,
bản nháp, một `.env.example` copy nhầm). Một route đọc **một đường dẫn hằng** thì
không có chế độ hỏng ấy.

Nội dung được đọc **một lần** lúc import: nó nằm trong image, không đổi lúc chạy,
và đọc đĩa mỗi request là I/O đồng bộ trên vòng lặp sự kiện.

## ⭐⭐ CSP, và vì sao nó chặt đến mức này

Trang này hiển thị **hai** nguồn nội dung không tin được: văn bản chunk corpus
(`W4-12` gắn cờ nó ngay trong khung `sources`) và câu trả lời của model. Trang
không bao giờ dùng `innerHTML` — nhưng "không bao giờ" là một lời hứa của mã,
còn CSP là một hàng rào của trình duyệt, và hai thứ ấy hỏng theo hai cách khác
nhau.

`script-src 'unsafe-inline'` bắt buộc phải có vì script nằm inline (xem trên).
Đổi lại: `default-src 'none'` nên không tải được gì từ bên ngoài, `connect-src
'self'` nên không gửi được gì đi đâu khác, và `frame-ancestors 'none'` chặn
clickjacking lên đúng cái nút gửi câu hỏi.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

__all__ = ["INDEX_PATH", "router"]

router = APIRouter(tags=["ui"])

INDEX_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"

_INDEX = INDEX_PATH.read_text(encoding="utf-8")

#: Xem docstring module. `style-src` cần `'unsafe-inline'` cùng lý do `script-src`.
CSP = (
    "default-src 'none'; "
    "script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src 'self' data:; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Trang chat. Công khai — nội dung tĩnh, mọi lời gọi API vẫn cần khoá."""
    return HTMLResponse(
        _INDEX,
        headers={
            "Content-Security-Policy": CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            # Trang thay đổi theo mỗi lần deploy và nó nhỏ. `no-cache` để một
            # bản vá giao diện không bị một trình duyệt giữ lại vô thời hạn —
            # chế độ hỏng ấy tốn hàng giờ để chẩn đoán vì nó chỉ xảy ra với
            # người đã mở trang trước đó.
            "Cache-Control": "no-cache",
        },
    )
