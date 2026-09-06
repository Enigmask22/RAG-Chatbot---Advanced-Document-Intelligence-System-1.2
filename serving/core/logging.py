"""Log JSON một dòng, mang `request_id`, cho Serving Plane — `W4-03`.

## Vì sao không phải structlog

Kế hoạch ghi "structlog JSON". Đổi sang `logging` chuẩn có lý do đo được, không
phải sở thích: **phần lớn dòng log của một tiến trình serving không do mã của ta
sinh ra**. `uvicorn`, `qdrant-client`, `httpx`, `sentence-transformers` đều ghi
qua `logging` chuẩn. Dùng structlog nghĩa là vẫn phải cấu hình `logging` chuẩn để
bắc cầu chúng sang — tức giữ cả hai, và phần cấu hình cầu nối dài hơn chính cái
formatter nó thay thế. Thứ hạng mục này cần chỉ là hai điều: **mọi dòng là JSON**
và **mỗi dòng mang `request_id`**. Cả hai là một formatter và một `ContextVar`.

`pipeline.ingest.app` và toàn bộ `rag_core` cũng đang dùng `logging` chuẩn, nên
lựa chọn này giữ đúng một cách ghi log trong repo.

## ⚠️ `ensure_ascii=False` bắt buộc phải đi kèm stream UTF-8

Log của dự án này là tiếng Việt. `json.dumps` mặc định `ensure_ascii=True` biến
mỗi chữ có dấu thành `\\uXXXX` — vẫn đúng JSON, nhưng `grep` trên log thành vô
dụng. Bật `ensure_ascii=False` thì chuỗi UTF-8 đi thẳng ra stream, và **trên
Windows stderr mặc định không phải UTF-8**: dòng log đầu tiên có dấu sẽ ném
`UnicodeEncodeError` *bên trong* `logging`, chỗ mà lỗi bị nuốt thành một dòng
"--- Logging error ---" và bản ghi biến mất. Nên `configure_logging` ép lại
encoding của stream trước khi gắn handler.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import IO, Any

from rag_core.generation import RedactingFilter

__all__ = [
    "JsonFormatter",
    "RedactingFilter",
    "bind_request_id",
    "configure_logging",
    "current_request_id",
    "reset_request_id",
]

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Thuộc tính chuẩn của `LogRecord`, để tách ra đâu là `extra` do người gọi thêm.
#:
#: Lấy bằng cách **dựng một record thật** chứ không chép danh sách: Python có
#: thêm thuộc tính theo phiên bản (`taskName` xuất hiện ở 3.12), và một danh sách
#: chép tay sẽ lặng lẽ đẩy thuộc tính mới ấy vào JSON như thể nó là dữ liệu của
#: ứng dụng.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
}


def current_request_id() -> str | None:
    return _request_id.get()


def bind_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Một bản ghi = một dòng JSON.

    `extra=` của người gọi được đưa lên **cùng cấp** với các khoá lõi, không lồng
    dưới một khoá `extra`: truy vấn `level=ERROR AND collection=rag_bgem3_ctx`
    trong log aggregator chỉ viết được khi trường ấy phẳng.

    Khoá lõi ghi **sau** nên nó thắng khi trùng tên. Có chủ đích: một
    `logger.info(..., extra={"level": "ok"})` vô ý không được phép làm hỏng
    trường mà bộ lọc mức độ nghiêm trọng dựa vào.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            key: value for key, value in record.__dict__.items() if key not in _RESERVED
        }
        payload.update(
            {
                "ts": datetime.fromtimestamp(record.created, UTC).isoformat(
                    timespec="milliseconds"
                ),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
        )
        request_id = _request_id.get()
        if request_id is not None:
            payload["request_id"] = request_id
        if record.exc_info:
            # ⭐⭐ `W6-06`: ưu tiên `record.exc_text`. `RedactingFilter` kết xuất
            # traceback rồi che PII/credential vào đúng trường ấy, và gọi thẳng
            # `formatException` ở đây sẽ **đi vòng qua** bản đã che — đúng lỗ mà
            # probe của `W6-06` đo được (email bị che ở `msg`, lọt nguyên vẹn ở
            # traceback). Fallback giữ lại cho trường hợp formatter này được
            # dùng mà không có filter (test, hoặc một handler lắp tay).
            payload["exc"] = record.exc_text or self.formatException(record.exc_info)
        # `default=str` để một `Path`, một `datetime` hay một object bất kỳ lọt
        # vào `extra` không làm chết formatter — một dòng log bị format xấu vẫn
        # tốt hơn một dòng log biến mất. Tiện thể nó cũng an toàn với `SecretStr`:
        # `str()` của nó là `'**********'`.
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, stream: IO[str] | None = None) -> None:
    """Gắn một handler JSON duy nhất lên root, và **gỡ mọi handler có sẵn**.

    Gỡ chứ không thêm: `uvicorn` tự cài `logging.config.dictConfig` của nó khi
    khởi động, và nếu cả hai cùng sống thì mỗi dòng ra **hai** lần — một bản JSON
    và một bản văn xuôi — nên log vừa gấp đôi dung lượng vừa không parse được.

    `uvicorn.access` bị **tắt hẳn** thay vì định dạng lại: dòng access của nó là
    văn xuôi cố định (không đi qua formatter của ta) và nó ghi cả query string.
    Dòng access thay thế do `RequestContextMiddleware` sinh ra — cùng định dạng
    JSON, mang `request_id`, và cố ý **không** ghi query string.
    """
    target = stream if stream is not None else sys.stderr
    reconfigure = getattr(target, "reconfigure", None)
    if reconfigure is not None:  # pragma: no cover - phụ thuộc nền tảng
        # Stream không đổi được encoding (đã bị bọc, hoặc là một `StringIO`
        # trong test) thì bỏ qua — không phải lỗi, `StringIO` vốn giữ `str`.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")

    handler = logging.StreamHandler(target)
    handler.setFormatter(JsonFormatter())
    # ⭐ `W4-12`: che PII gắn lên **handler**, không lên logger.
    #
    # Gắn lên logger thì chỉ bản ghi đi qua logger ấy được che, và phần lớn dòng
    # log của một tiến trình serving không do mã của ta sinh ra (cùng lý lẽ ở
    # §đầu module): `httpx` ghi URL kèm query string, một `logger.exception` in
    # nguyên payload provider. Gắn lên handler thì **mọi** bản ghi tới được chỗ
    # ghi ra đều đã đi qua nó — kể cả của thư viện chưa ai nghĩ tới.
    handler.addFilter(RedactingFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
