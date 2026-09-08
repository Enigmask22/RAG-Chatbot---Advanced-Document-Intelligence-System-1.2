"""Xác thực + giới hạn nhịp, dạng ASGI middleware — `W4-04`.

## ⭐⭐ Vì sao middleware chứ không `Depends(...)` trên từng route

`Depends` là cách mọi ví dụ FastAPI làm, và nó hỏng theo **hướng sai**: quên gắn
dependency vào một route mới thì route đó **công khai**, im lặng, và không có
gì trong diff trông bất thường — chỉ là một endpoint mới không có tham số thừa.

Middleware chặn theo mặc định thì hỏng theo hướng ngược lại: quên nghĩ về một
route mới nghĩa là nó **bị khoá**, và điều đó lộ ra ngay ở lần gọi thử đầu tiên.
Muốn mở thì phải viết tên nó vào `PUBLIC_PATHS` — một dòng nhìn thấy được trong
review, thay vì một dòng *không* được viết.

Cùng lý lẽ cho scope admin: quy tắc là **tiền tố đường dẫn** (`/admin/**` cần
`ADMIN_SCOPE`), không phải một dependency gắn tay từng route. Một route admin mới
được bảo vệ vì nó *ở trong* `/admin`, không vì ai đó nhớ.

## `/health` và `/ready` phải công khai — và đó là quyết định bảo mật có cân nhắc

Bắt probe mang credential thì credential ấy nằm trong manifest deploy của mọi
môi trường, xoay vòng được cùng lúc với việc restart toàn cụm, và ai đọc được
manifest thì gọi được API. Đổi lại, để chúng công khai làm lộ **hai bit**: tiến
trình sống, và nó đã nạp bundle chưa. `/ready` có trả tên phiên bản bundle —
chấp nhận được với một dịch vụ nội bộ sau LB, và phải xem lại nếu nó ra Internet.

⚠️ `/docs` và `/openapi.json` **không** công khai: chúng mô tả toàn bộ bề mặt
tấn công, và đó là thứ đắt hơn nhiều so với hai bit trên.

## ⚠️ Request chưa xác thực KHÔNG bị giới hạn nhịp

Giới hạn theo tenant, mà tenant chỉ biết được **sau** khi xác thực — nên một
trận lũ request không key đi qua tự do. Đó không phải sơ suất: một lượt từ chối
tốn đúng một SHA-256 và một phép tra dict, nên thứ bị tiêu thụ là *kết nối*, chứ
không phải tài nguyên của ứng dụng. Chỗ đúng để chặn theo IP là reverse proxy
phía trước, nơi có sẵn thông tin kết nối thật; làm ở đây thì mỗi replica đếm
riêng và `X-Forwarded-For` giả được. Xem `TD-39`.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from serving.core.auth import ADMIN_SCOPE, ApiKeyStore, Principal, key_hint
from serving.core.ratelimit import RateLimiter

__all__ = ["ADMIN_PREFIX", "PUBLIC_PATHS", "AuthMiddleware", "principal_of"]

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({"/", "/health", "/ready"})
"""Danh sách **đóng**, và có test đếm lại nó với `app.routes`.

Cố ý là tập đường dẫn chính xác chứ không phải tiền tố: một tiền tố `/health`
cũng mở luôn `/health-internal-debug` mà không ai nhận ra.

⭐ `/` (`W6-01`) là **trang tĩnh**, không phải một endpoint dữ liệu: nó không đọc
bundle, không chạm Postgres, không biết gì về tenant. Mọi lời gọi nó phát ra
(`/chat`, `/feedback`, `/admin/ingest`) vẫn cần khoá như trước — trang chỉ là
chỗ người dùng gõ khoá vào. Bắt nó xác thực sẽ tạo ra một bài toán con gà–quả
trứng: không có giao diện để nhập khoá thì không có cách nhập khoá.
"""

ADMIN_PREFIX = "/admin"

_BEARER = "bearer "


def principal_of(request: Any) -> Principal:
    """Lấy principal mà middleware đã đặt.

    Thiếu = middleware chưa chạy, tức lỗi lập trình chứ không phải lỗi của người
    gọi — nên nó nổ chứ không trả 401. Một 401 ở đây sẽ che mất việc app đã được
    lắp sai.
    """
    principal = getattr(request.state, "principal", None)
    if principal is None:  # pragma: no cover - chỉ xảy ra khi lắp app sai
        raise RuntimeError(
            "không có principal trong request.state — `AuthMiddleware` chưa được gắn"
        )
    return principal  # type: ignore[no-any-return]


class AuthMiddleware:
    """Chặn theo mặc định, rồi tính hạn mức theo tenant."""

    def __init__(self, app: Any, *, keys: ApiKeyStore, limiter: RateLimiter) -> None:
        self.app = app
        self.keys = keys
        self.limiter = limiter

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        raw_key = _bearer_token(scope)
        principal = self.keys.lookup(raw_key) if raw_key else None
        if principal is None:
            # Cùng một phản hồi cho "không có key" và "key sai": phân biệt hai
            # ca đó cho người gọi biết một chuỗi bất kỳ **có phải** key hợp lệ
            # hay không, tức biến endpoint thành máy kiểm key.
            logger.warning(
                "401 %s %s (%s)",
                scope.get("method"),
                scope.get("path"),
                key_hint(raw_key) if raw_key else "không có key",
            )
            await _deny(send, 401, "cần API key hợp lệ ở header `Authorization: Bearer …`")
            return

        if scope["path"].startswith(ADMIN_PREFIX) and not principal.is_admin:
            # 403 chứ không 401: key hợp lệ, quyền thì không — và bảo họ thử
            # credential khác là hướng dẫn sai.
            logger.warning("403 %s thiếu scope %s", principal.key_id, ADMIN_SCOPE)
            await _deny(send, 403, f"key này không có scope {ADMIN_SCOPE!r}")
            return

        # ⭐ `TD-39`: `RateLimiter.check` đồng bộ, `RedisRateLimiter.check` thì
        # không — bộ đếm dùng chung phải đi qua mạng. Nhận **cả hai** thay vì
        # bắt mọi chỗ gọi đổi kiểu: 3 dòng ở đây rẻ hơn một lần đổi chữ ký lan
        # qua hàng chục fixture, và biên async/sync là chi tiết triển khai của
        # bộ đếm chứ không phải của hợp đồng "hỏi rồi cho qua hay chặn".
        ket_qua = self.limiter.check(principal.tenant_id, principal.rate_limit_per_minute)
        decision = await ket_qua if inspect.isawaitable(ket_qua) else ket_qua
        if not decision.allowed:
            logger.warning("429 tenant %s vượt %d/phút", principal.tenant_id, decision.limit)
            await _deny(
                send,
                429,
                f"vượt hạn mức {decision.limit} request/phút cho tenant {principal.tenant_id!r}",
                extra_headers=decision.headers(),
            )
            return

        scope.setdefault("state", {})["principal"] = principal
        await self.app(scope, receive, _with_headers(send, decision.headers()))


def _bearer_token(scope: Any) -> str | None:
    for name, value in scope.get("headers", []):
        if name != b"authorization":
            continue
        text = value.decode("latin-1")
        if text.lower().startswith(_BEARER):
            return str(text[len(_BEARER) :].strip())
        return None
    return None


def _with_headers(send: Any, headers: dict[str, str]) -> Any:
    async def wrapped(message: Any) -> None:
        if message["type"] == "http.response.start":
            message["headers"] = [
                *message.get("headers", []),
                *((k.lower().encode("ascii"), v.encode("ascii")) for k, v in headers.items()),
            ]
        await send(message)

    return wrapped


async def _deny(
    send: Any, status: int, detail: str, *, extra_headers: dict[str, str] | None = None
) -> None:
    body = json.dumps({"detail": detail}, ensure_ascii=False).encode("utf-8")
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if status == 401:
        # RFC 7235: 401 **phải** kèm `WWW-Authenticate`, nếu không thì client
        # không biết cơ chế nào đang được đòi.
        headers.append((b"www-authenticate", b"Bearer"))
    for key, value in (extra_headers or {}).items():
        headers.append((key.lower().encode("ascii"), value.encode("ascii")))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
