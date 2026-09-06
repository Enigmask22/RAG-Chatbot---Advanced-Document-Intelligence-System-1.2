"""`POST /chat` (SSE) và `GET /conversations/{id}` — `W4-06`.

File này cố ý **mỏng**: nó dịch giữa HTTP và `ChatService`, không chứa logic nào
của một lượt hỏi–đáp. Đường phân giới "còn trả được status" / "chỉ còn khung
SSE" nằm ở `serving/core/chat.py`, và ở đây nó hiện ra thành một điều rất cụ
thể: `await service.prepare(...)` chạy **trước** khi `StreamingResponse` được
tạo, nên mọi exception của nó còn thành `HTTPException` được.

⚠️ Đảo hai dòng đó (đưa `prepare()` vào trong generator "cho gọn") là một thay
đổi trông vô hại, và nó biến mọi lỗi 404/403/503 của hạng mục này thành
`200 OK` kèm một khung lỗi mà client mặc định sẽ bỏ qua. Có test ghim
(`test_a_missing_conversation_is_404_not_a_200_with_an_error_frame`).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, time, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_core.llm import BudgetExceeded
from rag_core.retrieval.filters import MetadataFilter
from serving.api.security import principal_of
from serving.api.sse import SSE_HEADERS, encode
from serving.core.auth import CrossTenantError, Principal
from serving.core.chat import (
    HISTORY_PAGE,
    ChatService,
    ConversationNotFound,
    GenerationUnavailable,
    HistoryCursorNotFound,
)
from serving.core.chat import load_history as _load_history
from serving.core.logging import current_request_id
from serving.core.tracing import Trace

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

MAX_FILTER_VALUES = 100
"""Trần số giá trị cho MỘT field của `filters` — xem `ChatRequest._bound_filter_lists`.

100 chọn theo cách dùng thật: lọc theo `doc_type`/`lang` là vài giá trị, lọc theo
`doc_id` cho một bộ tài liệu là vài chục. Trên mức ấy thì đó không còn là một câu
hỏi của người dùng nữa."""


def _seconds_to_utc_midnight() -> int:
    """Bao nhiêu giây nữa thì ngân sách ngày được nạp lại. Xem `DailyBudget`."""
    now = datetime.now(UTC)
    tomorrow = datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=UTC)
    return max(1, int((tomorrow - now).total_seconds()))


def get_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat
    return service


def get_principal(request: Request) -> Principal:
    return principal_of(request)


# Khai ở tầng module, không trong factory — cùng lý do đã ghi ở `health.py`.
ServiceDep = Annotated[ChatService, Depends(get_service)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    """8000 ký tự ≈ 2–3k token. Không giới hạn thì một `POST` duy nhất tiêu hết
    ngân sách ngày của cả tenant, và hạn mức theo *số request* của `W4-04` không
    thấy điều đó."""

    conversation_id: str | None = Field(default=None, max_length=32)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: MetadataFilter | None = None
    """⚠️ `tenant_id` mà client gửi ở đây **không** được tin: `tenant_filter()`
    ghi đè nó bằng tenant của token, và từ chối nếu hai bên khác nhau."""

    @field_validator("filters")
    @classmethod
    def _bound_filter_lists(cls, value: MetadataFilter | None) -> MetadataFilter | None:
        """⭐⭐ `W6-06`: mọi field của `MetadataFilter` nhận `list[str]` **không
        giới hạn độ dài**, và `message` bị chặn ở 8000 ký tự khiến chỗ ấy trông
        như đã được rào.

        `{"filters": {"chunk_id": [… 200.000 mục …]}}` là một thân request vài
        chục MB, được đọc trọn vào bộ nhớ trước khi Pydantic nhìn tới nó, rồi
        thành một `MatchAny` khổng lồ gửi sang Qdrant. Hạn mức của `W4-04` đếm
        **số request**, nên nó không thấy gì cả — cùng lý lẽ đã đặt trần 8000 ký
        tự cho `message`, chỉ là trần ấy chưa phủ trục thứ hai.

        ⚠️ Chặn ở **đây**, không ở `MetadataFilter`: `rag_core` phục vụ cả đường
        eval, nơi một danh sách dài là hợp lệ (lọc theo cả một tập golden). Chỗ
        phân biệt được "người gọi tin được hay không" là biên HTTP — đúng lý lẽ
        `tenant_filter()` dùng để không nhận `tenant_id` từ người gọi.
        """
        if value is None:
            return None
        for name in ("chunk_id", "doc_id", "tenant_id", "doc_type", "lang"):
            field = getattr(value, name)
            if isinstance(field, list) and len(field) > MAX_FILTER_VALUES:
                raise ValueError(
                    f"filters.{name} có {len(field)} giá trị, trần là {MAX_FILTER_VALUES}"
                )
        return value


@router.post("/chat")
async def chat(
    body: ChatRequest, service: ServiceDep, principal: PrincipalDep
) -> StreamingResponse:
    """Một lượt hỏi–đáp, trả về dạng `text/event-stream`.

    Khung: `meta` → `sources` → `delta`* → (`done` | `error`).

    ⭐ Client **phải** đợi khung kết thúc (`done` hoặc `error`). Một dòng
    `delta` dừng lại không nói được điều gì cả: nó giống hệt nhau khi model nói
    xong, khi kết nối đứt, và khi provider hết hạn mức giữa chừng.
    """
    # ⭐⭐ `W5-06`: trace mở ở đây, **trước** `prepare()`, và đó là điều kiện để
    # nó nhìn thấy phần đáng nhìn nhất.
    #
    # Chỗ hiển nhiên để mở trace là bên trong `ChatService` — nhưng `ChatTurn`
    # chỉ tồn tại ở *dòng cuối* của `prepare()`, tức mọi lượt 404/403/429/503
    # sẽ không có trace nào. Một hệ quan sát phủ đúng những request đã chạy
    # trót lọt là một hệ quan sát nói rằng mọi thứ đều ổn.
    #
    # `X-Request-ID` của `W4-03` đi vào trace: đó là chỗ duy nhất nối được một
    # dòng log JSON với một trace trong Langfuse. Không có nó thì hai hệ thống
    # cùng ghi một sự cố mà không ai ghép được chúng lại.
    trace = Trace(
        name="chat",
        sink=service.sink,
        user_id=principal.tenant_id,
        input=body.message,
        tags=[f"tenant:{principal.tenant_id}"],
        metadata={"request_id": current_request_id(), "top_k": body.top_k},
    )
    try:
        turn = await service.prepare(
            principal,
            question=body.message,
            conversation_id=body.conversation_id,
            top_k=body.top_k,
            filters=body.filters,
            trace=trace,
        )
    except ConversationNotFound as exc:
        trace.finish(level="ERROR", status=f"404 {exc}")
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except BudgetExceeded as exc:
        # ⭐ `429`, không `503`: trần chi phí là một **hạn mức**, và nó hết cho
        # tới nửa đêm UTC chứ không phải "thử lại sau vài giây". `Retry-After`
        # nói ra con số ấy để client không phải đoán — cùng khuôn với hạn mức
        # nhịp của `W4-04`.
        trace.finish(level="ERROR", status=f"429 {exc}")
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            str(exc),
            headers={"Retry-After": str(_seconds_to_utc_midnight())},
        ) from exc
    except CrossTenantError as exc:
        # Lời của lỗi cố ý **không** nhắc tenant mà client đã xin (`W4-04`).
        # ⚠️ Cùng lý do, trace ghi mã lỗi chứ không ghi tenant bị từ chối.
        trace.finish(level="ERROR", status="403 cross-tenant")
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except GenerationUnavailable as exc:
        trace.finish(level="ERROR", status=f"503 {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        # Truy hồi nằm trong `prepare()`, nên Qdrant chết tới được đây — và đây
        # là chỗ **cuối cùng** nó còn biến thành một status đọc được bằng máy.
        # ⚠️ `NEW-08`/`AU-03`: chi tiết exception ở lại server (log + trace).
        # `f"{type(exc).__name__}: {exc}"` trả cho client là trả tên service
        # nội bộ, tên collection, topology — mọi thứ một `ConnectionError` của
        # Qdrant mang theo. Client nhận thông báo chung + `request_id` để đối
        # chiếu với log, theo đúng mẫu `_send_error` của middleware.
        logger.exception("chuẩn bị lượt chat thất bại")
        trace.finish(level="ERROR", status=f"503 {type(exc).__name__}: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "detail": "lỗi tạm thời khi chuẩn bị lượt chat — thử lại sau",
                "request_id": current_request_id(),
            },
        ) from exc

    async def frames() -> AsyncIterator[bytes]:
        async for event in service.stream_turn(turn):
            yield encode(event.event, event.data)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream; charset=utf-8",
        headers={**SSE_HEADERS, "X-Conversation-Id": turn.conversation_id},
    )


@router.get("/admin/llm")
def llm_status(service: ServiceDep) -> dict[str, Any]:
    """Trạng thái bộ định tuyến LLM — `W4-08`.

    Ở file này chứ không ở `admin.py` vì `admin.py` có prefix `/admin/bundle` và
    tài nguyên ở đây là `ChatService`, không phải registry. Scope `admin` vẫn
    được ép **tự động**: `W4-04` kiểm theo tiền tố đường dẫn ở middleware, chứ
    không bằng một dependency gắn tay từng route — nên một route admin mới không
    thể quên hàng rào.

    ⭐ Đây là chỗ duy nhất nhìn thấy cầu dao. Không có nó thì "primary đang bị
    cắt, mọi câu trả lời đến từ nhánh dự phòng" là một trạng thái **không quan
    sát được**: request vẫn 200, câu trả lời vẫn có, chỉ có model khác và hoá
    đơn khác.
    """
    router_obj = getattr(service.llm, "status", None)
    if router_obj is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "chưa cấu hình bộ định tuyến LLM")
    result: dict[str, Any] = router_obj()
    return result


@router.get("/admin/tracing")
def tracing_status(request: Request) -> dict[str, Any]:
    """Trace có **thật sự** tới Langfuse không — `W5-06`.

    ⭐ Không có endpoint này thì quan sát là thứ duy nhất trong hệ thống không
    quan sát được. Hàng đợi đầy, khoá sai, host sai — cả ba đều biểu hiện y hệt
    nhau từ phía `/chat`: 200, câu trả lời đúng, và một bảng Langfuse trống mà
    người ta sẽ đọc là "hôm nay ít traffic".

    `dropped` là con số quan trọng nhất ở đây và nó phải được đọc cùng
    `sent`: trace bị vứt khi hàng đợi đầy, và hàng đợi đầy đúng lúc hệ thống
    bận — nên một `dropped` lớn nghĩa là bảng đang thiếu **đúng** những lượt
    đáng xem nhất, chứ không thiếu ngẫu nhiên.
    """
    sink = getattr(request.app.state, "trace_sink", None)
    if sink is None:
        return {"enabled": False, "reason": "chưa cấu hình LANGFUSE_* — xem build_sink()"}
    status_of = getattr(sink, "status", None)
    if not callable(status_of):
        return {"enabled": True, "reason": "sink không khai status()"}
    result: dict[str, Any] = {"enabled": True, **status_of()}
    return result


@router.get("/conversations/{conversation_id}")
async def conversation(
    conversation_id: str,
    service: ServiceDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = HISTORY_PAGE,
    after: Annotated[str | None, Query(max_length=32)] = None,
) -> dict[str, Any]:
    """Đọc lại lịch sử — nửa thứ hai của DoD ("sống sót qua restart container").

    ⭐ `AU-08` vá ở `W6-06`: có phân trang, con trỏ theo `Message.id`. Xem
    `load_history` cho lý do không dùng `OFFSET`.

    `next_after` khác `None` nghĩa là **còn** trang nữa. Suy ra từ việc trang này
    đầy đúng `limit` — nên lần gọi cuối cùng của một vòng lặp phân trang có thể
    trả về một trang rỗng. Đó là cái giá của việc không chạy thêm một `COUNT(*)`
    trên mỗi trang, và nó được nói ra ở đây thay vì để client tự phát hiện.
    """
    if service.sessions is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "chưa cấu hình Postgres")
    try:
        messages = await _load_history(
            service.sessions, principal, conversation_id, limit=limit, after=after
        )
    except ConversationNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except HistoryCursorNotFound as exc:
        # 422 chứ không 404: hội thoại **có** tồn tại, thứ sai là con trỏ client
        # cầm. Một 404 ở đây bảo họ ngừng hỏi về một hội thoại vẫn đang sống.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "next_after": messages[-1]["id"] if len(messages) == limit else None,
    }
