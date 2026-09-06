"""Cầu nối tới API ingestion (`W3-08`) cho giao diện — `W6-01`.

## ⭐⭐ DoD của `W6-01` viết "upload progress", và hệ thống này **không có upload**

`pipeline/ingest/schemas.py:IngestRequest` nhận một **tên config**, và docstring
ở đó nói thẳng vì sao nó không nhận đường dẫn: nhận đường dẫn thì `../../.env`
hay một YAML bất kỳ trên đĩa đều đi qua được. Không có endpoint nào nhận file.

Và đó không phải một thiếu sót cần vá ở đây. Quy tắc cứng của dự án là **corpus
phải công khai, license cho phép redistribute** (`pipeline/corpus/` cưỡng chế
bằng manifest + license + DVC). Một nút "tải tài liệu lên" mở đúng con đường mà
luật ấy sinh ra để đóng: tài liệu không rõ nguồn đi thẳng vào index, rồi vào
prompt, rồi vào một câu trả lời có trích dẫn.

Nên cái dựng được **một cách trung thực** là tiến độ của *job ingest*: chạy lại
một config đã có, và xem `documents_done / documents_total` chạy. Đó là phần
"progress" thật; phần "upload" cần một quyết định về license mà không hạng mục
nào của `W6` đã lấy.

## ⭐ Vì sao **proxy**, không để trình duyệt gọi thẳng cổng 8001

`AU-10`: API ingestion **không có auth**, và biện pháp giảm nhẹ hiện tại là bind
`127.0.0.1`. Cho UI gọi thẳng nó sẽ (a) buộc mở CORS trên một dịch vụ không xác
thực, (b) buộc nó rời khỏi loopback. Đi vòng qua `/admin/ingest` thì tầng auth
của `W4-04` áp dụng nguyên vẹn — `ADMIN_PREFIX` che theo **tiền tố đường dẫn**,
nên route này được bảo vệ vì nó *ở trong* `/admin`, không vì ai đó nhớ.

⚠️ Mặc định **tắt** (`INGEST_API_URL` rỗng ⇒ 503 kèm lời giải thích). Một bề
mặt điều khiển pipeline mở sẵn ở mọi lần deploy là thứ không ai xin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from rag_core.settings import Settings

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/ingest", tags=["ingest"])

#: Hạn giờ cho lời gọi sang dịch vụ ingest. Ngắn: nó chỉ xếp job vào hàng đợi
#: hoặc đọc một hàng Redis — cả hai đều dưới một giây khi khoẻ. Dài hơn nghĩa là
#: một dịch vụ ingest treo sẽ giữ luôn worker của API chat.
TIMEOUT_S = 5.0


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated["Settings", Depends(get_settings)]


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str = Field(min_length=1, max_length=63)
    """Tên config, **không** phải đường dẫn — `pipeline.ingest.schemas` từ chối
    mọi thứ có dấu phân cách, và giới hạn ấy phải giữ nguyên khi đi qua proxy."""

    doc_ids: tuple[str, ...] = ()
    recreate: bool = False


def _base_url(settings: Settings) -> str:
    url = (settings.ingest_api_url or "").strip().rstrip("/")
    if not url:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "chưa cấu hình INGEST_API_URL — bảng tiến độ ingest đang tắt. "
            "Đây là mặc định có chủ đích: xem docstring serving/api/ingest.py.",
        )
    return url


async def _call(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.request(method, url, json=payload)
    except httpx.HTTPError as exc:
        # ⚠️ **Không** dội nguyên văn lỗi ra client: `AU-03` (`NEW-08`) đã vá
        # đúng chế độ này ở đường chat — thân lỗi của một dịch vụ nội bộ mang
        # host, cổng, và đôi khi cả header đi kèm.
        logger.warning("gọi dịch vụ ingest thất bại: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "không gọi được dịch vụ ingest") from exc
    if response.status_code >= 400:
        logger.warning("dịch vụ ingest trả %d: %s", response.status_code, response.text[:300])
        raise HTTPException(response.status_code, "dịch vụ ingest từ chối yêu cầu")
    return response.json()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start(body: StartRequest, settings: SettingsDep) -> Any:
    """Xếp một job re-index vào hàng đợi. Trả về `job_id` để hỏi tiến độ."""
    return await _call(
        "POST",
        f"{_base_url(settings)}/ingest",
        {"config": body.config, "doc_ids": list(body.doc_ids), "recreate": body.recreate},
    )


@router.get("/{job_id}")
async def progress(job_id: str, settings: SettingsDep) -> Any:
    """Tiến độ một job: `documents_done / documents_total`, `chunks_embedded`."""
    return await _call("GET", f"{_base_url(settings)}/ingest/{job_id}")
