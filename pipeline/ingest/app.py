"""API điều khiển ingestion. `W3-08`.

    uv run uvicorn pipeline.ingest.app:app --port 8001

DoD: `POST /ingest` trả `job_id` **< 200 ms**. Cách duy nhất giữ được lời hứa đó
là endpoint **không làm gì** ngoài ghi một bản trạng thái và đẩy job vào hàng
đợi — hai lượt Redis cục bộ. Mọi thứ nặng (nạp model, đọc corpus, embed) nằm ở
worker.

⚠️ Module này **không** import `pipeline.indexing` ở tầng module, và đó không
phải chuyện phong cách: `build_index` kéo theo torch + qdrant-client, tức API sẽ
mất vài giây để khởi động và giữ 2 GB RAM cho một tiến trình chỉ đẩy job.
`tasks.py` import chúng bên trong thân hàm, và hàm đó chỉ chạy ở worker.
"""

from __future__ import annotations

import hmac
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status

from rag_core.settings import get_settings

from .schemas import IngestRequest, JobState, JobStatus, resolve_config
from .store import JobStore
from .tasks import INGEST_TASK

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


class QueuedJob(JobStatus):
    """`JobStatus` kèm tiến độ đã tính sẵn — `progress` là property nên nó không
    tự vào JSON, và client không nên phải tự chia hai số."""

    progress_ratio: float = 0.0

    @classmethod
    def of(cls, source: JobStatus) -> QueuedJob:
        return cls(**source.model_dump(), progress_ratio=round(source.progress, 4))


def get_store(request: Request) -> JobStore:
    """Lấy store từ `app.state`, không đóng gói qua closure.

    ⚠️ Phải khai ở **tầng module**: `from __future__ import annotations` biến mọi
    annotation thành chuỗi, và FastAPI phân giải chúng trong không gian tên của
    module. Một alias `Annotated[...]` định nghĩa bên trong `create_app` sẽ không
    tìm thấy — FastAPI khi đó coi tham số là **query param** và trả 422 "Field
    required: store". Không có lỗi import nào, không có gợi ý nào; chỉ là mọi
    endpoint đột nhiên đòi thêm một tham số truy vấn.
    """
    return JobStore(request.app.state.pool)


StoreDep = Annotated[JobStore, Depends(get_store)]

LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def guard(request: Request) -> None:
    """Ai được gọi dịch vụ này — `AU-10`, vá ở `W6-06`.

    ## ⭐⭐ Biện pháp giảm nhẹ cũ là một **quy ước triển khai**, không phải một bất biến

    `AU-10` ghi: "`POST /ingest` nhận cả `recreate=true` (xoá collection). Giảm
    nhẹ hiện tại: bind `127.0.0.1`, Docker không expose." Cả hai vế ấy đúng, và
    cả hai đều nằm **ngoài** mã: chúng là một cờ dòng lệnh và một dòng YAML. Một
    `--host 0.0.0.0` gõ vội trong lúc gỡ lỗi xoá sạch chúng, không để lại gì
    trong diff và không có gì đỏ.

    Nên `W6-06` biến quy ước ấy thành thứ **mã tự kiểm**:

    1. `INGEST_API_TOKEN` có đặt ⇒ mọi route (trừ `/healthz`) đòi đúng token ấy
       ở `Authorization: Bearer …`. Đây là đường cho một triển khai thật, nơi
       proxy `/admin/ingest` của Serving Plane là client duy nhất.
    2. Không đặt ⇒ **chỉ loopback gọi được**. Máy dev không phải cấu hình gì
       thêm, mà một lần bind ra ngoài cũng không mở được cửa.

    ⭐ So sánh token bằng `hmac.compare_digest`, khác `ApiKeyStore` — ở đó digest
    là *khoá tra dict* nên không có phép so nào chạy trên bí mật (xem docstring
    `serving.core.auth`); ở đây thì có đúng một token và phép so là so chuỗi
    thật, tức thời gian so **là** một kênh phụ.

    ⚠️ `/healthz` để ngoài, cùng lý lẽ `/health` của Serving Plane: bắt probe
    mang credential là đưa credential vào manifest deploy của mọi môi trường.
    """
    settings = get_settings()
    token = settings.ingest_api_token
    if token is not None:
        header = request.headers.get("authorization", "")
        prefix = "bearer "
        supplied = header[len(prefix) :].strip() if header.lower().startswith(prefix) else ""
        if not supplied or not hmac.compare_digest(supplied, token.get_secret_value()):
            logger.warning("401 %s %s", request.method, request.url.path)
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "cần `Authorization: Bearer …` khớp INGEST_API_TOKEN",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return
    host = request.client.host if request.client else None
    if host not in LOOPBACK:
        logger.warning(
            "403 %s %s từ %s (không có INGEST_API_TOKEN)",
            request.method,
            request.url.path,
            host,
        )
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "dịch vụ ingest chưa cấu hình INGEST_API_TOKEN nên chỉ nhận lời gọi "
            "từ loopback. Đặt token nếu cần gọi từ máy khác.",
        )


GuardDep = Annotated[None, Depends(guard)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from arq import create_pool

    from .worker import queue_name, redis_settings

    app.state.pool = await create_pool(redis_settings(), default_queue_name=queue_name())
    try:
        yield
    finally:
        await app.state.pool.aclose()


def create_app() -> FastAPI:
    api = FastAPI(
        title="RAG ingestion control",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
    )

    @api.get("/healthz")
    async def healthz() -> dict[str, str]:
        await api.state.pool.ping()
        return {"status": "ok"}

    @api.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
    async def enqueue(request: IngestRequest, store: StoreDep, _: GuardDep) -> QueuedJob:
        # Kiểm config **ở đây**, không để worker phát hiện: một tên config sai là
        # lỗi của người gọi và họ phải biết ngay, chứ không phải nhận `job_id`
        # rồi ba giây sau thấy job FAILED.
        try:
            resolve_config(request.config)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        job_id = uuid.uuid4().hex
        queued = JobStatus(
            job_id=job_id,
            state=JobState.QUEUED,
            config=request.config,
            queued_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        # Ghi trạng thái TRƯỚC khi đẩy job. Ngược lại thì worker có thể nhận job
        # và gọi `patch` vào một bản ghi chưa tồn tại — `patch` trả `None` và mọi
        # tiến độ của job đó biến mất, im lặng. Đổi lại, `enqueue_job` hỏng sẽ để
        # lại một bản ghi QUEUED mồ côi; đó là cái giá rẻ hơn.
        await store.put(queued)

        job: Any = await api.state.pool.enqueue_job(
            INGEST_TASK, request.model_dump(mode="json"), _job_id=job_id
        )
        if job is None:  # pragma: no cover - chỉ xảy ra khi `_job_id` trùng
            raise HTTPException(status.HTTP_409_CONFLICT, f"job {job_id} đã tồn tại")
        return QueuedJob.of(queued)

    @api.get("/ingest/{job_id}")
    async def get_job(job_id: str, store: StoreDep, _: GuardDep) -> QueuedJob:
        found = await store.get(job_id)
        if found is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"không có job {job_id} (đã quá 24 giờ, hoặc chưa từng tồn tại)",
            )
        return QueuedJob.of(found)

    return api


app = create_app()
