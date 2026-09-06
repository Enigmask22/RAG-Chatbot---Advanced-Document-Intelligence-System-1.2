"""`/health` và `/ready` — `W4-03`.

## ⭐⭐ Hai endpoint này **không** là hai tên gọi của một thứ

Đây là chỗ dễ làm sai nhất của cả hạng mục, vì làm sai không lộ ra cho tới lần
sự cố đầu tiên — và lúc đó nó *khuếch đại* sự cố.

| | `/health` (liveness) | `/ready` (readiness) |
|---|---|---|
| câu hỏi | "tiến trình này còn cứu được không?" | "gửi traffic vào đây được chưa?" |
| trả lời sai ⇒ | **khởi động lại container** | rút khỏi load balancer |
| được phép chạm phụ thuộc | **không** | có |

⚠️ **Vì sao `/health` tuyệt đối không được hỏi Qdrant.** Giả sử nó có hỏi. Qdrant
chớp 30 giây → mọi replica trả 503 ở `/health` → orchestrator **khởi động lại
toàn bộ chúng cùng lúc**. Việc khởi động lại không chữa được Qdrant, mà mỗi
replica mới lại mất vài phút nạp 3,3 GB trọng số bge-m3. Một trục trặc 30 giây
của phụ thuộc trở thành một sự cố nhiều phút của chính mình, và cái gây ra nó là
phép thử sức khoẻ.

Nên `/health` chỉ trả lời được đúng một điều — vòng lặp sự kiện còn nhận và trả
được một request — và điều đó đã đủ đúng với việc nó điều khiển.

## `/ready` trả **cùng một thân JSON** cho cả 200 và 503

Người vận hành gõ `curl` khi mọi thứ đang hỏng. Một 503 rỗng buộc họ đi tìm log
của đúng pod ấy; một 503 kèm bảng phụ thuộc trả lời ngay tại chỗ.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from serving.core.metrics import CONTENT_TYPE_LATEST
from serving.core.probes import ReadinessProbes
from serving.core.registry import BundleRegistry

__all__ = ["router"]

logger = logging.getLogger(__name__)


def _worker_count() -> int:
    """Số worker mà bản phơi bày này đại diện — `TD-75`.

    Đọc `WEB_CONCURRENCY`, **cùng biến** mà uvicorn dùng để quyết số worker (xem
    `serving/Dockerfile`). Hai chỗ khai riêng thì gauge sẽ khai một giả định
    không còn đúng, và một gauge nói dối về giả định của mình còn tệ hơn không
    có gauge nào.
    """
    raw = os.environ.get("WEB_CONCURRENCY", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("WEB_CONCURRENCY=%r không phải số — báo 1 worker", raw)
        return 1


router = APIRouter(tags=["health"])


# ⚠️ Phải khai ở **tầng module**, không phải trong một factory: `from __future__
# import annotations` biến annotation thành chuỗi và FastAPI phân giải chúng
# trong không gian tên của module. Một alias `Annotated[...]` định nghĩa bên
# trong hàm sẽ không tìm thấy, và FastAPI im lặng coi tham số là query param —
# đúng cái bẫy đã ghi ở `pipeline/ingest/app.py`.
def get_registry(request: Request) -> BundleRegistry:
    registry: BundleRegistry = request.app.state.registry
    return registry


def get_probes(request: Request) -> ReadinessProbes:
    probes: ReadinessProbes = request.app.state.probes
    return probes


RegistryDep = Annotated[BundleRegistry, Depends(get_registry)]
ProbesDep = Annotated[ReadinessProbes, Depends(get_probes)]


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness. Không chạm gì. Xem bảng ở đầu module trước khi thêm bất cứ gì."""
    return {"status": "alive"}


@router.get("/ready")
async def ready(response: Response, registry: RegistryDep, probes: ProbesDep) -> dict[str, Any]:
    """Readiness: có bundle đang phục vụ **và** mọi phụ thuộc trả lời được.

    Thứ tự có ý nghĩa: hỏi bundle **trước**, và chỉ chạy probe khi đã có bundle.
    Chưa nạp bundle nghĩa là chưa có gì để phục vụ, nên gửi thêm một lượt truy
    vấn vào Qdrant chỉ để cũng trả 503 là tải thừa — nhân với mọi replica đang
    khởi động cùng lúc sau một lần deploy.
    """
    checks: dict[str, Any] = {
        "bundle": {"ok": registry.is_ready, "detail": None if registry.is_ready else "chưa nạp"}
    }
    if registry.is_ready:
        for result in await probes.run():
            checks[result.name] = result.as_json()

    ok = all(entry["ok"] for entry in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ok, "checks": checks, **registry.status()}


@router.get("/metrics")
def metrics(request: Request) -> Response:
    """Bản phơi bày Prometheus — `W5-07`.

    ## ⭐⭐ Endpoint này **có** xác thực, và đó là lựa chọn ngược quy ước

    Quy ước Prometheus là `/metrics` mở, vì nó thường nằm trên một cổng nội bộ
    mà chỉ scraper với tới được. Ở đây không có cổng ấy: cùng một tiến trình,
    cùng một cổng 8000, và `W4-13` publish nó ra `127.0.0.1:8000`.

    Nên `/metrics` **không** nằm trong `PUBLIC_PATHS`. Nó cần một khoá hợp lệ —
    và `AuthMiddleware` cho nó đúng thế, không hơn: đường dẫn không bắt đầu
    bằng `/admin` nên không đòi scope admin, vì scraper là một tiến trình hạ
    tầng chứ không phải một người vận hành.

    ⚠️ Điều kiện đi kèm: **không nhãn nào ở đây được mang tên tenant** — xem
    docstring `serving/core/metrics.py`. Một khoá hợp lệ bất kỳ đọc được endpoint
    này, kể cả khoá của tenant khác; nếu nhãn có `tenant` thì mọi khách hàng đọc
    được danh sách khách hàng.

    ⭐ Hai gauge được **làm mới ngay tại đây** thay vì cập nhật lúc chúng đổi.
    Chúng là *trạng thái*, không phải *sự kiện*: phiên bản bundle đang phục vụ
    và độ dài hàng đợi trace đúng ở thời điểm scrape, không đúng ở thời điểm
    một request nào đó tình cờ chạm vào chúng. Với một gauge, "mới nhất lúc
    được hỏi" là định nghĩa đúng.
    """
    bag = getattr(request.app.state, "metrics", None)
    if bag is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "chưa bật số đo")

    registry: BundleRegistry = request.app.state.registry
    version = registry.status().get("active")
    if version:
        bag.bundle.labels(version=version).set(1)

    # ⭐⭐ `TD-75`: `prometheus_client` giữ số đo trong bộ nhớ của MỘT tiến trình.
    # Hôm nay đúng vì chỉ chạy 1 worker (`TD-63`: khoá GPU khiến worker thứ hai
    # không mua thêm thông lượng). Bật worker thứ hai thì mỗi worker có sổ riêng
    # và scraper hỏi trúng một cái ngẫu nhiên — bảng tụt đi một nửa mà không có
    # gì báo, đúng họ "sai theo hướng có lợi".
    #
    # Bản vá đúng là `PROMETHEUS_MULTIPROC_DIR`. Bản vá ở đây rẻ hơn và giải
    # quyết phần nguy hiểm hơn: nó bắt bản phơi bày **tự nói ra giả định của
    # mình**. Một bảng đọc `rag_scrape_workers > 1` biết ngay mọi con số bên
    # dưới đang bị chia; không có dòng này thì không ai biết gì cả.
    bag.scrape_workers.set(_worker_count())

    sink = getattr(request.app.state, "trace_sink", None)
    status_of = getattr(sink, "status", None)
    if callable(status_of):
        for key, value in status_of().items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bag.trace_sink.labels(state=key).set(value)

    return Response(content=bag.render(), media_type=CONTENT_TYPE_LATEST)
