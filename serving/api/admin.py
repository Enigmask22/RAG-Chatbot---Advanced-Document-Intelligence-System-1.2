"""Điều khiển bundle lúc đang chạy — `W4-03`, mở khoá lõi của `W4-02`.

## 🔒 Xác thực: có, và nó đến từ **tiền tố đường dẫn**

Ba route dưới đây đổi được **hệ thống đang phục vụ production**. Chúng nằm dưới
`/admin/bundle`, nên `AuthMiddleware` (`W4-04`) đòi `ADMIN_SCOPE` cho cả ba —
không phải vì file này khai gì, mà vì quy tắc là tiền tố. Một route admin thứ tư
thêm vào đây được bảo vệ ngay khi nó ra đời.

⚠️⚠️ **Đoạn trên từng viết ngược lại, và `W6-06` bắt được.** Nguyên văn cũ:
*"🔓 CHƯA CÓ XÁC THỰC … hiện ai gọi cũng được"*, kèm lời hứa rằng một test tên
`test_admin_routes_are_still_open` sẽ đỏ khi `W4-04` gắn auth. `W4-04` đã gắn
auth **và** đã xoá test ấy — nhưng không ai xoá đoạn văn. Nó sống thêm hai tuần
như một tuyên bố sai về tình trạng bảo mật của chính file mình, ở đúng chỗ người
đọc tin nhất.

Bài học không phải "nhớ sửa docstring". Nó là: một dòng chữ mô tả **trạng thái**
(khác với mô tả *ý định*) là một bản sao thứ hai của sự thật, và bản sao ấy
không có gì bắt nó đồng bộ — cùng họ `AU-12`. Chỗ duy nhất trả lời được câu
"route này có cần scope không" là `PUBLIC_PATHS` + `ADMIN_PREFIX`, và `W6-06`
thêm một test đọc **bảng route thật** để so với chúng.

## Vì sao ba hàm này là `def` chứ không `async def`

`BundleRegistry.activate()` **chặn**: nó đọc đĩa, nạp trọng số, mở kết nối
Qdrant — hàng chục giây với một cross-encoder chưa nằm trong cache. Viết
`async def` thì toàn bộ khoảng đó vòng lặp sự kiện **không chạy gì khác**, tức
`/health` cũng không trả lời — và orchestrator đọc đúng điều đó là "tiến trình
chết", rồi giết nó **giữa lúc đang reload**.

FastAPI chạy handler đồng bộ trong threadpool, nên `def` giữ vòng lặp rảnh.
`anyio` sao chép context sang luồng đó nên `request_id` vẫn theo được.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from rag_core.bundle import BundleValidationError
from serving.api.health import RegistryDep
from serving.core.registry import ActiveBundle, BundleRegistry, NothingToRollBackError
from serving.core.runtime import BundleRuntimeError, drift_of

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/bundle", tags=["admin"])


class ReloadRequest(BaseModel):
    version: str = Field(min_length=1, examples=["0.1.0"])


def _describe(active: ActiveBundle) -> dict[str, Any]:
    components = active.bundle.components
    return {
        "version": active.version,
        "loaded_at": active.loaded_at.isoformat(),
        "collection": components.index.collection,
        "retrieval_mode": components.retrieval.mode,
        "rerank_model": components.rerank.model if components.rerank else None,
        "serves_generation": active.bundle.serves_generation,
        "retriever_name": components.retriever_name,
        # ⚠️ Khác `None` nghĩa là bundle đang chạy **không** phải hệ thống đã đo,
        # và ai đó đã bật `BUNDLE_ALLOW_RUNTIME_DRIFT` để cho qua. Đặt ở đây chứ
        # không chỉ trong log: một cửa thoát im lặng mới là cửa thoát nguy hiểm.
        "runtime_drift": drift_of(active.version),
    }


@router.get("")
def show(registry: RegistryDep) -> dict[str, Any]:
    """Đang chạy bản nào, lùi được về đâu. Một lệnh `curl`, không cần vào log."""
    payload: dict[str, Any] = dict(registry.status())
    payload["active_detail"] = _describe(registry.active) if registry.is_ready else None
    return payload


@router.post("/reload")
def reload(body: ReloadRequest, registry: RegistryDep) -> dict[str, Any]:
    """Đổi sang bundle khác mà không restart tiến trình.

    Mọi nhánh lỗi ở đây đều nói **bản cũ vẫn đang phục vụ**. Đó là câu hỏi đầu
    tiên của người vận hành khi thấy một lệnh reload đỏ, và luật 1 của `W4-02`
    làm cho câu trả lời luôn là "còn" — nhưng chỉ khi phản hồi nói ra.
    """
    try:
        active = registry.activate(body.version)
    except FileNotFoundError as exc:
        raise _refused(status.HTTP_404_NOT_FOUND, registry, exc) from exc
    except (BundleValidationError, ValueError) as exc:
        # Manifest sai hoặc chữ ký không khớp: lỗi của **bundle**, sửa ở phía
        # đóng gói. 422 chứ không 400 để nó phân biệt được với một thân request sai.
        raise _refused(status.HTTP_422_UNPROCESSABLE_CONTENT, registry, exc) from exc
    except BundleRuntimeError as exc:
        # Bundle đúng nhưng **máy này** không dựng được nó (lệch số chiều, số điểm
        # không khớp, schema collection sai). 409 chứ không 503: 503 nghĩa là "thử
        # lại sau", mà thử lại y nguyên sẽ hỏng y nguyên.
        raise _refused(status.HTTP_409_CONFLICT, registry, exc) from exc
    except Exception as exc:
        # ⭐ Nhánh này do **lần chạy thật đầu tiên** thêm vào, không do suy luận.
        # Qdrant tắt ⇒ `verify_schema` ném `ResponseHandlingException("timed
        # out")` — không phải `BundleRuntimeError`, không phải `ValueError` — nên
        # nó xuyên qua mọi `except` ở trên, thành 500 `{"detail": "Lỗi nội bộ"}`
        # của middleware. Người vận hành nhận đúng thứ vô dụng nhất: không lý do,
        # và **không** có câu "bản cũ vẫn đang phục vụ".
        #
        # 503 chứ không 409: lỗi đã phân loại được thì đã bị bắt ở trên rồi, nên
        # tới đây là "chưa biết", và "chưa biết" thường là hạ tầng — tức có thể
        # thử lại.
        raise _refused(status.HTTP_503_SERVICE_UNAVAILABLE, registry, exc) from exc
    logger.warning(
        "đã đổi bundle sang %s", active.version, extra={"bundle_version": active.version}
    )
    return _describe(active)


@router.post("/rollback")
def rollback(registry: RegistryDep) -> dict[str, Any]:
    """Quay về bản trước. Không dựng lại gì nên không hỏng được (`W4-02` §5).

    ⚠️ Chỉ đổi trạng thái **trong bộ nhớ**. Restart container sau một lần rollback
    sẽ quay lại đúng bản vừa bị lùi khỏi, trừ khi `BUNDLE_VERSION` được ghim lại
    trong cấu hình deploy. Phản hồi nhắc điều đó vì đây là lúc dễ quên nhất.
    """
    try:
        active = registry.rollback()
    except NothingToRollBackError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    payload = _describe(active)
    payload["warning"] = (
        "rollback chỉ sống trong bộ nhớ tiến trình — ghim `BUNDLE_VERSION="
        f"{active.version}` trong cấu hình deploy, nếu không restart sẽ quay lại bản cũ."
    )
    return payload


def _refused(code: int, registry: BundleRegistry, exc: Exception) -> HTTPException:
    still = registry.status()["active"]
    logger.warning("reload bị từ chối: %s", exc, exc_info=code >= 500)
    return HTTPException(
        code,
        {
            # Kèm tên lớp exception: lời của nhiều lỗi hạ tầng là một chuỗi trần
            # như `"timed out"`, và một mình nó không nói được **cái gì** hết giờ.
            "detail": f"{type(exc).__name__}: {exc}",
            "still_serving": still,
            "note": "bundle đang phục vụ không bị chạm tới",
        },
    )
