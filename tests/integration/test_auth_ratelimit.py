"""`W4-04` — xác thực và hạn mức nhịp ở tầng HTTP.

Nhóm 2 là nhóm đáng giá nhất: **mọi route hoặc công khai có chủ đích, hoặc bị
khoá**. Nó không kiểm một hành vi cụ thể nào mà kiểm rằng *không tồn tại* một
route nào lọt ra ngoài — và đó đúng là loại lỗi mà `Depends` từng route để lọt
mà không có gì trong diff trông bất thường.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_core.bundle import RagBundle
from rag_core.settings import Settings
from serving.api.app import create_app
from serving.api.security import PUBLIC_PATHS
from serving.core.auth import digest_of
from serving.core.probes import ReadinessProbes
from serving.core.registry import BundleRegistry
from tests.integration.test_bundle_reload import write_bundle

ADMIN_KEY = "rag_key_admin"
TENANT_KEY = "rag_key_acme"
OTHER_KEY = "rag_key_globex"

KEYS: dict[str, dict[str, Any]] = {
    digest_of(ADMIN_KEY): {
        "tenant_id": "ops",
        "key_id": "ops-admin",
        "scopes": ["admin"],
        "rate_limit_per_minute": 10_000,
    },
    digest_of(TENANT_KEY): {
        "tenant_id": "acme",
        "key_id": "acme-1",
        "rate_limit_per_minute": 3,
    },
    digest_of(OTHER_KEY): {
        "tenant_id": "globex",
        "key_id": "globex-1",
        "rate_limit_per_minute": 3,
    },
}


def _runtime(bundle: RagBundle) -> tuple[Any, None]:
    return object(), None


def _probes(registry: BundleRegistry) -> ReadinessProbes:
    return ReadinessProbes(checks={}, ttl_s=0.0)


@pytest.fixture
def app_with_keys(tmp_path: Path) -> Iterator[TestClient]:
    root = tmp_path / "bundles"
    write_bundle(root, "0.1.0")
    keys = tmp_path / "api-keys.json"
    keys.write_text(json.dumps(KEYS), encoding="utf-8")
    settings = Settings(bundle_root=root, log_level="CRITICAL", api_keys_file=keys)
    with TestClient(
        create_app(settings=settings, build_runtime=_runtime, probe_factory=_probes)
    ) as client:
        yield client


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# 1. Xác thực
# ---------------------------------------------------------------------------


def test_no_key_is_401(app_with_keys: TestClient) -> None:
    assert app_with_keys.get("/admin/bundle").status_code == 401


def test_a_401_names_the_scheme_it_wants(app_with_keys: TestClient) -> None:
    """RFC 7235: 401 mà không kèm `WWW-Authenticate` để client đoán cơ chế nào
    đang bị đòi."""
    assert app_with_keys.get("/admin/bundle").headers["www-authenticate"] == "Bearer"


def test_a_wrong_key_is_indistinguishable_from_no_key(app_with_keys: TestClient) -> None:
    """⭐ Phân biệt hai ca này biến endpoint thành **máy kiểm key**: người tấn công
    dán một chuỗi vào và biết được nó có phải key hợp lệ hay không, mà không cần
    quyền nào."""
    without = app_with_keys.get("/admin/bundle")
    wrong = app_with_keys.get("/admin/bundle", headers=bearer("rag_khong_co_that"))
    assert wrong.status_code == without.status_code == 401
    assert wrong.json() == without.json()


def test_a_valid_key_gets_through(app_with_keys: TestClient) -> None:
    assert app_with_keys.get("/admin/bundle", headers=bearer(ADMIN_KEY)).status_code == 200


def test_a_non_bearer_authorization_header_is_not_accepted(app_with_keys: TestClient) -> None:
    assert (
        app_with_keys.get(
            "/admin/bundle", headers={"Authorization": f"Basic {ADMIN_KEY}"}
        ).status_code
        == 401
    )


def test_a_401_still_carries_a_request_id(app_with_keys: TestClient) -> None:
    """⭐ Ràng buộc **thứ tự middleware**, và nó ngược trực giác.

    `add_middleware` chèn lên đầu, nên cái thêm sau nằm ngoài. Gắn auth *sau*
    `RequestContextMiddleware` thì auth thành lớp ngoài cùng và mọi phản hồi
    401/403/429 của nó **không** đi qua chỗ gắn `X-Request-ID` — tức đúng những
    phản hồi mà người vận hành cần truy vết lại là những phản hồi không truy được.
    """
    assert app_with_keys.get("/admin/bundle").headers["x-request-id"]


# ---------------------------------------------------------------------------
# 1b. `NEW-12` — trần thân request, và **vị trí** của nó trong chồng
# ---------------------------------------------------------------------------


def _than_qua_tran() -> bytes:
    from rag_core.settings import Settings as _S

    return b"x" * (_S().max_body_bytes + 1)


def test_than_qua_tran_bi_tu_choi_413(app_with_keys: TestClient) -> None:
    response = app_with_keys.post("/chat", content=_than_qua_tran(), headers=bearer(TENANT_KEY))
    assert response.status_code == 413


def test_413_van_mang_request_id(app_with_keys: TestClient) -> None:
    """Nửa **trong** của quyết định vị trí: trần nằm dưới
    `RequestContextMiddleware`, nên 413 truy vết được như mọi phản hồi khác.
    Cùng ràng buộc với `test_a_401_still_carries_a_request_id` ngay trên."""
    response = app_with_keys.post("/chat", content=_than_qua_tran(), headers=bearer(TENANT_KEY))
    assert response.headers["x-request-id"]


def test_khong_khoa_thi_401_chu_khong_phai_413(app_with_keys: TestClient) -> None:
    """⭐⭐ Nửa **ngoài** của quyết định vị trí, và bản đầu làm ngược.

    `AuthMiddleware` quyết định hoàn toàn bằng header và không chạm `receive`,
    nên nó từ chối một thân 200 MB **không khoá** sau 0 byte, trong khi trần
    phải đếm tới `max_bytes` mới biết. Phép từ chối rẻ hơn phải ra ngoài hơn —
    và 413 trước 401 còn tiết lộ con số trần cho người chưa xác thực.
    """
    response = app_with_keys.post("/chat", content=_than_qua_tran())
    assert response.status_code == 401


def test_duong_cong_khai_van_duoc_tran_phu(app_with_keys: TestClient) -> None:
    """Hệ quả phải kiểm của việc đặt trần **trong** auth: `PUBLIC_PATHS` đi
    thẳng qua auth, nên nếu trần nằm sai chỗ thì `/health` thành một cửa nhận
    thân request không giới hạn."""
    response = app_with_keys.post("/health", content=_than_qua_tran())
    assert response.status_code == 413


def test_than_hop_le_lon_nhat_KHONG_bi_chan(app_with_keys: TestClient) -> None:
    """⭐ 204.090 byte — `/ingest` với 1000 `doc_ids` × 200 ký tự, đúng trần mà
    `StartRequest` tự khai. Một trần đặt thấp hơn con số này làm chết đường
    ingest, và triệu chứng ("ingest thỉnh thoảng 413") không trỏ về đây."""
    body = json.dumps({"config": "c" * 63, "doc_ids": ["d" * 200] * 1000}).encode()
    assert len(body) == 204_090
    response = app_with_keys.post(
        "/admin/ingest",
        content=body,
        headers={**bearer(ADMIN_KEY), "content-type": "application/json"},
    )
    # ⚠️ Bản đầu gọi `/ingest/start` — một đường **không tồn tại**. Nó trả 404,
    # và `!= 413` xanh vì lý do sai: một bài test không thể đỏ. Nên ở đây ghim
    # thẳng con số status, không ghim một phép **khác**.
    assert response.status_code != 404, "route sai thì bài này không đo gì cả"
    assert response.status_code != 413


# ---------------------------------------------------------------------------
# 2. ⭐⭐ Không route nào lọt ra ngoài
# ---------------------------------------------------------------------------


def test_the_probes_stay_public(app_with_keys: TestClient) -> None:
    """Bắt probe mang credential thì credential ấy nằm trong manifest deploy của
    mọi môi trường và xoay vòng được cùng lúc với việc restart cả cụm."""
    assert app_with_keys.get("/health").status_code == 200
    assert app_with_keys.get("/ready").status_code in (200, 503)


def test_the_ui_page_stays_public(app_with_keys: TestClient) -> None:
    """`W6-01`. Trang là chỗ người dùng **gõ khoá vào**; bắt nó xác thực tạo ra
    một bài toán con gà–quả trứng.

    ⚠️ Bài này phải chạy trên app **thật** có `AuthMiddleware`. Phép tiêm `M5`
    (xoá `"/"` khỏi `PUBLIC_PATHS`) sống sót qua cả `tests/unit/test_ui.py` vì
    fixture ở đó dựng một FastAPI trần không có middleware — nó kiểm được nội
    dung trang, không kiểm được ai vào được trang.
    """
    response = app_with_keys.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_the_public_page_does_not_make_the_api_public(app_with_keys: TestClient) -> None:
    """Mặt kia của bài trên: trang mở KHÔNG được kéo theo endpoint dữ liệu nào."""
    assert app_with_keys.post("/chat", json={"message": "xin chào"}).status_code == 401
    assert app_with_keys.get("/admin/ingest/j1").status_code == 401


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/redoc"])
def test_the_api_description_is_not_public(app_with_keys: TestClient, path: str) -> None:
    """`/health` lộ hai bit (sống, đã nạp bundle chưa). `/openapi.json` lộ **toàn
    bộ bề mặt tấn công** — hai thứ không cùng hạng."""
    assert app_with_keys.get(path).status_code == 401


def test_every_route_is_either_public_on_purpose_or_locked(app_with_keys: TestClient) -> None:
    """⭐⭐ Test đắt giá nhất file này, vì nó không kiểm một hành vi mà kiểm rằng
    **không tồn tại** một route lọt ra ngoài.

    Route mới mặc định bị khoá (middleware chặn theo mặc định), nên test này chỉ
    đỏ khi ai đó **thêm tên vào `PUBLIC_PATHS`** — tức đúng lúc cần một người
    thứ hai nhìn vào.

    ⚠️ **`app.routes` một mình là không đủ, và lần viết đầu tôi đã sai đúng chỗ
    đó.** FastAPI ở phiên bản này gói mỗi `include_router` vào một
    `_IncludedRouter` không có `.path`, nên vòng lặp qua `app.routes` chỉ thấy 4
    route tài liệu và **không thấy một route `/admin` nào** — test vẫn xanh trong
    khi nó không kiểm cái nó nói. Cứu nó là dòng `assert len(paths) >= 8`: một
    phép đếm cận dưới, thứ duy nhất phân biệt được "quét sạch, không có gì hở"
    với "không quét gì cả".
    """
    paths = {
        route.path
        for route in app_with_keys.app.router.routes  # type: ignore[attr-defined]
        if hasattr(route, "path")
    } | set(app_with_keys.app.openapi()["paths"])  # type: ignore[attr-defined]
    assert len(paths) >= 8, f"chỉ liệt kê được {len(paths)} route — phép quét đã hụt"

    for path in sorted(paths):
        if "{" in path:
            continue
        response = app_with_keys.get(path)
        if path in PUBLIC_PATHS:
            assert response.status_code != 401, f"{path} lẽ ra công khai"
        else:
            assert response.status_code == 401, (
                f"{path} trả {response.status_code} khi không có key — hoặc nó phải "
                "nằm trong `PUBLIC_PATHS`, hoặc middleware bị bỏ qua"
            )


# ---------------------------------------------------------------------------
# 3. Scope — key của tenant không được đổi bundle của cả cụm
# ---------------------------------------------------------------------------


def test_a_tenant_key_cannot_touch_the_admin_routes(app_with_keys: TestClient) -> None:
    """⭐ Key hợp lệ **không** đồng nghĩa với quyền đổi hệ thống đang phục vụ mọi
    khách hàng khác."""
    assert app_with_keys.get("/admin/bundle", headers=bearer(TENANT_KEY)).status_code == 403


def test_the_refusal_is_403_not_401(app_with_keys: TestClient) -> None:
    """401 nghĩa là "thử credential khác" — hướng dẫn sai khi credential đã đúng
    mà quyền thì không."""
    body = app_with_keys.post("/admin/bundle/rollback", headers=bearer(TENANT_KEY))
    assert body.status_code == 403
    assert "admin" in body.json()["detail"]


# ---------------------------------------------------------------------------
# 4. Hạn mức nhịp
# ---------------------------------------------------------------------------


def test_going_over_quota_is_429(app_with_keys: TestClient) -> None:
    for _ in range(3):
        assert app_with_keys.get("/docs", headers=bearer(TENANT_KEY)).status_code != 429
    assert app_with_keys.get("/docs", headers=bearer(TENANT_KEY)).status_code == 429


def test_a_429_says_when_to_come_back(app_with_keys: TestClient) -> None:
    for _ in range(4):
        response = app_with_keys.get("/docs", headers=bearer(TENANT_KEY))
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


def test_retry_after_is_never_zero(app_with_keys: TestClient) -> None:
    """⭐ `int()` hay `round()` cho **0** với mọi khoảng chờ dưới một giây, và một
    client lịch sự đọc `Retry-After: 0` thử lại ngay — nhận 429 tiếp, thử lại
    ngay. Header sinh ra để giảm tải lại thành một vòng lặp nóng."""
    seen: list[int] = []
    for _ in range(12):
        response = app_with_keys.get("/docs", headers=bearer(TENANT_KEY))
        if response.status_code == 429:
            seen.append(int(response.headers["Retry-After"]))
    assert seen and min(seen) >= 1


def test_one_tenant_cannot_starve_another(app_with_keys: TestClient) -> None:
    """⭐ Hạn mức theo **tenant**, không theo tiến trình. Thiếu tính chất này thì
    một khách hàng ồn ào làm mọi khách hàng khác nhận 429."""
    for _ in range(5):
        app_with_keys.get("/docs", headers=bearer(TENANT_KEY))
    assert app_with_keys.get("/docs", headers=bearer(TENANT_KEY)).status_code == 429
    assert app_with_keys.get("/docs", headers=bearer(OTHER_KEY)).status_code != 429


def test_a_successful_response_reports_the_budget(app_with_keys: TestClient) -> None:
    """Client tự điều tiết được thì nó không cần chạm 429 lần nào."""
    headers = app_with_keys.get("/docs", headers=bearer(TENANT_KEY)).headers
    assert headers["X-RateLimit-Limit"] == "3"
    assert int(headers["X-RateLimit-Remaining"]) <= 2


def test_the_bucket_refills(app_with_keys: TestClient) -> None:
    """3/phút = một token mỗi 20 giây, nên phép thử này không chờ được thật. Nó
    kiểm phần **nạp lại** ở tầng `RateLimiter` (đơn vị), còn ở đây chỉ ghim rằng
    429 là trạng thái tạm chứ không phải một cờ dính."""
    for _ in range(5):
        app_with_keys.get("/docs", headers=bearer(TENANT_KEY))
    limiter = app_with_keys.app.state.limiter  # type: ignore[attr-defined]
    limiter.buckets["acme"].updated = time.monotonic() - 60
    assert app_with_keys.get("/docs", headers=bearer(TENANT_KEY)).status_code != 429


def test_the_probes_do_not_eat_the_quota(app_with_keys: TestClient) -> None:
    """⭐ `/ready` bị orchestrator hỏi vài giây một lần. Nếu nó tính vào hạn mức
    thì chính phép thử sẵn sàng làm cạn quota của tenant — và nó cạn nhanh hơn
    khi cụm to ra."""
    for _ in range(50):
        app_with_keys.get("/ready")
    assert app_with_keys.get("/docs", headers=bearer(TENANT_KEY)).status_code == 200


# ---------------------------------------------------------------------------
# 5. Không có kho key
# ---------------------------------------------------------------------------


def test_a_missing_key_store_locks_everything_instead_of_opening_it(tmp_path: Path) -> None:
    """⭐⭐ Hướng hỏng của cấu hình thiếu.

    Không có file key thì lựa chọn là "khoá hết" hoặc "mở hết". Mở hết là điều
    duy nhất không được phép: nó biến một lỗi triển khai (quên mount volume)
    thành một API công khai, và **không gì báo** — mọi request đều thành công.
    """
    root = tmp_path / "bundles"
    write_bundle(root, "0.1.0")
    settings = Settings(
        bundle_root=root, log_level="CRITICAL", api_keys_file=tmp_path / "không-có.json"
    )
    with TestClient(
        create_app(settings=settings, build_runtime=_runtime, probe_factory=_probes)
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/admin/bundle").status_code == 401
        assert client.get("/admin/bundle", headers=bearer(ADMIN_KEY)).status_code == 401
