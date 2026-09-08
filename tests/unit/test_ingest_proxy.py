"""Cầu nối `/admin/ingest` → dịch vụ ingestion — `W6-01`.

Xem docstring `serving/api/ingest.py`: proxy tồn tại vì `AU-10` (dịch vụ ingest
không có auth, đang được che bằng bind `127.0.0.1`). Đi vòng qua `/admin` cho
tầng auth của `W4-04` áp dụng được **theo tiền tố đường dẫn**, chứ không phải
theo trí nhớ của người thêm route.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_core.settings import Settings
from serving.api import ingest
from serving.core.auth import Principal


def _client(url: str | None, principal: Principal | None = None) -> TestClient:
    app = FastAPI()
    if principal is not None:
        # Trong app thật, `AuthMiddleware` đặt principal; ở đây một middleware
        # tối giản đóng vai nó — route chỉ đọc `request.state.principal`.
        @app.middleware("http")
        async def stamp(request: Any, call_next: Any) -> Any:
            request.state.principal = principal
            return await call_next(request)

    app.include_router(ingest.router)
    app.state.settings = Settings(ingest_api_url=url)
    return TestClient(app)


class TestItIsOffByDefault:
    def test_no_url_means_503_with_a_reason(self) -> None:
        """⭐ Một bề mặt điều khiển pipeline mở sẵn ở mọi lần deploy là thứ không
        ai xin. Tắt mặc định, và **nói ra** là đang tắt — 503 câm sẽ bị đọc là
        "dịch vụ chết" và ai đó sẽ đi khởi động lại một thứ đang khoẻ."""
        response = _client(None).get("/admin/ingest/job-1")
        assert response.status_code == 503
        assert "INGEST_API_URL" in response.json()["detail"]

    def test_an_empty_string_counts_as_off(self) -> None:
        """`INGEST_API_URL=` trong một tệp env là chuỗi rỗng, không phải `None`.
        Đọc nó là "đã cấu hình" thì proxy sẽ gọi `"/ingest"` — một URL tương đối
        — và lỗi hiện ra là một lỗi httpx khó hiểu thay vì một câu tiếng người."""
        assert _client("   ").post("/admin/ingest", json={"config": "x"}).status_code == 503


class TestItDoesNotLeakTheInternalService:
    def test_a_transport_error_becomes_a_plain_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`AU-03` (`NEW-08`) đã vá đúng chế độ này ở đường chat: thân lỗi của
        một dịch vụ nội bộ mang host, cổng, và đôi khi cả header đi kèm."""

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("kết nối bị từ chối tới 10.0.0.7:8001")

        _install(monkeypatch, boom)
        response = _client("http://ingest:8001").get("/admin/ingest/job-1")
        assert response.status_code == 502
        assert "10.0.0.7" not in response.text
        assert response.json()["detail"] == "không gọi được dịch vụ ingest"

    def test_an_upstream_4xx_keeps_its_status_but_not_its_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mã trạng thái là thông tin **cần** cho client (404 = job không tồn
        tại, khác hẳn 502). Thân lỗi thì không."""

        def not_found(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "job ở /var/lib/rag/jobs không có"})

        _install(monkeypatch, not_found)
        response = _client("http://ingest:8001").get("/admin/ingest/nope")
        assert response.status_code == 404
        assert "/var/lib/rag" not in response.text


class TestItPassesTheRequestThrough:
    def test_start_forwards_config_and_returns_the_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def accept(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = request.read().decode()
            return httpx.Response(202, json={"job_id": "j1", "state": "queued"})

        _install(monkeypatch, accept)
        response = _client("http://ingest:8001/").post(
            "/admin/ingest", json={"config": "bgem3", "recreate": True}
        )
        assert response.status_code == 202
        assert response.json()["job_id"] == "j1"
        assert seen["url"] == "http://ingest:8001/ingest", "dấu `/` thừa phải bị cắt"
        import json

        forwarded = json.loads(seen["body"])
        assert forwarded == {"config": "bgem3", "doc_ids": [], "recreate": True}

    def test_a_config_that_looks_like_a_path_is_refused_before_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """⭐⭐ `pipeline.ingest.schemas` từ chối đường dẫn vì nhận nó thì
        `../../.env` đi qua được. Một proxy nới lỏng giới hạn ấy đã vô hiệu hoá
        nó — hàng rào ở dịch vụ trong không cứu được nếu proxy dựng URL từ chuỗi
        chưa kiểm.

        Ở đây `max_length=63` chặn phần lớn, và `job_id` đi qua
        `encodeURIComponent` + FastAPI path param. Bài này ghim rằng một config
        dài bất thường **không** tới được dịch vụ trong."""
        called = False

        def spy(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(202, json={"job_id": "x"})

        _install(monkeypatch, spy)
        response = _client("http://ingest:8001").post("/admin/ingest", json={"config": "a" * 200})
        assert response.status_code == 422
        assert not called

    def test_progress_reads_the_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def job(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://ingest:8001/ingest/j1"
            return httpx.Response(
                200, json={"job_id": "j1", "state": "running", "documents_done": 3}
            )

        _install(monkeypatch, job)
        body = _client("http://ingest:8001").get("/admin/ingest/j1").json()
        assert body["documents_done"] == 3


_UPLOAD = {
    "config": "bgem3",
    "title": "Báo cáo thuỷ lợi 2026",
    "content": "Đầu tư công cho thuỷ lợi tăng đều qua các năm. " * 40,
    "license": "CC BY 4.0",
    "source_url": "https://example.org/thuy-loi",
}


class TestUploadThroughTheProxy:
    """`NEW-11` — cửa nhận tài liệu, nhìn từ phía duy nhất trình duyệt thấy."""

    def test_it_forwards_and_stamps_the_authenticated_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`uploaded_by` là danh tính ĐÃ XÁC THỰC — proxy điền từ principal,
        và đó là nửa "ai chịu trách nhiệm" của quyết định `NEW-11`: sổ đăng ký
        corpus phải nói được ai đưa tài liệu này vào."""
        seen: dict[str, Any] = {}

        def accept(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = request.read().decode()
            return httpx.Response(201, json={"doc_id": "up-bao-cao-abc123"})

        _install(monkeypatch, accept)
        who = Principal(tenant_id="acme", key_id="key-1", scopes=frozenset({"admin"}))
        response = _client("http://ingest:8001", principal=who).post(
            "/admin/ingest/upload", json=_UPLOAD
        )
        assert response.status_code == 201
        assert seen["url"] == "http://ingest:8001/upload"
        import json

        forwarded = json.loads(seen["body"])
        assert forwarded["uploaded_by"] == "acme:key-1"
        assert forwarded["title"] == _UPLOAD["title"]
        assert forwarded["license"] == "CC BY 4.0"

    def test_a_client_cannot_claim_its_own_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Một client tự khai `uploaded_by` là một trường THỪA (`extra=forbid`)
        — nhận nó là để manifest ghi một danh tính do kẻ gửi chọn."""
        called = False

        def spy(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(201, json={})

        _install(monkeypatch, spy)
        response = _client("http://ingest:8001").post(
            "/admin/ingest/upload", json={**_UPLOAD, "uploaded_by": "root"}
        )
        assert response.status_code == 422
        assert not called

    def test_a_forbidden_license_dies_here_with_a_readable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_call` che thân lỗi của dịch vụ trong (`AU-03`), nên nếu chỉ dịch vụ
        trong kiểm license thì người dùng nhận một 4xx câm. Kiểm SỚM ở proxy là
        cách duy nhất thông điệp "giấy phép nào được nhận" tới được form."""
        called = False

        def spy(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(201, json={})

        _install(monkeypatch, spy)
        response = _client("http://ingest:8001").post(
            "/admin/ingest/upload", json={**_UPLOAD, "license": "CC BY-ND 4.0"}
        )
        assert response.status_code == 422
        assert "danh sách cho phép" in response.text
        assert not called

    def test_the_byte_cap_counts_utf8_not_characters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """200.000 ký tự "ạ" = 600.000 byte: dưới trần ký tự, trên trần byte.
        Đo bằng ký tự thì request này đi tiếp và chết ở `BodyLimitMiddleware`
        với một thông điệp về thân request, không phải về tài liệu."""
        called = False

        def spy(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(201, json={})

        _install(monkeypatch, spy)
        response = _client("http://ingest:8001").post(
            "/admin/ingest/upload", json={**_UPLOAD, "content": "ạ" * 200_000}
        )
        assert response.status_code == 422
        assert "byte" in response.text
        assert not called


def _install(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Ép mọi `httpx.AsyncClient` trong module dùng transport giả."""
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    class Patched(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Patched)
