"""`W4-03` — `/health`, `/ready`, và ba route admin.

Không cần Qdrant, không cần GPU: `create_app` nhận `build_runtime` từ ngoài
(cùng lý lẽ với `BundleRegistry` ở `W4-02`), nên "Qdrant chết" hay "GPU hết chỗ"
dựng lại được bằng một cờ boolean.

Nhóm đáng giá nhất là nhóm 2: **`/health` phải mù trước phụ thuộc**. Nếu nó
không mù thì một trục trặc 30 giây của Qdrant làm orchestrator khởi động lại
toàn bộ replica, và cái đó không có test nào ngoài chỗ này bắt được.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_core.bundle import RagBundle
from rag_core.reranking.base import Reranker
from rag_core.settings import Settings
from serving.api.app import create_app
from serving.core.auth import digest_of
from serving.core.probes import ReadinessProbes
from serving.core.registry import BundleRegistry
from serving.core.runtime import BundleRuntimeError
from tests.integration.test_bundle_reload import write_bundle


@dataclass
class FakeRetriever:
    version: str
    name: str = "fake"


@dataclass
class World:
    """Trạng thái hạ tầng giả, bật tắt được giữa hai request."""

    build_fails: bool = False
    qdrant_down: bool = False
    built: list[str] = field(default_factory=list)

    def build(self, bundle: RagBundle) -> tuple[Any, None]:
        if self.build_fails:
            raise BundleRuntimeError(
                f"collection có 12 điểm nhưng bundle {bundle.bundle_version} được eval trên 15.814"
            )
        self.built.append(bundle.bundle_version)
        return FakeRetriever(bundle.bundle_version), None

    def probes(self, registry: BundleRegistry) -> ReadinessProbes:
        def check() -> None:
            if self.qdrant_down:
                raise ConnectionError("connection refused")

        # ttl=0 để test điều khiển được từng lượt; hành vi nhớ tạm có bộ test riêng.
        return ReadinessProbes(checks={"qdrant": check}, ttl_s=0.0)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def bundles(tmp_path: Path) -> Path:
    root = tmp_path / "bundles"
    for version in ("0.1.0", "0.2.0"):
        write_bundle(root, version)
    return root


ADMIN_KEY = "rag_test_admin_key"


def keys_file(tmp: Path) -> Path:
    """Kho key cho test: một key admin, hạn mức rộng để không vướng `W4-04`.

    Ghi **digest** đúng như production — test đi qua chính đường tra key thật
    chứ không qua một cửa sau chỉ có trong test.
    """
    path = tmp / "api-keys.json"
    path.write_text(
        json.dumps(
            {
                digest_of(ADMIN_KEY): {
                    "tenant_id": "test",
                    "key_id": "test-admin",
                    "scopes": ["admin"],
                    "rate_limit_per_minute": 10_000,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def make_client(root: Path, world: World, *, pin: str | None = None) -> TestClient:
    settings = Settings(
        bundle_root=root,
        bundle_version=pin,
        log_level="CRITICAL",
        api_keys_file=keys_file(root.parent),
        # Bộ đếm hạn mức cục bộ, bất kể Docker: xem chú thích ở `chat_app.make`.
        quota_shared=False,
    )
    app = create_app(settings=settings, build_runtime=world.build, probe_factory=world.probes)
    return TestClient(app, headers={"Authorization": f"Bearer {ADMIN_KEY}"})


@pytest.fixture
def client(bundles: Path, world: World) -> Iterator[TestClient]:
    with make_client(bundles, world) as ready_client:
        yield ready_client


# ---------------------------------------------------------------------------
# 1. Khởi động
# ---------------------------------------------------------------------------


def test_the_newest_bundle_is_activated_when_none_is_pinned(
    client: TestClient, world: World
) -> None:
    """Semver, không phải thứ tự tên file — `0.2.0` thắng `0.1.0`."""
    assert world.built == ["0.2.0"]
    assert client.get("/ready").json()["active"] == "0.2.0"


def test_a_pinned_version_wins_over_the_newest(bundles: Path, world: World) -> None:
    """Ghim tường minh là cách duy nhất để một lần rollback sống sót qua restart."""
    with make_client(bundles, world, pin="0.1.0") as client:
        assert client.get("/ready").json()["active"] == "0.1.0"


def test_the_api_comes_up_even_when_no_bundle_can_be_loaded(tmp_path: Path, world: World) -> None:
    """⭐⭐ Fail-fast ở đây là sai, và cái sai đó tự che dấu vết của nó.

    Thư mục bundle rỗng (mount sai volume — ca thật, và là ca thường gặp nhất
    của lần deploy đầu). Nếu tiến trình chết thì nó crashloop, log của pod biến
    mất trước khi ai kịp đọc, và không còn endpoint nào để hỏi vì sao.
    """
    with make_client(tmp_path / "trống", world) as client:
        assert client.get("/health").status_code == 200
        body = client.get("/ready").json()
        assert body["ready"] is False
        assert body["checks"]["bundle"] == {"ok": False, "detail": "chưa nạp"}


def test_a_bundle_that_fails_to_build_does_not_kill_startup(bundles: Path, world: World) -> None:
    """Cùng lý lẽ, nhưng nguyên nhân khác: bundle đọc được, hạ tầng từ chối."""
    world.build_fails = True
    with make_client(bundles, world) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


# ---------------------------------------------------------------------------
# 2. ⭐⭐ `/health` mù trước phụ thuộc, `/ready` thì không
# ---------------------------------------------------------------------------


def test_health_stays_200_while_qdrant_is_down(client: TestClient, world: World) -> None:
    """⭐⭐ Test đắt giá nhất file này.

    `/health` điều khiển việc **khởi động lại container**. Nếu nó đỏ khi Qdrant
    chớp thì orchestrator giết toàn bộ replica cùng lúc; việc đó không chữa được
    Qdrant, mà mỗi replica mới mất vài phút nạp lại 3,3 GB trọng số. Một trục
    trặc 30 giây của phụ thuộc thành một sự cố nhiều phút của chính mình.
    """
    world.qdrant_down = True
    assert client.get("/health").status_code == 200
    assert client.get("/health").json() == {"status": "alive"}


def test_ready_turns_503_while_qdrant_is_down(client: TestClient, world: World) -> None:
    assert client.get("/ready").status_code == 200
    world.qdrant_down = True
    assert client.get("/ready").status_code == 503


def test_ready_recovers_on_its_own(client: TestClient, world: World) -> None:
    """Không cần restart: đúng nửa còn lại của việc tách liveness khỏi readiness."""
    world.qdrant_down = True
    assert client.get("/ready").status_code == 503
    world.qdrant_down = False
    assert client.get("/ready").status_code == 200


def test_a_503_says_which_dependency_failed(client: TestClient, world: World) -> None:
    """Người vận hành gõ `curl` lúc mọi thứ đang hỏng; một 503 rỗng bắt họ đi
    tìm log của đúng pod đó."""
    world.qdrant_down = True
    checks = client.get("/ready").json()["checks"]
    assert checks["bundle"]["ok"] is True
    assert checks["qdrant"]["ok"] is False
    assert "connection refused" in checks["qdrant"]["detail"]


def test_the_probe_reaches_the_qdrant_store_through_every_wrapper() -> None:
    """⭐ `_store_of` đi xuống bằng `getattr("base")`/`getattr("store")`, tức nó
    dựa vào một **quy ước đặt tên thuộc tính** của `rag_core` mà không gì ép.

    Đổi tên `RerankedRetriever.base` thành `inner` là hợp lệ với mọi test khác
    trong repo, và hậu quả ở đây im lặng: `_store_of` dừng ở lớp ngoài cùng,
    `.count()` không tồn tại, probe ném `AttributeError`, và `/ready` trả 503
    vĩnh viễn với lý do `"AttributeError"` trong khi Qdrant hoàn toàn khoẻ.

    Nên chuỗi dưới đây dựng bằng **lớp thật** — không có server nào bị chạm tới
    vì `QdrantDenseRetriever` chỉ mở client khi có truy vấn đầu tiên.
    """
    from rag_core.embedding.hashing import HashingEmbeddingProvider
    from rag_core.retrieval.hybrid import QdrantHybridRetriever
    from rag_core.retrieval.qdrant_store import QdrantDenseRetriever
    from rag_core.retrieval.reranked import RerankedRetriever
    from serving.api.app import _store_of

    store = QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=8, sparse=True), collection="rag_test"
    )
    stack: list[Any] = [
        store,
        QdrantHybridRetriever(store, k=1),
        RerankedRetriever(QdrantHybridRetriever(store, k=1), _StubReranker()),
    ]
    for retriever in stack:
        assert _store_of(retriever) is store, retriever.name


class _StubReranker(Reranker):
    name = "stub"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:  # pragma: no cover
        return [0.0] * len(texts)


def test_ready_does_not_probe_qdrant_before_a_bundle_exists(tmp_path: Path, world: World) -> None:
    """Sau một lần deploy, mọi replica cùng khởi động và cùng hỏi `/ready` vài
    giây một lần. Thăm dò một phụ thuộc chỉ để cũng trả 503 là tải thừa nhân với
    số replica."""
    probed: list[int] = []

    def probes(registry: BundleRegistry) -> ReadinessProbes:
        return ReadinessProbes(checks={"qdrant": lambda: probed.append(1)}, ttl_s=0.0)

    settings = Settings(bundle_root=tmp_path / "trống", log_level="CRITICAL", quota_shared=False)
    with TestClient(
        create_app(settings=settings, build_runtime=world.build, probe_factory=probes)
    ) as client:
        assert client.get("/ready").status_code == 503
    assert probed == []


# ---------------------------------------------------------------------------
# 3. request_id
# ---------------------------------------------------------------------------


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    assert client.get("/health").headers["x-request-id"]


def test_an_upstream_request_id_is_reused(client: TestClient) -> None:
    """Điều kiện để một dấu vết đi xuyên nhiều dịch vụ chứ không đứt ở cổng vào."""
    response = client.get("/health", headers={"X-Request-ID": "lb-abc123"})
    assert response.headers["x-request-id"] == "lb-abc123"


@pytest.mark.parametrize(
    "hostile",
    ["x" * 200, "a b", "trace\nlevel=INFO msg=fake"],
)
def test_an_unusable_upstream_id_is_replaced_not_echoed(client: TestClient, hostile: str) -> None:
    """JSON đã chặn phần nguy hiểm nhất (một `\\n` không tách được dòng log vì
    nó bị escape). Còn lại là độ dài vô hạn và việc dội một chuỗi tuỳ ý vào
    header phản hồi — hai thứ này chặn ở đây."""
    echoed = client.get("/health", headers={"X-Request-ID": hostile}).headers["x-request-id"]
    assert echoed != hostile
    assert len(echoed) == 32


def test_the_test_client_cannot_prove_streaming_works(bundles: Path, world: World) -> None:
    """⚠️ Ghim một **giới hạn của công cụ**, không phải một tính chất của mã.

    `middleware.py` là ASGI thuần thay vì `BaseHTTPMiddleware` vì cái sau đệm lại
    phản hồi dạng dòng, và `W4-06` là `POST /chat` SSE. Chỗ này lẽ ra là nơi ghim
    tính chất ấy — nhưng đo thử thì **`TestClient` tự nó đã đệm**: cùng đoạn mã
    dưới đây, chạy trên một `FastAPI()` trắng không middleware nào, vẫn gộp cả
    hai mẩu vào lần `iter_bytes()` đầu tiên.

    Nên một test viết ở đây sẽ **xanh với cả hai cách cài đặt** — tệ hơn là không
    có, vì nó tạo cảm giác đã kiểm. Điều đó có hệ quả trực tiếp cho `W4-06`: SSE
    ở đó **không** xác minh được bằng `TestClient`, phải dựng uvicorn thật.

    Test này đỏ khi starlette sửa hành vi ấy — lúc đó thay nó bằng phép kiểm thật.
    """
    from fastapi.responses import StreamingResponse

    settings = Settings(
        bundle_root=bundles,
        log_level="CRITICAL",
        api_keys_file=keys_file(bundles.parent),
        quota_shared=False,
    )
    app = create_app(settings=settings, build_runtime=world.build, probe_factory=world.probes)

    @app.get("/sse")
    def sse() -> StreamingResponse:
        return StreamingResponse(
            iter([b"data: mot\n\n", b"data: hai\n\n"]), media_type="text/event-stream"
        )

    with TestClient(app, headers=_auth()) as client, client.stream("GET", "/sse") as response:
        first = next(response.iter_bytes())
    assert b"hai" in first, "TestClient đã stream thật — viết lại test này thành phép kiểm thật"


def test_a_500_still_carries_the_request_id(bundles: Path, world: World) -> None:
    """⭐ Đúng phản hồi mà người vận hành cần truy vết lại là phản hồi dễ thiếu
    mã nhất: `ServerErrorMiddleware` của Starlette nằm **ngoài** middleware này
    nên 500 do nó gửi không đi qua `send` đã bọc."""
    settings = Settings(
        bundle_root=bundles,
        log_level="CRITICAL",
        api_keys_file=keys_file(bundles.parent),
        quota_shared=False,
    )
    app = create_app(settings=settings, build_runtime=world.build, probe_factory=world.probes)

    @app.get("/nổ")
    def boom() -> None:
        raise RuntimeError("vỡ")

    with TestClient(app, raise_server_exceptions=False, headers=_auth()) as client:
        response = client.get("/nổ", headers={"X-Request-ID": "trace-9"})
    assert response.status_code == 500
    assert response.headers["x-request-id"] == "trace-9"
    # Mã nằm trong **thân** phản hồi nữa, để người dùng cuối dán được vào báo lỗi.
    assert response.json()["request_id"] == "trace-9"


# ---------------------------------------------------------------------------
# 4. Route admin — mở khoá lõi `W4-02`
# ---------------------------------------------------------------------------


def test_reload_swaps_the_serving_bundle(client: TestClient, world: World) -> None:
    assert client.post("/admin/bundle/reload", json={"version": "0.1.0"}).status_code == 200
    assert client.get("/admin/bundle").json()["active"] == "0.1.0"
    assert world.built == ["0.2.0", "0.1.0"]


def test_rollback_returns_to_the_previous_bundle(client: TestClient, world: World) -> None:
    client.post("/admin/bundle/reload", json={"version": "0.1.0"})
    assert client.post("/admin/bundle/rollback").json()["version"] == "0.2.0"
    # Không dựng lại gì — đó là toàn bộ lý do rollback không hỏng được (`W4-02` §5).
    assert world.built == ["0.2.0", "0.1.0"]


def test_rollback_warns_that_it_does_not_survive_a_restart(client: TestClient) -> None:
    """Nó chỉ đổi trạng thái trong bộ nhớ. Lúc dễ quên nhất là ngay sau khi rollback
    thành công và mọi thứ trông đã ổn."""
    client.post("/admin/bundle/reload", json={"version": "0.1.0"})
    assert "BUNDLE_VERSION" in client.post("/admin/bundle/rollback").json()["warning"]


def test_rollback_with_no_history_is_a_conflict_not_a_silent_ok(client: TestClient) -> None:
    assert client.post("/admin/bundle/rollback").status_code == 409


def test_reloading_a_missing_version_says_the_old_one_still_serves(client: TestClient) -> None:
    """⭐ Câu hỏi đầu tiên của người vận hành khi một lệnh reload đỏ. Luật 1 của
    `W4-02` làm cho câu trả lời luôn là "còn" — nhưng chỉ khi phản hồi nói ra."""
    response = client.post("/admin/bundle/reload", json={"version": "9.9.9"})
    assert response.status_code == 404
    assert response.json()["detail"]["still_serving"] == "0.2.0"
    assert client.get("/ready").status_code == 200


def test_a_malformed_version_is_rejected_before_touching_the_disk(client: TestClient) -> None:
    assert client.post("/admin/bundle/reload", json={"version": "mới nhất"}).status_code == 422


def test_a_bundle_the_hardware_refuses_is_409_not_503(client: TestClient, world: World) -> None:
    """503 nghĩa là "thử lại sau". Số điểm không khớp thì thử lại y nguyên sẽ
    hỏng y nguyên — đó là 409."""
    world.build_fails = True
    response = client.post("/admin/bundle/reload", json={"version": "0.1.0"})
    assert response.status_code == 409
    assert "15.814" in response.json()["detail"]["detail"]


def test_an_unclassified_failure_is_still_a_refusal_not_a_500(bundles: Path, world: World) -> None:
    """⭐⭐ Nhánh này do **lần chạy thật đầu tiên** tìm ra, không do suy luận.

    Tắt Qdrant rồi gọi reload thật: `verify_schema` ném
    `qdrant_client...ResponseHandlingException("timed out")`. Nó không phải
    `BundleRuntimeError`, không phải `ValueError`, nên nó xuyên qua mọi `except`
    trong `reload()` và ra ngoài thành 500 `{"detail": "Lỗi nội bộ"}` của
    middleware — tức người vận hành nhận đúng thứ vô dụng nhất: không lý do, và
    **không** có câu quan trọng nhất là "bản cũ vẫn đang phục vụ".

    Mọi test trước đó xanh, vì builder giả chỉ biết ném đúng những lớp lỗi mà mã
    đã bắt. Đó là giới hạn của đồ giả, và nó chỉ lộ ra khi chạy thật.
    """

    class ResponseHandlingException(Exception):
        pass

    settings = Settings(
        bundle_root=bundles,
        bundle_version="0.2.0",
        log_level="CRITICAL",
        api_keys_file=keys_file(bundles.parent),
        quota_shared=False,
    )
    fails = False

    def build(bundle: RagBundle) -> tuple[Any, None]:
        if fails:
            raise ResponseHandlingException("timed out")
        return world.build(bundle)

    app = create_app(settings=settings, build_runtime=build, probe_factory=world.probes)
    with TestClient(app, headers=_auth()) as client:
        fails = True
        response = client.post("/admin/bundle/reload", json={"version": "0.1.0"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["still_serving"] == "0.2.0"
    # Lời của lỗi hạ tầng thường là một chuỗi trần; một mình nó không nói được
    # **cái gì** hết giờ.
    assert detail["detail"] == "ResponseHandlingException: timed out"


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}
