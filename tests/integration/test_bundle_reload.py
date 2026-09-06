"""`W4-02` — đổi bundle lúc đang chạy, và lùi lại được.

DoD có ba câu; câu ở giữa (*"request đang chạy không bị lỗi"*) là câu duy nhất
không kiểm được bằng cách gọi hàm rồi đọc giá trị trả về. Nó là một tính chất về
**thời điểm**, nên phần lớn file này chạy nhiều luồng thật.

Không cần Qdrant hay GPU: `BundleRegistry` nhận `RuntimeBuilder` từ ngoài, nên
những cách hỏng đáng sợ nhất — dựng lỗi giữa chừng, đổi ngay giữa một request —
dựng lại được bằng một builder giả có kiểm soát.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag_core.bundle import (
    BundleChecksumError,
    BundleComponents,
    ChunkingComponent,
    EmbeddingComponent,
    EvalReport,
    GateRecord,
    GateStatus,
    IndexComponent,
    RagBundle,
    RerankComponent,
    RetrievalComponent,
    save_bundle,
)
from rag_core.schemas import RetrievedChunk
from serving.core.registry import (
    WARMUP_QUERY,
    ActiveBundle,
    BundleRegistry,
    NoBundleLoadedError,
    NothingToRollBackError,
)

# ---------------------------------------------------------------------------
# Đồ giả
# ---------------------------------------------------------------------------


@dataclass
class FakeRetriever:
    """Đủ để phân biệt được "runtime nào đang trả lời" và để **chặn giữa chừng**."""

    version: str
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    closed: bool = False

    @property
    def name(self) -> str:
        return f"fake:{self.version}"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self.entered.set()
        self.release.wait(timeout=5)
        return []

    def close(self) -> None:
        self.closed = True


@dataclass
class RecordingBuilder:
    """Đếm số lần dựng, và hỏng theo yêu cầu."""

    built: list[str] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)
    made: dict[str, FakeRetriever] = field(default_factory=dict)

    def __call__(self, bundle: RagBundle) -> tuple[Any, None]:
        version = bundle.bundle_version
        if version in self.fail_on:
            raise RuntimeError(f"GPU hết chỗ khi dựng {version}")
        self.built.append(version)
        retriever = FakeRetriever(version)
        self.made[version] = retriever
        return retriever, None


def write_bundle(root: Path, version: str, *, collection: str = "rag_bgem3_ctx") -> RagBundle:
    bundle = RagBundle(
        bundle_version=version,
        created_at=datetime(2026, 9, 3, 10, 0, tzinfo=UTC),
        git_sha="e833015",
        components=BundleComponents(
            chunking=ChunkingComponent(
                strategy="hybrid",
                chunk_size=1000,
                chunk_overlap=100,
                contextual=True,
                chunking_fingerprint="c7ca3e6fc4da29a5",
            ),
            embedding=EmbeddingComponent(model="BAAI/bge-m3", dim=1024, normalize=True),
            index=IndexComponent(
                backend="qdrant",
                collection=collection,
                fingerprint="a" * 64,
                n_chunks=15814,
                n_documents=60,
            ),
            retrieval=RetrievalComponent(mode="hybrid", top_k=20, options={"k": 1}),
            rerank=RerankComponent(
                model="BAAI/bge-reranker-v2-m3", candidates=50, top_n=6, max_length=512
            ),
        ),
        eval=EvalReport(
            golden_set="golden_v1",
            n_queries=209,
            evaluated_with_generator="deepseek-chat@2026-09",
            retrieval_metrics={"ndcg@10": 0.6888},
        ),
        gate=GateRecord(status=GateStatus.NOT_RUN),
    )
    save_bundle(bundle, root)
    return bundle


@pytest.fixture
def registry(tmp_path: Path) -> BundleRegistry:
    for version in ("1.0.0", "1.1.0", "1.2.0"):
        write_bundle(tmp_path, version, collection=f"col_{version.replace('.', '_')}")
    # `warmup=False`: nhóm test này nói về ngữ nghĩa đổi/lùi, và `FakeRetriever`
    # cố ý CHẶN trong `retrieve()` để dựng cảnh "request đang chạy lúc reload".
    # Bật làm nóng ở đây thì mỗi `activate()` tự dính đúng cái chặn ấy. Phần làm
    # nóng có nhóm test riêng bên dưới (`TestWarmup`).
    return BundleRegistry(root=tmp_path, build_runtime=RecordingBuilder(), warmup=False)


def builder_of(registry: BundleRegistry) -> RecordingBuilder:
    assert isinstance(registry.build_runtime, RecordingBuilder)
    return registry.build_runtime


# ---------------------------------------------------------------------------
# 1. Trước khi nạp
# ---------------------------------------------------------------------------


def test_asking_for_the_active_bundle_before_loading_is_a_named_error(
    registry: BundleRegistry,
) -> None:
    """`W4-03` ánh xạ đúng lỗi này thành `/ready` 503 — "đang khởi động", không
    phải "sự cố". Nên nó phải là một kiểu riêng, không phải `RuntimeError` chung."""
    assert registry.is_ready is False
    with pytest.raises(NoBundleLoadedError, match="chưa có bundle"):
        _ = registry.active


def test_status_is_answerable_before_anything_is_loaded(registry: BundleRegistry) -> None:
    assert registry.status() == {"active": None, "active_since": None, "rollback_to": None}


# ---------------------------------------------------------------------------
# 2. ⭐⭐ Dựng lỗi KHÔNG được chạm vào bản đang phục vụ
# ---------------------------------------------------------------------------


def test_a_failed_reload_leaves_the_old_bundle_serving(registry: BundleRegistry) -> None:
    """⭐⭐ Cách hỏng tệ nhất của cả hạng mục.

    Nếu gỡ bản cũ ra trước rồi mới dựng bản mới, thì một lần reload lỗi biến
    `/chat` từ *"đang phục vụ bản cũ"* thành *"không phục vụ gì"* — một thao tác
    nhằm cải thiện hệ thống lại là thao tác làm sập nó.
    """
    registry.activate("1.0.0")
    builder_of(registry).fail_on.add("1.1.0")

    with pytest.raises(RuntimeError, match="GPU hết chỗ"):
        registry.activate("1.1.0")

    assert registry.active.version == "1.0.0"
    assert registry.active.retriever.name == "fake:1.0.0"
    assert registry.status()["rollback_to"] is None


def test_a_corrupt_manifest_also_leaves_the_old_bundle_serving(
    registry: BundleRegistry, tmp_path: Path
) -> None:
    """Hỏng ở bước **đọc** cũng phải vô hại như hỏng ở bước dựng."""
    registry.activate("1.0.0")
    manifest = tmp_path / "rag-bundle-v1.1.0" / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("15814", "999"), "utf-8")

    with pytest.raises(BundleChecksumError):
        registry.activate("1.1.0")
    assert registry.active.version == "1.0.0"


def test_a_failed_first_load_does_not_leave_a_half_ready_registry(
    registry: BundleRegistry,
) -> None:
    """Không có bản cũ để giữ thì phải vẫn là "chưa sẵn sàng", không phải "sẵn sàng với None"."""
    builder_of(registry).fail_on.add("1.0.0")
    with pytest.raises(RuntimeError):
        registry.activate("1.0.0")
    assert registry.is_ready is False


def test_an_unknown_version_is_refused_before_touching_disk(registry: BundleRegistry) -> None:
    registry.activate("1.0.0")
    with pytest.raises(Exception, match="semver"):
        registry.activate("latest")
    assert registry.active.version == "1.0.0"


# ---------------------------------------------------------------------------
# 3. ⭐⭐ Request đang chạy không bị ảnh hưởng
# ---------------------------------------------------------------------------


def test_a_snapshot_taken_before_the_swap_keeps_serving_the_old_runtime(
    registry: BundleRegistry,
) -> None:
    """Ảnh chụp bất biến là **cơ chế**, không phải tiện lợi.

    Nếu `active` trả về registry rồi người gọi đọc `.retriever` từ đó, hai lần
    đọc trong một request có thể ra hai runtime — câu trả lời sẽ trích chunk của
    index này bằng điểm số của index kia, và không có gì đỏ.
    """
    held: ActiveBundle = registry.activate("1.0.0")
    registry.activate("1.1.0")

    assert held.version == "1.0.0"
    assert held.retriever.name == "fake:1.0.0"
    assert registry.active.version == "1.1.0"


def test_an_in_flight_request_completes_across_a_swap(registry: BundleRegistry) -> None:
    """⭐⭐ Đúng câu giữa của DoD, và câu duy nhất phải chạy nhiều luồng mới kiểm được.

    Kịch bản thật: request bắt đầu, chạm Qdrant, và **trong lúc nó đang chờ** thì
    một lệnh reload chạy xong. Request ấy phải kết thúc bình thường trên runtime
    nó đã cầm.
    """
    snapshot = registry.activate("1.0.0")
    old = snapshot.retriever
    assert isinstance(old, FakeRetriever)
    result: dict[str, Any] = {}

    def serve() -> None:
        try:
            result["value"] = snapshot.retriever.retrieve("câu hỏi", top_k=5)
        except BaseException as exc:  # pragma: no cover - chỉ chạy khi test đỏ
            result["error"] = exc

    worker = threading.Thread(target=serve)
    worker.start()
    assert old.entered.wait(timeout=5), "request chưa kịp bắt đầu"

    registry.activate("1.1.0")  # đổi NGAY GIỮA request đang chạy
    old.release.set()
    worker.join(timeout=5)

    assert "error" not in result, f"request đang chạy bị hỏng: {result.get('error')!r}"
    assert result["value"] == []
    assert registry.active.version == "1.1.0"


def test_the_outgoing_runtime_is_not_closed_on_swap(registry: BundleRegistry) -> None:
    """⭐ Cám dỗ tự nhiên là `close()` runtime cũ để thu hồi kết nối — và làm thế
    là kéo đổ đúng những request mà ảnh chụp bất biến vừa bảo vệ."""
    registry.activate("1.0.0")
    old = builder_of(registry).made["1.0.0"]
    registry.activate("1.1.0")
    assert old.closed is False


# ---------------------------------------------------------------------------
# 4. Rollback
# ---------------------------------------------------------------------------


def test_rollback_returns_to_the_previous_bundle(registry: BundleRegistry) -> None:
    registry.activate("1.0.0")
    registry.activate("1.1.0")
    assert registry.rollback().version == "1.0.0"
    assert registry.active.version == "1.0.0"


def test_rollback_does_not_rebuild_anything(registry: BundleRegistry) -> None:
    """⭐⭐ Rollback chỉ có ích khi mọi thứ đang hỏng, nên nó **không được phép hỏng**.

    Dựng lại là mở đường cho nó hỏng (đĩa đổi, mạng rớt, GPU hết chỗ). Nó phải
    kích hoạt lại đúng object runtime đã chạy — và đó chính là lý do runtime cũ
    không bị `close()` lúc đổi.
    """
    registry.activate("1.0.0")
    original = registry.active.retriever
    registry.activate("1.1.0")
    builder = builder_of(registry)
    builder.fail_on.add("1.0.0")  # dựng lại 1.0.0 giờ SẼ hỏng

    assert registry.rollback().retriever is original
    assert builder.built == ["1.0.0", "1.1.0"]


def test_rollback_with_no_history_raises_instead_of_doing_nothing(
    registry: BundleRegistry,
) -> None:
    """No-op im lặng để người vận hành tin rằng hệ thống vừa quay lại — đúng lúc
    họ đang xử lý sự cố."""
    registry.activate("1.0.0")
    with pytest.raises(NothingToRollBackError, match="không có bản nào"):
        registry.rollback()


def test_rollback_before_any_activation_raises(registry: BundleRegistry) -> None:
    with pytest.raises(NothingToRollBackError):
        registry.rollback()


def test_rolling_back_twice_returns_to_where_you_started(registry: BundleRegistry) -> None:
    """Lịch sử sâu một bậc: một lệnh rollback luôn có nghĩa xác định thay vì phụ
    thuộc vào đã gọi nó bao nhiêu lần."""
    registry.activate("1.0.0")
    registry.activate("1.1.0")
    assert registry.rollback().version == "1.0.0"
    assert registry.rollback().version == "1.1.0"


def test_a_failed_reload_does_not_become_the_rollback_target(registry: BundleRegistry) -> None:
    """⚠️ Bản dựng hỏng không bao giờ phục vụ ai, nên nó không được vào lịch sử —
    ngược lại thì rollback sẽ trỏ vào một bundle chưa từng chạy."""
    registry.activate("1.0.0")
    registry.activate("1.1.0")
    builder_of(registry).fail_on.add("1.2.0")
    with pytest.raises(RuntimeError):
        registry.activate("1.2.0")

    assert registry.status()["rollback_to"] == "1.0.0"
    assert registry.rollback().version == "1.0.0"


def test_reactivating_the_same_version_does_not_make_it_its_own_rollback_target(
    registry: BundleRegistry,
) -> None:
    """Nạp lại `1.1.0` hai lần rồi rollback phải về `1.0.0`, không phải về chính nó."""
    registry.activate("1.0.0")
    registry.activate("1.1.0")
    registry.activate("1.1.0")
    assert registry.rollback().version == "1.0.0"


# ---------------------------------------------------------------------------
# 5. Đồng thời
# ---------------------------------------------------------------------------


def test_concurrent_activations_do_not_interleave(registry: BundleRegistry) -> None:
    """Hai lệnh reload chồng nhau dựng hai runtime rồi ghi đè lẫn nhau, và bản
    thua cuộc thành một runtime không ai tham chiếu nhưng vẫn giữ GPU."""
    registry.activate("1.0.0")
    errors: list[BaseException] = []

    def swap(version: str) -> None:
        try:
            registry.activate(version)
        except BaseException as exc:  # pragma: no cover - chỉ chạy khi test đỏ
            errors.append(exc)

    threads = [threading.Thread(target=swap, args=(v,)) for v in ("1.1.0", "1.2.0") for _ in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert registry.active.version in {"1.1.0", "1.2.0"}
    # Bất biến thật sự quan trọng: bản đang phục vụ và bản lùi về là hai bản khác
    # nhau, và cả hai đều là bundle đã dựng thành công.
    status = registry.status()
    assert status["active"] != status["rollback_to"]
    assert status["rollback_to"] in {"1.0.0", "1.1.0", "1.2.0"}


# ---------------------------------------------------------------------------
# 8. Làm nóng khi kích hoạt (`TD-72`, quyết ở `W6-05`)
# ---------------------------------------------------------------------------


@dataclass
class WarmableRetriever:
    """Retriever ghi lại mọi truy vấn — kể cả lượt làm nóng."""

    version: str
    queries: list[str] = field(default_factory=list)
    fail: bool = False

    @property
    def name(self) -> str:
        return f"warmable:{self.version}"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("Qdrant chưa lên")
        return []


@dataclass
class WarmableBuilder:
    fail: bool = False
    made: dict[str, WarmableRetriever] = field(default_factory=dict)

    def __call__(self, bundle: RagBundle) -> tuple[Any, None]:
        retriever = WarmableRetriever(bundle.bundle_version, fail=self.fail)
        self.made[bundle.bundle_version] = retriever
        return retriever, None


class TestWarmup:
    """`W6-05` đo: request đầu 13.386 ms so với 4.272 ms nóng (+9,1 s, 3,1×),
    toàn bộ phần phạt ở `prepare` — kernel CUDA khởi tạo ở `score()` đầu tiên,
    sau khi `/ready` đã xanh."""

    def _registry(self, tmp_path: Path, *, warmup: bool, fail: bool = False) -> BundleRegistry:
        for version in ("1.0.0", "1.1.0"):
            write_bundle(tmp_path, version, collection=f"col_{version.replace('.', '_')}")
        return BundleRegistry(
            root=tmp_path, build_runtime=WarmableBuilder(fail=fail), warmup=warmup
        )

    def test_activate_runs_one_retrieval_before_publishing(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path, warmup=True)
        registry.activate("1.0.0")
        builder = registry.build_runtime
        assert isinstance(builder, WarmableBuilder)
        assert builder.made["1.0.0"].queries == [WARMUP_QUERY]

    def test_warmup_off_touches_nothing(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path, warmup=False)
        registry.activate("1.0.0")
        builder = registry.build_runtime
        assert isinstance(builder, WarmableBuilder)
        assert builder.made["1.0.0"].queries == []

    def test_reload_warms_too_not_just_startup(self, tmp_path: Path) -> None:
        """⭐⭐ Chế độ hỏng thuộc về KÍCH HOẠT, không phải khởi động.

        `POST /admin/bundle/reload` dựng một runtime mới với trọng số mới, và
        người dùng ngay sau lệnh reload nhận đúng 13 giây ấy — không có deploy
        nào để đổ lỗi. Một bản vá đặt ở `lifespan` sẽ bỏ sót đúng đường này.
        """
        registry = self._registry(tmp_path, warmup=True)
        registry.activate("1.0.0")
        registry.activate("1.1.0")
        builder = registry.build_runtime
        assert isinstance(builder, WarmableBuilder)
        assert builder.made["1.1.0"].queries == [WARMUP_QUERY]

    def test_a_failing_warmup_does_not_block_activation(self, tmp_path: Path) -> None:
        """⭐ Qdrant chưa lên là chuyện thường lúc container khởi động. Một bản
        vá latency biến sự cố tạm thời của phụ thuộc thành "không deploy được"
        đã đổi một vấn đề nhỏ lấy một vấn đề lớn hơn."""
        registry = self._registry(tmp_path, warmup=True, fail=True)
        assert registry.activate("1.0.0").version == "1.0.0"
        assert registry.is_ready

    def test_rollback_does_not_warm_again(self, tmp_path: Path) -> None:
        """Luật 3 của module: rollback kích hoạt lại CHÍNH object runtime cũ,
        không dựng lại gì — nên nó cũng đã nóng sẵn. Làm nóng lại ở đây là thêm
        một chỗ có thể hỏng vào đúng cơ chế không được phép hỏng."""
        registry = self._registry(tmp_path, warmup=True)
        registry.activate("1.0.0")
        registry.activate("1.1.0")
        builder = registry.build_runtime
        assert isinstance(builder, WarmableBuilder)
        registry.rollback()
        assert builder.made["1.0.0"].queries == [WARMUP_QUERY]
