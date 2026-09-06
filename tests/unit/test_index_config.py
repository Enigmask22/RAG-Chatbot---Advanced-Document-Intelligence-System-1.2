"""W1-08 — `IndexConfig`: fingerprint, factory, đọc YAML.

Trọng tâm là `fingerprint`. Nó là thứ duy nhất trả lời được "index trong Qdrant
có phải do config này sinh ra không", nên phải đúng theo cả hai chiều: đổi thứ
ảnh hưởng tới vector thì fingerprint phải đổi, đổi thứ chỉ ảnh hưởng tốc độ thì
phải giữ nguyên. Sai chiều nào cũng tệ — chiều một cho phép trộn hai cấu hình
vào một collection, chiều hai bắt build lại vài giờ GPU mỗi lần đổi máy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pipeline.indexing.config import IndexConfig, load_index_config
from rag_core.chunking import ChunkingConfig, ChunkingStrategy
from rag_core.chunking.cache import CachedChunker
from rag_core.embedding import HashingEmbeddingProvider

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**kwargs: object) -> IndexConfig:
    base: dict[str, object] = {"name": "t", "tenant_id": "t", "embedding_model": "hashing:64"}
    base.update(kwargs)
    return IndexConfig.model_validate(base)


class TestFingerprint:
    def test_stable_across_identical_configs(self) -> None:
        assert _config().fingerprint == _config().fingerprint

    @pytest.mark.parametrize(
        "field,value",
        [
            ("embedding_model", "hashing:128"),
            ("embedding_normalize", False),
            ("languages", ("vi",)),
            ("doc_types", ("legal",)),
            ("max_documents", 5),
            ("embedding_kwargs", {"query_prefix": "query: "}),
        ],
    )
    def test_changes_when_content_affecting_field_changes(self, field: str, value: object) -> None:
        assert _config().fingerprint != _config(**{field: value}).fingerprint

    def test_changes_when_chunking_changes(self) -> None:
        other = ChunkingConfig(chunk_size=900)
        assert _config().fingerprint != _config(chunking=other).fingerprint

    @pytest.mark.parametrize(
        "field,value",
        [
            ("embedding_device", "cpu"),
            ("embedding_batch_size", 8),
            ("upsert_batch_size", 16),
            ("cache_path", Path("/tmp/other.sqlite3")),
            ("state_dir", Path("/tmp/state")),
            ("use_cache", False),
            ("collection", "khac"),
            ("manifest_path", Path("khac.csv")),
        ],
    )
    def test_unchanged_when_only_speed_or_location_changes(self, field: str, value: object) -> None:
        """Chạy trên laptop hay trên GPU thuê phải ra cùng một index về mặt logic.

        Nếu `device` vào fingerprint thì mỗi lần đổi máy là một lần build lại
        toàn bộ — đúng thứ kiến trúc hai plane sinh ra để tránh.
        """
        assert _config().fingerprint == _config(**{field: value}).fingerprint

    def test_neighbor_context_is_part_of_fingerprint(self) -> None:
        """Sai lệch nguy hiểm nhất so với POC phải được fingerprint bắt được."""
        poc = ChunkingConfig(neighbor_context_chars=100)
        assert _config().fingerprint != _config(chunking=poc).fingerprint


class TestDerived:
    def test_collection_defaults_to_name(self) -> None:
        assert _config(name="baseline").collection_name == "rag_baseline"

    def test_explicit_collection_wins(self) -> None:
        assert _config(collection="riêng").collection_name == "riêng"

    def test_state_path_is_per_run(self) -> None:
        assert _config(name="abl-01").state_path.name == "abl-01.json"

    def test_rejects_unknown_field(self) -> None:
        """`extra=forbid`: gõ sai tên trường trong YAML phải báo, không im lặng bỏ qua."""
        with pytest.raises(ValidationError):
            IndexConfig.model_validate({"name": "t", "embeding_model": "typo"})

    def test_rejects_bad_device(self) -> None:
        with pytest.raises(ValidationError, match="embedding_device"):
            _config(embedding_device="tpu")


class TestFactories:
    def test_builds_hashing_provider_without_torch(self) -> None:
        provider = _config(embedding_model="hashing:64").build_embeddings()
        assert isinstance(provider, HashingEmbeddingProvider)
        assert provider.dimension == 64

    def test_wraps_chunker_in_cache_when_enabled(self, tmp_path: Path) -> None:
        config = _config(use_cache=True, cache_path=tmp_path / "c.sqlite3")
        chunker = config.build_chunker(HashingEmbeddingProvider(64))
        assert isinstance(chunker, CachedChunker)

    def test_no_cache_wrapper_when_disabled(self, tmp_path: Path) -> None:
        config = _config(use_cache=False, cache_path=tmp_path / "c.sqlite3")
        assert not isinstance(config.build_chunker(HashingEmbeddingProvider(64)), CachedChunker)


class TestLoadYaml:
    def test_reads_nested_chunking(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "x",
                    "tenant_id": "t",
                    "embedding_model": "hashing:64",
                    "chunking": {"strategy": "fixed", "chunk_size": 800},
                }
            ),
            encoding="utf-8",
        )
        config = load_index_config(path)
        assert config.chunking.strategy is ChunkingStrategy.FIXED
        assert config.chunking.chunk_size == 800

    def test_overrides_skip_none(self, tmp_path: Path) -> None:
        """CLI truyền `None` cho cờ không đặt — không được ghi đè giá trị trong file."""
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump({"name": "x", "tenant_id": "t", "collection": "giu_nguyen"}), "utf-8"
        )
        config = load_index_config(path, collection=None, max_documents=7)
        assert config.collection == "giu_nguyen"
        assert config.max_documents == 7

    def test_missing_file_points_at_template(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"configs/indexing/baseline\.yaml"):
            load_index_config(tmp_path / "khong-co.yaml")

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_index_config(path)


class TestShippedConfigs:
    """Config đi kèm repo phải hợp lệ — hỏng thì `make index` chết ở dòng đầu."""

    @pytest.mark.parametrize("name", ["baseline", "smoke"])
    def test_parses(self, name: str) -> None:
        config = load_index_config(REPO_ROOT / "configs" / "indexing" / f"{name}.yaml")
        assert config.name == name

    def test_baseline_keeps_poc_neighbor_context(self) -> None:
        """Bảo vệ cảnh báo lớn nhất của `W1-13`.

        Mặc định của `ChunkingConfig` là 0, bản POC luôn chạy ở 100. Ai đó dọn
        dẹp config mà bỏ dòng này đi thì baseline sẽ đo một hệ thống chưa từng
        tồn tại, và không có triệu chứng nào ngoài việc con số hơi khác.
        """
        config = load_index_config(REPO_ROOT / "configs" / "indexing" / "baseline.yaml")
        assert config.chunking.neighbor_context_chars == 100


class TestTenant:
    """`TD-40` — mọi point phải có chủ, và việc quên phải nổ chứ không im.

    Món nợ này không phải một bug trong mã: `W2-06` đã ghi và ghim đúng hành vi
    "point thiếu `tenant_id` không khớp filter tenant nào". Nó là một **cấu hình
    thiếu** đã sống ba tuần, và nó chỉ lộ ra khi cả ba tầng cùng chạy một lần ở
    `W4-06`. Nhóm test này biến "phải nhớ" thành "không quên được".
    """

    def test_a_config_without_a_tenant_cannot_be_built(self) -> None:
        with pytest.raises(ValidationError, match="tenant_id"):
            IndexConfig.model_validate({"name": "t", "embedding_model": "hashing:64"})

    def test_there_is_no_default_tenant(self) -> None:
        """⭐ Một mặc định đóng lỗ theo **hướng sai**.

        Quên tenant hiện tại nghĩa là "không ai đọc được" — an toàn và ồn ào.
        Với mặc định `"public"` nó thành "tài liệu riêng của khách hàng nằm
        trong kho công khai" — im lặng, và không thu hồi được.
        """
        assert IndexConfig.model_fields["tenant_id"].is_required()

    def test_the_tenant_is_not_part_of_the_fingerprint(self) -> None:
        """Nó là field payload, không chạm vector nào.

        Đưa nó vào `fingerprint` sẽ làm mọi bundle đã ký hỏng chữ ký vì một lý
        do không liên quan tới chất lượng truy hồi — và buộc build lại 15.814
        chunk để đổi đúng một chuỗi trong payload.
        """
        assert _config(tenant_id="a").fingerprint == _config(tenant_id="b").fingerprint
        assert (
            _config(tenant_id="a").chunking_fingerprint
            == _config(tenant_id="b").chunking_fingerprint
        )

    @pytest.mark.parametrize("bad", ["", "có dấu", "a b", "x" * 65, "a/b"])
    def test_a_tenant_that_would_not_survive_a_payload_filter_is_refused(self, bad: str) -> None:
        """Khớp `String(64)` của `serving/db/models.py` và `MatchValue` của Qdrant.

        Một tenant chỉ khác nhau ở khoảng trắng hoặc dài quá cột DB là chỗ mà
        hai tầng cùng "đúng" mà không khớp nhau.
        """
        with pytest.raises(ValidationError):
            _config(tenant_id=bad)

    def test_the_tenant_reaches_the_store_that_build_retriever_makes(self) -> None:
        """Dòng nối duy nhất giữa config và payload.

        Không có test này thì `tenant_id` bắt buộc trong config vẫn đúng, mọi
        test `_payload` vẫn xanh, và index dựng ra vẫn không có tenant — tức
        `TD-40` quay lại nguyên vẹn qua một chỗ không ai nhìn.
        """
        config = _config(tenant_id="acme")
        store = config.build_retriever(config.build_embeddings(), url="http://127.0.0.1:6333")

        assert store.tenant_id == "acme"

    def test_every_shipped_index_config_declares_a_tenant(self) -> None:
        """⚠️ Phép kiểm phủ định phải kèm bằng chứng nó nhìn thấy gì (`W3-04`)."""
        configs = sorted(Path("configs/indexing").glob("*.yaml"))

        assert len(configs) >= 8, f"chỉ thấy {len(configs)} config — glob hỏng?"
        for path in configs:
            assert load_index_config(path).tenant_id


# ---------------------------------------------------------------------------
# `TD-82` — vân tay cũ vẫn đọc được, và state được nâng lên tại chỗ
# ---------------------------------------------------------------------------


class _CountingRetriever:
    """Đủ để `_reconcile_state` làm việc: nó chỉ hỏi `count()`."""

    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


def _state_for(config: IndexConfig, fingerprint: str) -> object:
    from pipeline.indexing.build_index import DocState, IndexState

    return IndexState(
        config_name=config.name,
        fingerprint=fingerprint,
        collection=config.collection_name,
        embedding_model=config.embedding_model,
        embedding_dim=1024,
        chunker_name="hybrid",
        documents={"doc": DocState(content_hash="a" * 64, n_chunks=3)},
    )


def _reconcile(config: IndexConfig, fingerprint: str) -> object:
    from pipeline.indexing.build_index import _reconcile_state

    return _reconcile_state(
        _state_for(config, fingerprint),  # type: ignore[arg-type]
        config,
        _CountingRetriever(3),  # type: ignore[arg-type]
        embedding_dim=1024,
        chunker_name="hybrid",
        allow_mixed=False,
    )


def test_a_pre_td82_state_file_is_accepted_and_upgraded() -> None:
    """⭐⭐ Bản vá `TD-82` không được biến một index hợp lệ thành index bị từ chối.

    State file của collection đang nằm trong Qdrant được ghi trên Windows bằng
    công thức cũ. Sau khi sửa công thức, `state.fingerprint != config.fingerprint`
    — và luật "hai fingerprint trong một collection thì dừng" sẽ chặn lần build
    kế tiếp, dù index ấy **đúng** là do config này sinh ra.

    Chấp nhận thôi chưa đủ: state phải được **nâng lên** giá trị mới, nếu không
    thì dòng cảnh báo ấy vĩnh viễn và giá trị cũ không bao giờ biến mất.
    """
    config = load_index_config(Path("configs/indexing/bgem3-contextual.yaml"))
    reconciled = _reconcile(config, config.legacy_windows_fingerprint)
    assert reconciled.fingerprint == config.fingerprint  # type: ignore[attr-defined]
    assert reconciled.documents  # type: ignore[attr-defined]  # state cũ được giữ, không bị vứt


def test_a_genuinely_different_config_still_stops_the_build() -> None:
    """Nới lỏng phải hẹp: chỉ nhận đúng biến thể đường dẫn, không nhận mọi thứ."""
    config = load_index_config(Path("configs/indexing/bgem3-contextual.yaml"))
    with pytest.raises(RuntimeError, match="fingerprint"):
        _reconcile(config, "0" * 64)
