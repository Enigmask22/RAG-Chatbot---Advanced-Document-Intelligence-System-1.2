"""W1-04 — cache SQLite thay cache `pickle`."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from rag_core.chunking import ChunkingConfig, ChunkingStrategy, FixedSizeChunker
from rag_core.chunking.cache import CACHE_TABLE, CachedChunker, SQLiteChunkCache
from rag_core.schemas import Chunk, Document, DocumentMetadata


@pytest.fixture
def cache(tmp_path: Path) -> SQLiteChunkCache:
    return SQLiteChunkCache(tmp_path / "chunks.sqlite3")


@pytest.fixture
def sample_chunks(metadata: DocumentMetadata) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"d::{i:05d}",
            doc_id="d",
            content=f"Nội dung {i}",
            chunk_index=i,
            metadata=metadata,
        )
        for i in range(3)
    ]


class TestHitMiss:
    def test_miss_then_hit(self, cache: SQLiteChunkCache, sample_chunks: list[Chunk]) -> None:
        assert cache.get("h1", "c1", "fixed") is None
        cache.put("h1", "c1", "fixed", sample_chunks)
        assert cache.get("h1", "c1", "fixed") == sample_chunks

    def test_miss_on_content_change(
        self, cache: SQLiteChunkCache, sample_chunks: list[Chunk]
    ) -> None:
        cache.put("h1", "c1", "fixed", sample_chunks)
        assert cache.get("h2", "c1", "fixed") is None

    def test_miss_on_config_change(
        self, cache: SQLiteChunkCache, sample_chunks: list[Chunk]
    ) -> None:
        cache.put("h1", "c1", "fixed", sample_chunks)
        assert cache.get("h1", "c2", "fixed") is None

    def test_miss_on_chunker_change(
        self, cache: SQLiteChunkCache, sample_chunks: list[Chunk]
    ) -> None:
        """Semantic và fixed dùng chung config vẫn cho kết quả khác nhau — thiếu
        `chunker_name` trong khoá thì ablation đọc nhầm kết quả của nhánh khác."""
        cache.put("h1", "c1", "fixed", sample_chunks)
        assert cache.get("h1", "c1", "semantic") is None

    def test_stats_track_hit_rate(
        self, cache: SQLiteChunkCache, sample_chunks: list[Chunk]
    ) -> None:
        cache.get("miss", "c", "fixed")
        cache.put("h1", "c", "fixed", sample_chunks)
        cache.get("h1", "c", "fixed")
        stats = cache.stats()
        assert (stats.hits, stats.misses, stats.entries) == (1, 1, 1)
        assert stats.hit_rate == 0.5


class TestRobustness:
    def test_corrupt_entry_recovers_as_miss(
        self, cache: SQLiteChunkCache, sample_chunks: list[Chunk]
    ) -> None:
        cache.put("h1", "c1", "fixed", sample_chunks)
        with sqlite3.connect(cache.path) as conn:
            conn.execute(f"UPDATE {CACHE_TABLE} SET payload = '{{ khong phai json'")
            conn.commit()

        assert cache.get("h1", "c1", "fixed") is None
        assert cache.stats().entries == 0, "entry hỏng phải bị dọn, không để lại rác"

    def test_corrupt_database_file_is_rebuilt(self, tmp_path: Path) -> None:
        path = tmp_path / "chunks.sqlite3"
        path.write_bytes(b"day khong phai file sqlite")
        cache = SQLiteChunkCache(path)  # không được ném lỗi
        assert cache.stats().entries == 0

    def test_rag_core_never_imports_pickle(self) -> None:
        """`pickle.load` chạy mã tuỳ ý khi giải mã — cache là dữ liệu, phải là JSON.

        Quét AST thay vì tìm chuỗi, để docstring nhắc tới `pickle` không làm test
        đỏ nhầm, và để bắt được cả `from pickle import loads`.
        """
        import ast

        package = Path(__file__).resolve().parents[2] / "packages" / "rag_core"
        offenders: list[str] = []
        for source_file in package.rglob("*.py"):
            tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Import)
                    and any(
                        alias.name.split(".")[0] in {"pickle", "cPickle", "dill"}
                        for alias in node.names
                    )
                ) or (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[0]
                    in {
                        "pickle",
                        "dill",
                    }
                ):
                    offenders.append(f"{source_file.name}:{node.lineno}")

        assert not offenders, f"rag_core không được dùng pickle: {offenders}"

    def test_ttl_expires_entry(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        """⚠️ Bản cũ đỏ trên CI ở `ca457c1` — một commit **chỉ đụng hai file
        `.md`**. Nó dùng `ttl=0,05 s` rồi khẳng định entry vừa ghi **còn sống**:
        runner khựng 50 ms giữa `put` và `get` là đủ để entry hết hạn, và bài
        test đỏ vì môi trường chứ không vì mã.

        Không nới `sleep` — mọi cặp số cố định vẫn để lại đúng cửa sổ ấy, chỉ
        hẹp hơn. Cầm lấy đồng hồ thì cửa sổ **biến mất**, và bài test nhanh hơn
        130 ms."""
        now = 1_000.0
        cache = SQLiteChunkCache(tmp_path / "c.sqlite3", ttl_seconds=60.0, clock=lambda: now)
        cache.put("h1", "c1", "fixed", sample_chunks)
        assert cache.get("h1", "c1", "fixed") is not None
        now += 60.001
        assert cache.get("h1", "c1", "fixed") is None

    def test_ttl_KHONG_het_han_ngay_truoc_moc(
        self, tmp_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Nhóm chứng: bài trên xanh vì TTL **hết hạn**, hay vì `get` luôn trả
        `None` sau lần đầu? Nhảy tới sát mốc rồi đòi entry còn sống."""
        now = 1_000.0
        cache = SQLiteChunkCache(tmp_path / "c.sqlite3", ttl_seconds=60.0, clock=lambda: now)
        cache.put("h1", "c1", "fixed", sample_chunks)
        now += 59.999
        assert cache.get("h1", "c1", "fixed") is not None

    def test_lru_eviction(self, tmp_path: Path, sample_chunks: list[Chunk]) -> None:
        cache = SQLiteChunkCache(tmp_path / "c.sqlite3", max_entries=2)
        for i in range(4):
            cache.put(f"h{i}", "c", "fixed", sample_chunks)
            time.sleep(0.01)  # để `last_used_at` khác nhau
        assert cache.stats().entries == 2
        assert cache.get("h0", "c", "fixed") is None
        assert cache.get("h3", "c", "fixed") is not None

    def test_clear(self, cache: SQLiteChunkCache, sample_chunks: list[Chunk]) -> None:
        cache.put("h1", "c1", "fixed", sample_chunks)
        cache.clear()
        assert cache.stats().entries == 0


class TestCachedChunker:
    def test_second_call_hits_cache(
        self, cache: SQLiteChunkCache, short_document: Document
    ) -> None:
        inner = FixedSizeChunker(ChunkingConfig(strategy=ChunkingStrategy.FIXED))
        chunker = CachedChunker(inner, cache)

        first = chunker.chunk([short_document])
        second = chunker.chunk([short_document])

        assert first == second
        assert cache.stats().hits == 1
        assert cache.stats().misses == 1

    def test_cache_is_per_document(
        self, cache: SQLiteChunkCache, short_document: Document, metadata: DocumentMetadata
    ) -> None:
        """Sửa một tài liệu không được làm mất cache của các tài liệu khác.

        Bản POC hash chuỗi nối của cả corpus nên đổi một file là mất sạch. Đây
        cũng là nền cho re-index tăng dần ở W3-07.
        """
        other = Document(doc_id="other", content="Một tài liệu khác hoàn toàn.", metadata=metadata)
        chunker = CachedChunker(FixedSizeChunker(ChunkingConfig()), cache)

        chunker.chunk([short_document, other])
        assert cache.stats().misses == 2

        edited = Document(doc_id="other", content="Nội dung đã bị sửa đi.", metadata=metadata)
        chunker.chunk([short_document, edited])

        stats = cache.stats()
        assert stats.hits == 1, "tài liệu không đổi phải hit"
        assert stats.misses == 3, "chỉ tài liệu bị sửa mới miss"

    def test_config_change_invalidates(
        self, cache: SQLiteChunkCache, short_document: Document
    ) -> None:
        a = CachedChunker(FixedSizeChunker(ChunkingConfig(chunk_size=500)), cache)
        b = CachedChunker(FixedSizeChunker(ChunkingConfig(chunk_size=1000)), cache)
        a.chunk([short_document])
        b.chunk([short_document])
        assert cache.stats().misses == 2
