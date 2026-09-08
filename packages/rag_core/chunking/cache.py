"""Cache chunk trên SQLite, thay cho cache `pickle` của bản POC.

Ba lý do bỏ `pickle`:

1. **An toàn.** `pickle.load` thực thi mã tuỳ ý khi giải mã. Bản POC nạp thẳng
   file `.pkl` từ thư mục cache và chỉ bọc `try/except` — nghĩa là bất kỳ ai ghi
   được vào `.chunking_cache/` đều chạy được code trong tiến trình. Cache là dữ
   liệu, phải lưu ở định dạng dữ liệu: ở đây là JSON đã validate bằng pydantic.
2. **Đồng thời.** Mỗi entry một file `.pkl` không có khoá; hai worker cùng ghi
   thì hỏng file. SQLite lo giao dịch.
3. **Vận hành được.** Có TTL, có eviction LRU, đếm được hit/miss. Bản cũ chỉ
   phình mãi cho tới khi hết đĩa.

Khác biệt quan trọng thứ tư: **cache theo từng tài liệu**, không theo cả corpus.
Bản POC hash chuỗi nối của mọi tài liệu, nên sửa một tài liệu là mất sạch cache
của tất cả. Cache theo tài liệu chính là nền cho re-index tăng dần ở `W3-07`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from ..schemas import Chunk, Document
from .base import Chunker
from .pieces import TextPiece

__all__ = ["CACHE_TABLE", "CacheStats", "CachedChunker", "SQLiteChunkCache"]

logger = logging.getLogger(__name__)

_CHUNK_LIST = TypeAdapter(list[Chunk])

# Version nằm trong TÊN BẢNG, không phải trong một cột.
#
# Cần version vì `Chunk` có thể thêm field **optional** — `start_char`/`end_char`
# của `W1-11` là ví dụ. Thêm field optional thì `TypeAdapter.validate_json` trên
# payload cũ **không** báo lỗi, nó chỉ điền `None`. Nên cái guard `ValidationError`
# ở `get()` không bắt được, và cache cũ sẽ lặng lẽ trả về chunk không có offset —
# đúng loại hỏng âm thầm mà cả `W1-08` lẫn `W1-11` đã mất thời gian truy.
#
# Đổi tên bảng thì entry cũ trở thành không tồn tại (miss, tính lại) chứ không
# phải hỏng, và bảng cũ bị DROP nên file cache không phình mãi.
_PAYLOAD_VERSION = 2
CACHE_TABLE = f"chunk_cache_v{_PAYLOAD_VERSION}"

_SCHEMA = f"""
DROP TABLE IF EXISTS chunk_cache;
CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
    content_hash  TEXT    NOT NULL,
    config_hash   TEXT    NOT NULL,
    chunker_name  TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    size_bytes    INTEGER NOT NULL,
    created_at    REAL    NOT NULL,
    last_used_at  REAL    NOT NULL,
    hit_count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (content_hash, config_hash, chunker_name)
);
CREATE INDEX IF NOT EXISTS idx_{CACHE_TABLE}_lru ON {CACHE_TABLE} (last_used_at);
"""


@dataclass(frozen=True)
class CacheStats:
    entries: int
    total_bytes: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class SQLiteChunkCache:
    """Cache khoá theo `(content_hash, config_hash, chunker_name)`.

    Có cả `chunker_name` trong khoá vì hai chunker khác nhau dùng chung một
    `ChunkingConfig` vẫn cho ra kết quả khác nhau (semantic còn phụ thuộc model
    embedding). Thiếu nó thì ablation sẽ đọc nhầm kết quả của nhánh khác.
    """

    def __init__(
        self,
        path: str | Path = ".cache/chunks.sqlite3",
        *,
        ttl_seconds: float | None = None,
        max_entries: int = 20_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """⚠️ `clock` là một mối nối, không phải một tuỳ chọn cấu hình.

        `ttl_seconds` đo theo **giờ tường**, nên mọi phép kiểm TTL viết bằng
        `sleep()` đều là một cuộc đua với chính máy chạy nó: bài
        `test_ttl_expires_entry` dùng `ttl=0,05 s` rồi khẳng định entry vừa ghi
        **còn sống** — trên runner CI khựng 50 ms giữa `put` và `get` thì nó đã
        hết hạn, và bài test đỏ vì môi trường chứ không vì mã. Đã xảy ra thật
        (`ca457c1`, 08/09/2026, trên một commit **chỉ đụng hai file `.md`**).

        Tham số này để test cầm lấy đồng hồ. Production không truyền gì.
        """
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self.hits = 0
        self.misses = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------ nội bộ

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.DatabaseError:
            # File cache hỏng (đứt điện giữa lúc ghi, hoặc là file khác trùng tên).
            # Cache là dữ liệu tái sinh được — xoá và dựng lại, không làm sập ứng dụng.
            logger.warning("File cache %s hỏng, tạo lại từ đầu", self.path)
            self.path.unlink(missing_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)

    # ------------------------------------------------------------ công khai

    def get(self, content_hash: str, config_hash: str, chunker_name: str) -> list[Chunk] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload, created_at FROM {CACHE_TABLE} "
                "WHERE content_hash=? AND config_hash=? AND chunker_name=?",
                (content_hash, config_hash, chunker_name),
            ).fetchone()

            if row is None:
                self.misses += 1
                return None

            payload, created_at = row
            if self.ttl_seconds is not None and self._clock() - created_at > self.ttl_seconds:
                conn.execute(
                    f"DELETE FROM {CACHE_TABLE} "
                    "WHERE content_hash=? AND config_hash=? AND chunker_name=?",
                    (content_hash, config_hash, chunker_name),
                )
                self.misses += 1
                return None

            try:
                chunks = _CHUNK_LIST.validate_json(payload)
            except (ValidationError, json.JSONDecodeError):
                # Entry hỏng hoặc schema Chunk đã đổi. Coi như miss và dọn đi.
                logger.warning("Entry cache hỏng cho %s, xoá và tính lại", content_hash[:12])
                conn.execute(
                    f"DELETE FROM {CACHE_TABLE} "
                    "WHERE content_hash=? AND config_hash=? AND chunker_name=?",
                    (content_hash, config_hash, chunker_name),
                )
                self.misses += 1
                return None

            conn.execute(
                f"UPDATE {CACHE_TABLE} SET last_used_at=?, hit_count=hit_count+1 "
                "WHERE content_hash=? AND config_hash=? AND chunker_name=?",
                (self._clock(), content_hash, config_hash, chunker_name),
            )
            self.hits += 1
            return chunks

    def put(
        self,
        content_hash: str,
        config_hash: str,
        chunker_name: str,
        chunks: Sequence[Chunk],
    ) -> None:
        payload = _CHUNK_LIST.dump_json(list(chunks)).decode("utf-8")
        now = self._clock()
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {CACHE_TABLE} "
                "(content_hash, config_hash, chunker_name, payload, size_bytes, "
                " created_at, last_used_at, hit_count) VALUES (?,?,?,?,?,?,?,0)",
                (
                    content_hash,
                    config_hash,
                    chunker_name,
                    payload,
                    len(payload.encode("utf-8")),
                    now,
                    now,
                ),
            )
            self._evict(conn)

    def _evict(self, conn: sqlite3.Connection) -> None:
        (count,) = conn.execute(f"SELECT COUNT(*) FROM {CACHE_TABLE}").fetchone()
        if count <= self.max_entries:
            return
        conn.execute(
            f"DELETE FROM {CACHE_TABLE} WHERE rowid IN ("
            f"  SELECT rowid FROM {CACHE_TABLE} ORDER BY last_used_at ASC LIMIT ?"
            ")",
            (count - self.max_entries,),
        )

    def stats(self) -> CacheStats:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM {CACHE_TABLE}"
            ).fetchone()
        return CacheStats(entries=row[0], total_bytes=row[1], hits=self.hits, misses=self.misses)

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute(f"DELETE FROM {CACHE_TABLE}")
        self.hits = 0
        self.misses = 0


class CachedChunker(Chunker):
    """Bọc một `Chunker` bất kỳ bằng cache, tính theo từng tài liệu."""

    def __init__(self, inner: Chunker, cache: SQLiteChunkCache) -> None:
        super().__init__(inner.config)
        self.inner = inner
        self.cache = cache
        self.strategy = inner.strategy  # type: ignore[misc]

    @property
    def name(self) -> str:
        return self.inner.name

    def split_pieces(self, text: str) -> list[TextPiece]:
        return self.inner.split_pieces(text)

    def prepare(self, n_documents: int) -> None:
        super().prepare(n_documents)
        self.inner.prepare(n_documents)

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        # Chia nhỏ theo từng tài liệu là chuyện của cache, không phải của
        # chunker: nếu người gọi chưa khai báo thì lấy kích thước lô này.
        if self.inner.planned_documents is None:
            self.inner.prepare(len(documents))
        config_hash = self.config.config_hash
        # Đọc `name` sau `prepare`: với hybrid, tên đổi theo nhánh đã chọn.
        chunker_name = self.inner.name
        out: list[Chunk] = []
        for doc in documents:
            cached = self.cache.get(doc.content_hash, config_hash, chunker_name)
            if cached is not None:
                out.extend(cached)
                continue
            produced = self.inner.chunk([doc])
            self.cache.put(doc.content_hash, config_hash, chunker_name, produced)
            out.extend(produced)
        return out
