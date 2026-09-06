"""Cấu hình một lần build index — tiền thân của `RagBundle`.

Toàn bộ thứ quyết định **nội dung** của một index nằm gọn trong một file YAML:
corpus nào, chunk thế nào, embed bằng model gì, ghi vào collection nào. Lý do
gom lại một chỗ thay vì rải thành cờ dòng lệnh:

* Một lần chạy eval phải tái lập được từ một file. `--chunk-size 1000
  --overlap 100 --model ...` gõ tay lần sau sẽ lệch, và không ai biết là đã lệch.
* `fingerprint` băm đúng những trường ảnh hưởng tới vector. Đây là thứ để trả
  lời "index đang nằm trong Qdrant có phải do config này sinh ra không" — câu
  hỏi bắt buộc trước khi so hai con số eval với nhau.
* W4 sẽ đóng gói chính những trường này thành `RagBundle` có version. Định hình
  đúng từ bây giờ thì lúc đó chỉ là thêm chữ ký và metadata, không phải viết lại.

Cố ý **không** đưa `device` và `batch_size` vào `fingerprint`: chạy trên GPU thuê
hay trên laptop phải ra cùng một index về mặt logic, nếu không thì mọi lần đổi
máy đều buộc build lại toàn bộ. Chúng vẫn được ghi vào manifest để tra cứu.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
)

from rag_core.chunking import Chunker, ChunkingConfig, SQLiteChunkCache, build_chunker
from rag_core.chunking.cache import CachedChunker
from rag_core.embedding import EmbeddingProvider, build_embedding_provider

if TYPE_CHECKING:
    from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

__all__ = ["LEGACY_WINDOWS_PATHS", "FingerprintStatus", "IndexConfig", "load_index_config"]

logger = logging.getLogger(__name__)

LEGACY_WINDOWS_PATHS = "legacy_windows_paths"
"""Khoá `context` của Pydantic bật lại **cách serialise đường dẫn trước `TD-82`**.

Chỉ có một chỗ dùng nó: `IndexConfig.legacy_windows_fingerprint`. Nó tồn tại để
tính lại **đúng** giá trị mà công thức cũ đã ghi vào các artifact trên máy
Windows, chứ không phải để ai đó chọn kiểu serialise.
"""

FingerprintStatus = Literal["current", "legacy", "mismatch"]


class ContextualIndexConfig(BaseModel):
    """Cấu hình phía **tiêu thụ** của `W3-04`: dán ngữ cảnh đã sinh vào chunk.

    Tách khỏi `ContextualConfig` của `rag_core` một cách có chủ ý: cái kia mô tả
    **cách sinh** ngữ cảnh (prompt, cửa sổ, batch), cái này mô tả **cách dùng**
    artifact đã sinh. Build index không cần biết prompt nào đã tạo ra nó.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    """Mặc định **tắt**, vì bật là đổi mọi vector — phải là lựa chọn có ý thức."""

    contexts_path: Path = Path("data/contexts/contexts.jsonl")

    min_coverage: float = Field(default=0.95, ge=0.0, le=1.0)
    """Dưới ngưỡng này thì build **dừng** thay vì lặng lẽ index một nửa có ngữ cảnh.

    `apply_contexts` cố ý giữ nguyên chunk khi thiếu — nửa "fail 1 chunk không
    làm sập cả job" của DoD. Nhưng thiếu 1 chunk và thiếu 8.000 chunk trông
    giống hệt nhau ở phía build, nên ranh giới giữa hai thứ đó phải khai ra."""

    require_fingerprint: bool = True
    """Đòi artifact khai đúng vân tay cấu hình chunk — xem `chunking_fingerprint`."""

    @field_serializer("contexts_path")
    def _serialise_contexts_path(self, value: Path, info: SerializationInfo) -> str:
        """⭐⭐ `TD-82`: đường dẫn ra JSON theo **POSIX**, không theo hệ điều hành.

        `Path` serialise ra `data\\contexts\\contexts.jsonl` trên Windows và
        `data/contexts/contexts.jsonl` trên POSIX. Bình thường đó là chuyện hiển
        thị — nhưng chuỗi này đi thẳng vào `IndexConfig.fingerprint`, tức **cùng
        một config cho hai vân tay tuỳ theo máy nào tính nó**. Trên đúng cái
        trường tồn tại để chứng minh *"index này được build bằng config này"*.

        Vô hình suốt `W1`…`W5-08` vì mọi lần build đều chạy trên cùng một máy;
        lượt CI đầu tiên trên Linux (`W5-09`) làm nó lộ ra.

        `as_posix()` chứ không `str()`: POSIX là dạng chuẩn duy nhất mà cả hai
        nền tảng cùng viết ra được, và trên Linux nó **không đổi giá trị** — nên
        mọi artifact từng sinh trên Linux vẫn khớp. Chỉ artifact sinh trên
        Windows lệch, và `legacy_windows_fingerprint` tính lại được đúng chúng.
        """
        if (info.context or {}).get(LEGACY_WINDOWS_PATHS):
            return str(PureWindowsPath(value))
        return value.as_posix()


class IndexConfig(BaseModel):
    """Mô tả đầy đủ một lần build index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    """Tên run — dùng cho tên collection mặc định, file state và tên báo cáo."""

    collection: str = Field(default="", description="Rỗng thì suy ra từ `name`")

    tenant_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    """⭐⭐ Chủ sở hữu của mọi point mà lần build này ghi ra. `TD-40`.

    **Bắt buộc, không có mặc định**, và đó là điểm chính. `W4-04` áp
    `tenant_filter()` lên **mọi** request, còn Qdrant **không khớp** point thiếu
    field này — nên một index dựng mà quên tenant là một index **vô hình với mọi
    người dùng đã xác thực**. Nó không báo lỗi ở đâu cả: build xong, `/ready`
    xanh, `POST /chat` trả `sources: []`, và triệu chứng đọc được là *"hệ thống
    không tìm thấy gì"*.

    Đó đúng là chuyện đã xảy ra: 15.814 chunk của `rag_bgem3_ctx` được dựng qua
    ba tuần mà không có tenant nào, và nó chỉ lộ ra ở lần chạy thật đầu tiên của
    `W4-06` — khi cả ba tầng (`rag_core` filter → HTTP tenant → payload Qdrant)
    cùng chạy một lần.

    ⚠️ Một **mặc định** ở đây sẽ đóng lỗ theo hướng sai. Quên tenant thì lỗi
    hiện tại là "không ai đọc được" (an toàn, ồn ào); với mặc định `"public"` nó
    thành "tài liệu riêng của khách hàng nằm trong kho công khai" (im lặng, và
    không thu hồi được). Bắt phải khai là cách duy nhất giữ hướng hỏng đúng
    chiều.

    ⚠️ **Không** nằm trong `fingerprint`: nó là một field payload, không chạm
    vector nào — đúng lý lẽ đã cho phép `W2-06` backfill `published_at` mà mọi
    con số eval từ `W2-01` đến `W2-05` vẫn đúng nguyên. Có test ghim.
    """

    # ------------------------------------------------------------ nguồn
    manifest_path: Path = Path("data/corpus_manifest.csv")
    corpus_dir: Path = Path("data/corpus")
    languages: tuple[str, ...] = ()
    """Lọc theo ngôn ngữ; rỗng nghĩa là lấy hết."""
    doc_types: tuple[str, ...] = ()
    max_documents: int | None = Field(default=None, ge=1)

    # ------------------------------------------------------------ chunking
    chunking: ChunkingConfig = ChunkingConfig()

    # ------------------------------------------------------------ embedding
    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_max_batch_tokens: int | None = Field(default=None, ge=1)
    """Trần token mỗi forward pass, chỉ có ý nghĩa với provider cửa sổ dài.

    `embedding_batch_size` một mình không chặn được VRAM khi cửa sổ là 8192:
    16 câu × 8192 token = 131k token và OOM ngay. `None` = dùng mặc định của
    provider. Là knob **tốc độ/bộ nhớ** nên nằm ngoài `fingerprint`, cùng lý do
    như `device` và `batch_size`.
    """
    embedding_normalize: bool = True
    embedding_kwargs: dict[str, Any] = Field(default_factory=dict)
    """`query_prefix` / `document_prefix` cho model bất đối xứng (E5, BGE)."""

    # ------------------------------------------------------------ ghi
    contextual: ContextualIndexConfig = ContextualIndexConfig()
    """Dán ngữ cảnh định vị vào chunk trước khi embed (`W3-04`). Nằm trong `fingerprint`."""

    upsert_batch_size: int = Field(default=128, ge=1)
    use_cache: bool = True
    cache_path: Path = Path(".cache/chunks.sqlite3")
    state_dir: Path = Path(".cache/index_state")

    @field_validator("embedding_device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if value not in allowed:
            raise ValueError(f"embedding_device phải thuộc {sorted(allowed)}, nhận {value!r}")
        return value

    # ------------------------------------------------------------ dẫn xuất

    @property
    def collection_name(self) -> str:
        return self.collection or f"rag_{self.name}"

    @property
    def state_path(self) -> Path:
        return self.state_dir / f"{self.name}.json"

    @property
    def fingerprint(self) -> str:
        """Băm những trường quyết định nội dung vector.

        Không gồm `device`, `batch_size`, `upsert_batch_size`, đường dẫn cache —
        chúng đổi tốc độ chứ không đổi kết quả. Có gồm bộ lọc corpus vì chúng
        quyết định **tài liệu nào** có mặt trong index.
        """
        return self._fingerprint()

    @property
    def legacy_windows_fingerprint(self) -> str:
        """Giá trị mà công thức **trước `TD-82`** cho ra khi chạy trên Windows.

        Không phải một API để chọn dùng: nó có mặt để `fingerprint_status` phân
        biệt được *"artifact này ghi bằng công thức cũ trên Windows"* với
        *"artifact này thuộc về một index khác"*. Hai thứ đó cần hai câu trả lời
        khác nhau — một cái cảnh báo, một cái phải dừng.

        Trên Linux giá trị này **trùng** `fingerprint`, vì công thức cũ và mới
        chỉ khác nhau ở dấu phân cách mà POSIX vốn đã viết đúng.
        """
        return self._fingerprint(legacy_windows_paths=True)

    def fingerprint_status(self, recorded: str) -> FingerprintStatus:
        """Một vân tay đã ghi thuộc loại nào so với config hiện tại.

        `"legacy"` là câu trả lời mà `TD-82` bắt buộc phải có. Không có nó thì
        bản vá sẽ biến **mọi** artifact từng sinh trên Windows — ba manifest
        bundle, state file của index đang nằm trong Qdrant, `index_fingerprint`
        trong hơn hai chục file eval đã lưu — thành "không khớp", tức bản vá
        một lỗi im lặng lại tự tạo ra một lỗi ồn ào ở chỗ khác.

        Và nó cố ý **không** trả `True`/`False`: người gọi phải tự quyết cảnh
        báo hay dừng, vì "chấp nhận nhưng nói ra" chỉ đúng cho đường đọc lại
        artifact cũ, không đúng cho đường ghi artifact mới.
        """
        if recorded == self.fingerprint:
            return "current"
        if recorded == self.legacy_windows_fingerprint:
            return "legacy"
        return "mismatch"

    def _fingerprint(self, *, legacy_windows_paths: bool = False) -> str:
        payload = self._fingerprint_payload(legacy_windows_paths=legacy_windows_paths)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _fingerprint_payload(self, *, legacy_windows_paths: bool = False) -> dict[str, Any]:
        """Tách khỏi `_fingerprint` để test soi được **cái gì** đi vào hàm băm.

        `TD-82` là một trường lọt vào payload mang hình dạng của hệ điều hành.
        Bài test canh chuyện đó phải quét payload chứ không quét mã, nếu không
        thì trường path **tiếp theo** sẽ tái lập đúng lỗi này.
        """
        context = {LEGACY_WINDOWS_PATHS: True} if legacy_windows_paths else None
        return {
            "chunking": json.loads(self.chunking.model_dump_json(context=context)),
            "embedding_model": self.embedding_model,
            "embedding_normalize": self.embedding_normalize,
            "embedding_kwargs": self.embedding_kwargs,
            "languages": sorted(self.languages),
            "doc_types": sorted(self.doc_types),
            "max_documents": self.max_documents,
            "contextual": json.loads(self.contextual.model_dump_json(context=context)),
        }

    @property
    def chunking_fingerprint(self) -> str:
        """Băm những trường quyết định **bộ chunk** — `chunk_id` và nội dung của nó.

        Hẹp hơn `fingerprint`: bỏ `embedding_normalize`/`embedding_kwargs` vì
        chúng đổi *vector* chứ không đổi *chunk*. Giữ `embedding_model` vì
        `HybridChunker` mượn tokenizer của nó để đo kích thước.

        ⚠️ Tồn tại để chặn một lỗi im lặng: `chunk_id` là `doc::index`, nên ngữ
        cảnh sinh cho `chunk_size=1000` đem dán lên chunk của `chunk_size=550`
        vẫn **khớp id** trong khi nội dung khác hẳn. Coverage báo 100%, index
        nhận 15.814 câu mô tả sai đoạn, và không có gì đỏ.
        """
        payload = {
            "chunking": json.loads(self.chunking.model_dump_json()),
            "embedding_model": self.embedding_model,
            "languages": sorted(self.languages),
            "doc_types": sorted(self.doc_types),
            "max_documents": self.max_documents,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------ factory

    def build_embeddings(self) -> EmbeddingProvider:
        extra: dict[str, Any] = dict(self.embedding_kwargs)
        if self.embedding_max_batch_tokens is not None:
            extra["max_batch_tokens"] = self.embedding_max_batch_tokens
        return build_embedding_provider(
            self.embedding_model,
            device=self.embedding_device,
            batch_size=self.embedding_batch_size,
            normalize=self.embedding_normalize,
            **extra,
        )

    def build_chunker(self, embeddings: EmbeddingProvider) -> Chunker:
        chunker = build_chunker(self.chunking, embeddings)
        if not self.use_cache:
            return chunker
        return CachedChunker(chunker, SQLiteChunkCache(self.cache_path))

    def build_retriever(
        self,
        embeddings: EmbeddingProvider,
        *,
        url: str,
        api_key: str | None = None,
    ) -> QdrantDenseRetriever:
        from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

        return QdrantDenseRetriever(
            embeddings,
            collection=self.collection_name,
            url=url,
            api_key=api_key,
            tenant_id=self.tenant_id,
        )


def load_index_config(path: str | Path, **overrides: Any) -> IndexConfig:
    """Đọc config YAML. `overrides` để CLI ghi đè vài trường mà không sửa file."""
    import yaml

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Không thấy config index tại {source}. Mẫu có sẵn ở `configs/indexing/baseline.yaml`."
        )
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source} phải là một mapping YAML, đọc được {type(raw).__name__}")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return IndexConfig.model_validate(raw)
