"""Hợp đồng dữ liệu dùng chung cho cả hai plane.

Đây là *thứ duy nhất* mà pipeline và serving thống nhất với nhau về hình dạng dữ
liệu. Vì vậy mọi model ở đây đều:

* `extra="forbid"` — payload thừa field là lỗi, không im lặng bỏ qua. Một field
  gõ sai tên mà bị nuốt sẽ thành bug âm thầm trong index đã build xong.
* `frozen=True` với `Document`/`Chunk` — sau khi tạo thì bất biến, để hash nội
  dung luôn khớp với nội dung.
* round-trip được: `model_validate_json(x.model_dump_json()) == x`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "LICENSE_ALLOWLIST",
    "Answer",
    "Chunk",
    "Citation",
    "DocType",
    "Document",
    "DocumentMetadata",
    "Language",
    "QueryRequest",
    "RetrievalMode",
    "RetrievedChunk",
    "TextSpan",
    "TokenUsage",
]

NonEmptyStr = Annotated[str, Field(min_length=1)]

LICENSE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "CC BY 4.0",
        "CC BY 3.0",
        "CC BY 3.0 IGO",
        "CC BY-SA 4.0",
        "CC BY-NC 4.0",
        "CC BY-NC-SA 4.0",
        "CC0 1.0",
        "Public Domain",
        "OGL v3",
        "Vietnam Government Work",
    }
)
"""Giấy phép cho phép redistribute **và** cho phép tạo tác phẩm phái sinh.

Sống ở `rag_core` từ `NEW-11` (2026-09-08) vì nó là **từ vựng chung hai plane**:
Pipeline Plane cưỡng chế nó ở cửa manifest (`pipeline.corpus.manifest`, nơi nó
sinh ra), còn Serving Plane cần nó để từ chối sớm một upload mang giấy phép
ngoài danh sách — và serving **không được import pipeline**
(`test_architecture_boundaries`). `pipeline.corpus.manifest` re-export nguyên
tên, mọi người dùng cũ giữ nguyên câu import.

`CC BY-NC*` được chấp nhận vì dự án phi thương mại; nếu sau này đem đi thương
mại hoá thì phải rà lại danh sách này trước. Giấy phép có `ND` (NoDerivatives)
bị từ chối vì chunking + sinh context bằng LLM là tạo tác phẩm phái sinh.
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def sha256_of(text: str) -> str:
    """Hash nội dung đã chuẩn hoá — dùng cho cache key và dedupe.

    Chuẩn hoá xuống dòng và bỏ khoảng trắng thừa ở hai đầu mỗi dòng để cùng một
    tài liệu tải lại từ nguồn khác (CRLF vs LF) không sinh hash khác nhau.
    """
    normalized = "\n".join(line.strip() for line in text.replace("\r\n", "\n").split("\n"))
    return hashlib.sha256(normalized.strip().encode("utf-8")).hexdigest()


class Language(StrEnum):
    """Ngôn ngữ chính của tài liệu/chunk/truy vấn."""

    VI = "vi"
    EN = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DocType(StrEnum):
    """Loại tài liệu — dùng để breakdown metric và để lọc metadata.

    Ba giá trị đầu ứng với ba nguồn corpus đã chốt: mỗi nguồn phục vụ một nhóm
    metric khác nhau, nên breakdown theo `doc_type` là đầu ra chính của eval.
    """

    DEV_REPORT = "dev_report"
    """Báo cáo tổ chức phát triển (World Bank / ADB) — prose dài, song ngữ."""

    LEGAL = "legal"
    """Văn bản pháp luật — heading nhiều cấp, kiểm chứng `section_path`."""

    ANNUAL_REPORT = "annual_report"
    """Báo cáo thường niên doanh nghiệp — nhiều bảng, kiểm chứng `table_lookup`."""

    OTHER = "other"


class RetrievalMode(StrEnum):
    """Nhánh truy hồi đã sinh ra kết quả — cần cho việc phân tích ablation."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class DocumentMetadata(BaseModel):
    """Xuất xứ của một tài liệu.

    `source_url` và `license` là **bắt buộc**: corpus phải công khai và cho phép
    redistribute, vì repo public + demo public + máy GPU thuê đều là kênh công
    bố dữ liệu. Bắt buộc ở tầng schema thì không thể quên khi thêm tài liệu mới.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: NonEmptyStr
    license: NonEmptyStr
    source_path: str | None = None
    title: str | None = None
    lang: Language = Language.UNKNOWN
    doc_type: DocType = DocType.OTHER
    published_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=_utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Một tài liệu nguồn sau khi đã trích xuất text, trước khi chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: NonEmptyStr
    content: NonEmptyStr
    metadata: DocumentMetadata

    @property
    def content_hash(self) -> str:
        return sha256_of(self.content)


class TextSpan(BaseModel):
    """Một vùng ký tự trong **văn bản gốc** của một tài liệu.

    Đây là cách neo bằng chứng độc lập với cấu hình chunking. `chunk_id` của dự
    án này là `f"{doc_id}::{index:05d}"` — thuần vị trí. Đổi `chunk_size` là mọi
    id cũ trỏ vào văn bản khác, mà id vẫn tồn tại nên không phép kiểm nào cảnh
    báo (đo thật: 1000→600 ký tự cho 46/46 id còn sống, 0 id giữ nguyên nội
    dung). Golden set gán nhãn bằng span thì đổi chunking bao nhiêu lần cũng
    không phải gán lại — xem `TD-12`.

    Neo được vào văn bản gốc là vì nó **bất biến**: `data/corpus_manifest.csv`
    ghi sha256 từng tài liệu và `iter_documents` kiểm lại mỗi lần build index.

    `end` là **loại trừ**, theo đúng quy ước slice của Python.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: NonEmptyStr
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError(f"{self.doc_id}: span rỗng hoặc đảo ngược ({self.start}, {self.end})")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: TextSpan) -> int:
        """Số ký tự chồng nhau. 0 nếu khác tài liệu hoặc không giao nhau."""
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


class Chunk(BaseModel):
    """Một đơn vị được embed và index.

    `section_path` là đường dẫn heading (ví dụ
    `["Chương II", "Điều 15", "Khoản 2"]`). Nó có mặt ngay từ schema nền dù
    structure-aware chunker mãi tới W3 mới sinh ra được — chunker cũ để rỗng.
    Đặt trước như vậy để lên W3 không phải migrate lại toàn bộ index.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: NonEmptyStr
    doc_id: NonEmptyStr
    content: NonEmptyStr
    chunk_index: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
    parent_chunk_id: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    metadata: DocumentMetadata | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    """Vùng của `Document.content` mà chunk này được **dẫn ra từ**.

    ⚠️ Cố ý **không** phải chỉ dẫn cắt: trong trường hợp chung
    `content != document.content[start_char:end_char]`. Splitter đệ quy bỏ mảnh
    rỗng (`[s for s in text.split(sep) if s]`) rồi nối lại bằng separator, nên
    `"A\\n\\nB"` tách theo `"\\n"` cho ra `"A\\nB"` — ngắn hơn nguyên bản một ký
    tự. Splitter ngữ nghĩa thì nối câu bằng dấu cách, bất kể nguyên bản ngăn
    nhau bằng gì.

    Ép `content` thành substring nguyên văn sẽ **đổi nội dung chunk**, tức đổi
    cả index và mọi con số baseline. Nên hợp đồng đúng là: span là **vùng xuất
    xứ**, có thể rộng hơn `content` vài ký tự khoảng trắng. Đủ để ánh xạ
    span↔chunk qua mọi cấu hình chunking, và đó là toàn bộ mục đích của nó.

    `None` với point ghi trước `W1-11` (lúc đó chunker chưa sinh offset).
    """

    @property
    def content_hash(self) -> str:
        return sha256_of(self.content)

    @property
    def span(self) -> TextSpan | None:
        """Span của chunk, `None` nếu chunker không sinh offset."""
        if self.start_char is None or self.end_char is None:
            return None
        return TextSpan(doc_id=self.doc_id, start=self.start_char, end=self.end_char)

    @property
    def section_header(self) -> str:
        """Chuỗi heading để prepend vào text lúc embed."""
        return " > ".join(self.section_path)


class RetrievedChunk(BaseModel):
    """Một chunk kèm điểm số của lần truy hồi cụ thể.

    Giữ `dense_score`/`sparse_score`/`rerank_score` tách riêng thay vì chỉ một
    `score` tổng: khi phân tích ablation cần biết nhánh nào đã kéo chunk lên,
    và RRF làm mất thông tin đó nếu không lưu lại.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    mode: RetrievalMode = RetrievalMode.DENSE
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """Một trích dẫn trong câu trả lời, kèm kết quả xác minh.

    `verified=False` nghĩa là `quote` **không** tìm thấy trong chunk được cite —
    tức mô hình đã bịa. Câu trả lời vẫn trả về nhưng phải đánh dấu, không được
    im lặng bỏ qua.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: NonEmptyStr
    doc_id: NonEmptyStr
    quote: NonEmptyStr
    verified: bool = False
    source_url: str | None = None
    section_path: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    """Đầu ra của serving plane.

    `model` là **model thực tế đã phục vụ request**, đọc từ response của
    provider — không phải model mình yêu cầu. Router có fallback, và một metric
    dịch chuyển vì âm thầm rơi sang model khác là loại bug rất khó truy.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    model: NonEmptyStr
    prompt_version: str | None = None
    bundle_version: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_refusal_consistency(self) -> Answer:
        if self.refused and self.citations:
            raise ValueError("Câu trả lời đã từ chối thì không được kèm citation")
        if not self.refused and not self.text.strip():
            raise ValueError("Câu trả lời không từ chối thì `text` không được rỗng")
        return self


class QueryRequest(BaseModel):
    """Đầu vào của serving plane."""

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyStr
    top_k: int = Field(default=10, ge=1, le=200)
    conversation_id: str | None = None
    tenant_id: str | None = None
    lang: Language | None = None
    doc_types: list[DocType] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query không được chỉ gồm khoảng trắng")
        return stripped
