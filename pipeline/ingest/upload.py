"""Cửa nhận tài liệu của Pipeline Plane — `NEW-11` (chốt 08/09/2026: CHO PHÉP).

## Ba câu hỏi của `NEW-11`, và câu trả lời nằm trong thiết kế

1. **Ai chịu trách nhiệm cho tài liệu tải lên?** Người tải — và endpoint bắt họ
   nói ra điều đó bằng dữ liệu, không phải bằng checkbox: `source_url` công khai
   là trường **bắt buộc** và `license` phải nằm trong `LICENSE_ALLOWLIST`. Tức
   upload ở đây không phải "đẩy file riêng của bạn vào index" — nó là *"thêm một
   tài liệu công khai vào sổ đăng ký corpus"*, đi qua **đúng cái cửa** mà mọi tài
   liệu World Bank đã đi (`CorpusEntry` validate license, `validate_manifest`
   chặn trùng). Quy tắc cứng "corpus phải công khai, license cho phép
   redistribute" không bị nút upload phá — nó được **cưỡng chế tại cửa**, và đó
   là lý do quyết định này chấp nhận được. Danh tính người tải (tenant/key của
   proxy `/admin/ingest/upload`) ghi vào `notes` của entry — sổ đăng ký nói được
   *ai* đưa tài liệu này vào.

2. **Có trộn vào cùng collection với corpus công khai không?** Vào cùng
   **manifest** (sau khi qua cùng cửa license thì nó *là* tài liệu corpus — file
   nằm ở `data/corpus/uploads/`), nhưng vào **index** thì đi qua đường ingest
   bình thường (`POST /ingest` với `doc_ids`), nên nó chịu đúng các ràng buộc
   sẵn có: index vào collection mà config trỏ tới. ⚠️ Ghi thêm điểm vào
   collection mà bundle đang phục vụ đã được eval trên đó sẽ làm lệch `n_chunks`
   — `/admin/bundle/reload` kế tiếp sẽ TỪ CHỐI (`TD-38`, và đó là hành vi đúng:
   con số eval của bundle nói về một corpus không còn tồn tại). Đường sản xuất
   là: upload → ingest → build bundle mới → gate → promote (`W5-10`) — tài liệu
   tới người dùng **qua một bundle đã đo**, đúng kiến trúc hai plane.

3. **Câu trả lời trích dẫn nó thì trích cái gì?** Như mọi tài liệu corpus:
   `title` + chunk + quote đã xác minh (`W4-09`) — vì sau cửa manifest nó không
   phải một loại tài liệu thứ hai. `source="upload"` trong manifest giữ được
   dấu vết nguồn gốc cho eval breakdown.

## Vì sao nhận JSON text, không nhận multipart file

Corpus của dự án là `.txt` UTF-8 (parser là hàm đồng nhất — `TD-22`), nên
"file" ở đây là một chuỗi văn bản, và một trường JSON đi qua được nguyên vẹn
mọi tầng đã có: `BodyLimitMiddleware` (`NEW-12`) đếm nó, schema `extra="forbid"`
kiểm nó, proxy chuyển tiếp nó không cần multipart parser — một bề mặt parse
mới trên đường nhận dữ liệu không tin được là thứ phải có lý do mới thêm.
PDF cần Docling (`W3-01`) — khi corpus có PDF thật thì cửa này mở rộng sau.

## Thứ tự ghi, và vì sao file trước manifest sau

Hai lần ghi không có transaction chung. Ghi **file trước, manifest sau**: nếu
manifest hỏng giữa chừng thì còn lại một file mồ côi trong `uploads/` — vô hại,
không ai đọc nó (mọi đường đọc corpus đi qua manifest, và `dvc_state` đếm được
nó là file lạ). Chiều ngược lại để lại một entry trỏ vào file không tồn tại —
`iter_documents` sẽ nổ cho MỌI lượt ingest sau, kể cả của tài liệu khác.
Manifest ghi qua file tạm + `os.replace` vì worker ingest có thể đang đọc nó
ở tiến trình khác.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline.corpus.manifest import CorpusEntry, load_manifest, slugify, write_manifest
from rag_core.schemas import LICENSE_ALLOWLIST, DocType, Language

from .schemas import _SAFE_NAME, resolve_config

if TYPE_CHECKING:
    from pipeline.indexing.config import IndexConfig

__all__ = [
    "MAX_UPLOAD_BYTES",
    "UPLOAD_SUBDIR",
    "DuplicateUpload",
    "UploadReceipt",
    "UploadRequest",
    "receive_upload",
]

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 512 * 1024
"""Trần nội dung, tính bằng **byte UTF-8** chứ không phải ký tự — tiếng Việt có
dấu là 2–3 byte/ký tự và trần thân request (`NEW-12`, mặc định 1 MiB) đếm byte.
512 KiB nội dung + phong bì JSON nằm chắc dưới 1 MiB, nên một upload hợp lệ ở
đây không bao giờ chết ngang ở middleware với một thông điệp khác."""

UPLOAD_SUBDIR = "uploads"
"""Thư mục con trong `corpus_dir`. Tách khỏi tài liệu tải tự động để `dvc add`
và người đọc repo thấy được ranh giới nguồn gốc bằng mắt."""


class DuplicateUpload(Exception):
    """Nội dung hoặc `doc_id` đã có trong manifest — 409, không phải 400."""


class UploadRequest(BaseModel):
    """Một tài liệu ứng viên cho sổ đăng ký corpus.

    Mọi trường bảo mật kiểm ở **schema** (chạy cả khi handler được gọi thẳng),
    không phải ở handler.
    """

    model_config = ConfigDict(extra="forbid")

    config: str = Field(min_length=1, max_length=63)
    """Tên config indexing — nguồn duy nhất của `manifest_path`/`corpus_dir`.
    Cùng ràng buộc TÊN-không-phải-đường-dẫn với `IngestRequest`."""

    title: str = Field(min_length=3, max_length=300)

    content: str = Field(min_length=1, max_length=MAX_UPLOAD_BYTES)
    """`max_length` đếm ký tự — chỉ là chặn sớm rẻ tiền; trần thật là byte,
    kiểm ở validator dưới."""

    license: str = Field(min_length=1, max_length=100)
    source_url: str = Field(min_length=1, max_length=1000)
    license_url: str = Field(default="", max_length=1000)
    lang: Language = Language.UNKNOWN
    doc_type: DocType = DocType.OTHER
    notes: str = Field(default="", max_length=500)
    uploaded_by: str = Field(default="", max_length=200)
    """Proxy `/admin/ingest/upload` điền tenant/key đã xác thực. Người gọi thẳng
    dịch vụ này (loopback/token — `AU-10`) tự khai, và họ vốn đã là operator."""

    @field_validator("config")
    @classmethod
    def _name_not_path(cls, value: str) -> str:
        if not _SAFE_NAME.match(value):
            raise ValueError(f"config phải là TÊN (chữ thường, số, `-`, `_`), nhận {value!r}.")
        return value

    @field_validator("license")
    @classmethod
    def _license_must_be_allowed(cls, value: str) -> str:
        # `CorpusEntry` kiểm lại lần nữa — đây là bản sao CÓ CHỦ ĐÍCH để lỗi nổ
        # ở tầng 422 với thông điệp về upload, trước khi chạm vào đĩa.
        if value not in LICENSE_ALLOWLIST:
            raise ValueError(
                f"giấy phép {value!r} không nằm trong danh sách cho phép "
                f"redistribute + phái sinh. Chỉ nhận: {sorted(LICENSE_ALLOWLIST)}"
            )
        return value

    @field_validator("content")
    @classmethod
    def _content_within_byte_budget(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("nội dung toàn khoảng trắng")
        size = len(value.encode("utf-8"))
        if size > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"nội dung {size} byte UTF-8, trần {MAX_UPLOAD_BYTES} byte "
                "(tiếng Việt có dấu là 2–3 byte mỗi ký tự)"
            )
        return value


class UploadReceipt(BaseModel):
    """Biên nhận — đủ để gọi `POST /ingest` với `doc_ids=[doc_id]` ngay sau."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    relative_path: str
    sha256: str
    bytes: int
    manifest_entries: int


_MANIFEST_LOCK = threading.Lock()
"""Đọc-sửa-ghi manifest không nguyên tử. Dịch vụ ingest chạy MỘT tiến trình
(`AU-10`: loopback), nên một lock tiến trình là đủ; nhiều tiến trình ghi cùng
một manifest là một quyết định deploy chưa ai lấy, và lúc lấy thì lock này phải
thành lock file."""


def receive_upload(request: UploadRequest) -> UploadReceipt:
    """Ghi tài liệu vào `corpus_dir/uploads/` và đăng ký vào manifest.

    Ném `FileNotFoundError`/`ValueError` cho đầu vào sai (config không tồn tại,
    manifest chưa có), `DuplicateUpload` khi nội dung/doc_id đã đăng ký.
    """
    config = _load_config(request.config)
    payload = request.content.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    doc_id = f"up-{slugify(request.title, max_length=40)}-{digest[:12]}"
    relative_path = f"{UPLOAD_SUBDIR}/{doc_id}.txt"

    who = f"upload bởi {request.uploaded_by}" if request.uploaded_by else "upload"
    entry = CorpusEntry(
        doc_id=doc_id,
        relative_path=relative_path,
        source_url=request.source_url,
        license=request.license,
        license_url=request.license_url,
        sha256=digest,
        bytes=len(payload),
        source="upload",
        title=request.title,
        lang=request.lang,
        doc_type=request.doc_type,
        fetched_at=CorpusEntry.now_iso(),
        notes=f"{who}; {request.notes}"[:500] if request.notes else who,
        # `TD-22`: với `.txt` phép parse là hàm đồng nhất nên hai cột trùng nhau.
        text_sha256=digest,
    )

    with _MANIFEST_LOCK:
        entries = load_manifest(config.manifest_path)
        if not entries:
            raise ValueError(
                f"manifest {config.manifest_path} rỗng hoặc chưa tồn tại — upload "
                "chỉ THÊM vào một corpus đã đăng ký, không tạo corpus mới. Một "
                "config gõ nhầm tên sẽ chết ở đây thay vì sinh một manifest song song."
            )
        for existing in entries:
            if existing.sha256 == digest:
                raise DuplicateUpload(f"nội dung trùng tài liệu đã đăng ký: {existing.doc_id!r}")
            if existing.doc_id == doc_id:  # pragma: no cover - đòi va chạm sha 12 hex
                raise DuplicateUpload(f"doc_id {doc_id!r} đã tồn tại")

        target = Path(config.corpus_dir) / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        try:
            _write_manifest_atomically(config.manifest_path, [*entries, entry])
        except BaseException:
            # Không để lại file mồ côi khi chính lượt này thất bại — xem
            # docstring module về thứ tự ghi.
            target.unlink(missing_ok=True)
            raise

    logger.info(
        "upload: %s (%d byte, %s) vào %s — manifest %d tài liệu",
        doc_id,
        len(payload),
        request.license,
        config.manifest_path,
        len(entries) + 1,
    )
    return UploadReceipt(
        doc_id=doc_id,
        relative_path=relative_path,
        sha256=digest,
        bytes=len(payload),
        manifest_entries=len(entries) + 1,
    )


def _load_config(name: str) -> IndexConfig:
    from pipeline.indexing.config import load_index_config

    return load_index_config(resolve_config(name))


def _write_manifest_atomically(path: str | Path, entries: list[CorpusEntry]) -> None:
    """`write_manifest` ghi thẳng — đủ cho script offline, không đủ cho một
    endpoint mà worker ingest có thể đang đọc cùng file ở tiến trình khác."""
    target = Path(path)
    tmp = target.with_name(f"{target.name}.tmp")
    write_manifest(tmp, entries)
    os.replace(tmp, target)
