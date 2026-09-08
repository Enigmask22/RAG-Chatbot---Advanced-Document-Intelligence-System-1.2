"""Cửa nhận tài liệu của Pipeline Plane — `NEW-11` (chốt 08/09/2026: cho phép).

Ba thứ được kiểm ở đây, theo đúng ba câu hỏi của dòng nợ:

1. **Cửa license** — upload đi qua đúng cái cửa mà mọi tài liệu corpus đã đi:
   `LICENSE_ALLOWLIST`, `source_url` bắt buộc, và hai plane + trang UI phải
   thống nhất về danh sách ấy bằng **quan hệ** chứ không phải ba bản chép tay
   (họ `NEW-13`).
2. **Cơ học ghi** — file trước, manifest sau, manifest ghi nguyên tử, thất bại
   giữa chừng không để lại entry trỏ vào hư không.
3. **Trần byte** — trần đếm byte UTF-8 chứ không phải ký tự, vì tiếng Việt có
   dấu là 2–3 byte mỗi ký tự và `BodyLimitMiddleware` đếm byte.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pipeline.corpus.manifest import CorpusEntry, load_manifest, write_manifest
from pipeline.ingest import upload as upload_module
from pipeline.ingest.upload import (
    MAX_UPLOAD_BYTES,
    DuplicateUpload,
    UploadRequest,
    receive_upload,
)
from rag_core.schemas import LICENSE_ALLOWLIST, DocType, Language

_BODY = "Đầu tư công cho thuỷ lợi tăng đều qua các năm. " * 40


def _request(**overrides: object) -> UploadRequest:
    fields: dict[str, object] = {
        "config": "demo",
        "title": "Báo cáo thuỷ lợi 2026",
        "content": _BODY,
        "license": "CC BY 4.0",
        "source_url": "https://example.org/thuy-loi",
        "uploaded_by": "acme:key-1",
    }
    fields.update(overrides)
    return UploadRequest.model_validate(fields)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Corpus 1 tài liệu + config `demo`, cô lập trong thư mục tạm.

    `INGEST_CONFIG_DIR` trỏ vào đây thay vì nới lỏng `resolve_config` — cùng
    lý lẽ với fixture của `test_ingest_job.py`.
    """
    from rag_core.settings import get_settings

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = ("Tài liệu có sẵn. " * 30).encode()
    (corpus / "d-0.txt").write_bytes(payload)
    write_manifest(
        tmp_path / "manifest.csv",
        [
            CorpusEntry(
                doc_id="d-0",
                relative_path="d-0.txt",
                source_url="https://example.org/d-0",
                license="CC BY 4.0",
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
                source="test",
                lang=Language.VI,
                doc_type=DocType.DEV_REPORT,
            )
        ],
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name, manifest in (("demo", "manifest.csv"), ("mo-coi", "khong-ton-tai.csv")):
        (config_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "tenant_id": "test",
                    "manifest_path": str(tmp_path / manifest),
                    "corpus_dir": str(corpus),
                    "embedding_model": "hashing:64",
                    "use_cache": False,
                    "state_dir": str(tmp_path / "state"),
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setenv("INGEST_CONFIG_DIR", str(config_dir))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 1. Cửa license — một danh sách, ba người dùng, ghim bằng quan hệ
# ---------------------------------------------------------------------------


class TestTheLicenseGate:
    def test_a_noderivatives_license_is_refused_with_the_reason(self) -> None:
        """`ND` cấm tác phẩm phái sinh, mà chunking + sinh context LÀ phái
        sinh — cùng lý lẽ đã viết ở cửa manifest từ `W0`."""
        with pytest.raises(ValidationError, match="danh sách cho phép"):
            _request(license="CC BY-ND 4.0")

    @pytest.mark.parametrize("name", sorted(LICENSE_ALLOWLIST))
    def test_every_allowed_license_passes(self, name: str) -> None:
        assert _request(license=name).license == name

    def test_the_proxy_agrees_with_the_service(self) -> None:
        """Hai plane không import được nhau nên schema tồn tại hai lần; đây là
        phép kiểm **quan hệ** giữ chúng không trôi (họ `NEW-13`)."""
        from serving.api import ingest as proxy

        assert proxy.MAX_UPLOAD_BYTES == MAX_UPLOAD_BYTES
        with pytest.raises(ValidationError, match="danh sách cho phép"):
            proxy.UploadRequest.model_validate(
                {
                    "config": "demo",
                    "title": "Báo cáo thuỷ lợi 2026",
                    "content": _BODY,
                    "license": "CC BY-ND 4.0",
                    "source_url": "https://example.org/thuy-loi",
                }
            )

    def test_the_ui_offers_exactly_the_allowed_licenses(self) -> None:
        """Trang chỉ là gợi ý (máy chủ mới cưỡng chế), nhưng một gợi ý lệch
        danh sách là một form luôn bị 422 — hoặc một giấy phép hợp lệ không
        chọn được."""
        page = Path("serving/ui/index.html").read_text(encoding="utf-8")
        block = re.search(r"const LICENSES = \[(.*?)\];", page, re.DOTALL)
        assert block is not None, "trang không còn mảng LICENSES"
        offered = set(re.findall(r'"([^"]+)"', block.group(1)))
        assert offered == set(LICENSE_ALLOWLIST)

    def test_the_page_posts_to_the_admin_proxy_not_to_the_service(self) -> None:
        """Trình duyệt không được gọi thẳng cổng 8001 (`AU-10`) — đường duy
        nhất là proxy dưới `/admin`, nơi tầng auth của `W4-04` áp theo tiền tố."""
        page = Path("serving/ui/index.html").read_text(encoding="utf-8")
        assert "/admin/ingest/upload" in page
        assert ":8001" not in page


# ---------------------------------------------------------------------------
# 2. Trần byte
# ---------------------------------------------------------------------------


class TestTheByteBudget:
    def test_vietnamese_text_is_measured_in_bytes_not_characters(self) -> None:
        """200.000 ký tự "ạ" là 600.000 byte — dưới trần ký tự, trên trần byte.
        Đo bằng ký tự thì upload này chết ngang ở `BodyLimitMiddleware` với một
        thông điệp về thân request, không phải về tài liệu."""
        with pytest.raises(ValidationError, match="byte"):
            _request(content="ạ" * 200_000)

    def test_ascii_under_the_cap_passes(self) -> None:
        assert _request(content="a" * 400_000).content


# ---------------------------------------------------------------------------
# 3. Cơ học ghi
# ---------------------------------------------------------------------------


class TestReceiveUpload:
    def test_the_receipt_matches_the_disk_and_the_manifest(self, workspace: Path) -> None:
        receipt = receive_upload(_request())

        # `slugify` bỏ ký tự ngoài ASCII — tiêu đề có dấu vẫn ra tên file an toàn.
        assert receipt.doc_id.startswith("up-b-o-c-o-thu-l-i-2026-")
        stored = workspace / "corpus" / receipt.relative_path
        assert stored.read_text(encoding="utf-8") == _BODY
        assert receipt.relative_path.startswith("uploads/")
        entries = load_manifest(workspace / "manifest.csv")
        assert receipt.manifest_entries == len(entries) == 2
        entry = entries[-1]
        assert entry.doc_id == receipt.doc_id
        assert entry.source == "upload"
        assert "acme:key-1" in entry.notes
        assert entry.sha256 == receipt.sha256 == hashlib.sha256(_BODY.encode()).hexdigest()
        # `TD-22`: với `.txt`, phép parse là hàm đồng nhất — hai cột phải trùng.
        assert entry.text_sha256 == entry.sha256

    def test_duplicate_content_is_a_conflict_naming_the_original(self, workspace: Path) -> None:
        receive_upload(_request())
        with pytest.raises(DuplicateUpload) as exc:
            receive_upload(_request(title="Tên khác, ruột y hệt"))
        assert "up-" in str(exc.value)

    def test_an_empty_manifest_is_refused_not_created(self, workspace: Path) -> None:
        """Một config gõ nhầm tên phải chết ở đây, không sinh một manifest song
        song mà không đường đọc nào biết tới."""
        with pytest.raises(ValueError, match="corpus đã đăng ký"):
            receive_upload(_request(config="mo-coi"))

    def test_a_missing_config_is_a_client_error(self, workspace: Path) -> None:
        with pytest.raises(FileNotFoundError):
            receive_upload(_request(config="khong-co"))

    def test_a_failed_manifest_write_leaves_no_orphan_file(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File ghi trước, manifest ghi sau — nếu bước hai hỏng thì bước một
        phải được dọn, không thì `uploads/` tích rác qua từng lần lỗi."""

        def no(_path: object, _entries: object) -> None:
            raise OSError("đĩa đầy")

        monkeypatch.setattr(upload_module, "write_manifest", no)
        with pytest.raises(OSError):
            receive_upload(_request())
        assert not list((workspace / "corpus" / "uploads").glob("*")), (
            "file mồ côi sau một manifest hỏng"
        )
        assert len(load_manifest(workspace / "manifest.csv")) == 1

    def test_the_manifest_is_replaced_atomically(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worker ingest có thể đang đọc manifest ở tiến trình khác — nó phải
        thấy bản cũ hoặc bản mới, không bao giờ thấy nửa file."""
        import os

        calls: list[tuple[str, str]] = []
        real_replace = os.replace

        def spy(src: object, dst: object, **kwargs: object) -> None:
            calls.append((str(src), str(dst)))
            real_replace(src, dst)  # type: ignore[arg-type]

        # `upload.py` gọi `os.replace` qua module `os` toàn cục — vá ở đó.
        monkeypatch.setattr(os, "replace", spy)
        receive_upload(_request())
        manifest = str(workspace / "manifest.csv")
        assert any(dst == manifest and src.endswith(".tmp") for src, dst in calls), (
            "manifest phải được ghi qua file tạm + os.replace"
        )
