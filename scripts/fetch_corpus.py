#!/usr/bin/env python
"""Tải corpus công khai về `data/corpus/` và ghi manifest có nguồn + giấy phép.

    uv run python scripts/fetch_corpus.py --config configs/corpus/worldbank_vietnam.yaml
    uv run python scripts/fetch_corpus.py --config ... --dry-run   # xem trước, không tải

Có thể chạy lại nhiều lần: tài liệu đã có trong manifest và còn nguyên trên đĩa
sẽ được bỏ qua. Bị ngắt giữa chừng thì chạy lại là tiếp tục.

Hai nguồn được hỗ trợ:

* `worldbank_wds` — gọi API tìm kiếm của World Bank, tự động.
* `seed_list` — danh sách URL ghi tay trong config. Dùng cho ADB (adb.org trả 403
  cho truy cập tự động) và cho báo cáo thường niên HOSE.

Script **không** đọc `.env` và không cần API key nào.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from pipeline.corpus.manifest import (
    CorpusEntry,
    load_manifest,
    slugify,
    write_manifest,
)
from pipeline.corpus.worldbank import USER_AGENT, WdsDocument, search_wds
from rag_core.schemas import DocType, Language

logger = logging.getLogger("fetch_corpus")

# Trang lỗi/đăng nhập thường trả 200 kèm HTML. Nhận nhầm nó thành tài liệu sẽ đưa
# rác vào corpus và chỉ lộ ra rất muộn — lúc đọc kết quả eval thấy vô lý.
_HTML_SNIFF = re.compile(rb"<!doctype html|<html[\s>]", re.IGNORECASE)
_MIN_BYTES = 2_000


@dataclass
class FetchStats:
    downloaded: int = 0
    skipped_existing: int = 0
    rejected: int = 0
    failed: int = 0


def download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data: bytes = response.read()
    return data


def looks_like_document(payload: bytes, expect_text: bool) -> str | None:
    """Trả lý do từ chối, hoặc `None` nếu nội dung dùng được."""
    if len(payload) < _MIN_BYTES:
        return f"quá ngắn ({len(payload)} byte) — nhiều khả năng là trang lỗi"
    if expect_text and _HTML_SNIFF.search(payload[:2000]):
        return "trả về HTML thay vì text"
    if not expect_text and not payload.startswith(b"%PDF"):
        return "không phải PDF hợp lệ"
    if expect_text:
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return "không giải mã được UTF-8"
        if len(text.split()) < 200:
            return f"chỉ có {len(text.split())} từ — quá ít để chunk"
    return None


def entry_from_wds(
    doc: WdsDocument,
    *,
    payload: bytes,
    relative_path: str,
    license_name: str,
    license_url: str,
    doc_type: DocType,
    source_name: str,
) -> CorpusEntry:
    return CorpusEntry(
        doc_id=f"wb-{doc.guid}",
        relative_path=relative_path,
        source_url=doc.download_url,
        landing_url=doc.landing_url,
        license=license_name,
        license_url=license_url,
        title=doc.title,
        lang=doc.language,
        doc_type=doc_type,
        source=source_name,
        published_at=doc.published_at[:10],
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        fetched_at=CorpusEntry.now_iso(),
        notes=f"docty={doc.doc_type}; majdocty={doc.major_doc_type}",
    )


def _matches(doc: WdsDocument, block: dict[str, Any]) -> str | None:
    """Trả lý do loại bỏ, hoặc `None` nếu tài liệu hợp lệ."""
    if not doc.download_url:
        return "không có link tải"
    if not doc.is_public:
        return f"trạng thái công bố = {doc.disclosure_status!r}"
    if block.get("require_text", True) and not doc.text_url:
        return "chỉ có PDF (cần Docling ở W3-01)"
    allowed_major = block.get("major_doc_types")
    if allowed_major and doc.major_doc_type not in allowed_major:
        return f"majdocty={doc.major_doc_type!r} ngoài danh sách"
    excluded = block.get("exclude_doc_types") or []
    if doc.doc_type in excluded:
        return f"docty={doc.doc_type!r} bị loại"
    return None


def fetch_worldbank(
    block: dict[str, Any],
    *,
    out_dir: Path,
    known_ids: set[str],
    known_hashes: set[str],
    delay: float,
    timeout: float,
    dry_run: bool,
    stats: FetchStats,
    already_have: int = 0,
) -> list[CorpusEntry]:
    # `limit` là **tổng mục tiêu của nguồn này**, không phải "thêm ngần ấy mỗi lần
    # chạy". Trừ đi số đã có thì chạy lại lần hai là no-op — nếu không, mỗi lần
    # chạy lại corpus sẽ phình thêm mà manifest vẫn hợp lệ nên không ai để ý.
    limit = max(0, int(block.get("limit", 20)) - already_have)
    if limit == 0:
        logger.info("Đã đủ %d tài liệu cho nguồn này, không tải thêm", already_have)
        return []
    source_name = str(block.get("name", "worldbank_wds"))
    license_name = str(block["license"])
    license_url = str(block.get("license_url", ""))
    doc_type = DocType(block.get("doc_type", DocType.DEV_REPORT.value))
    lang_filter = block.get("language")

    entries: list[CorpusEntry] = []
    for doc in search_wds(
        str(block["qterm"]),
        language=lang_filter,
        rows_per_page=int(block.get("rows_per_page", 20)),
        max_pages=int(block.get("max_pages", 10)),
        timeout=timeout,
        extra_params=block.get("extra_params"),
    ):
        if len(entries) >= limit:
            break

        doc_id = f"wb-{doc.guid}"
        if doc_id in known_ids:
            stats.skipped_existing += 1
            continue

        reason = _matches(doc, block)
        if reason is not None:
            logger.debug("Bỏ %s: %s", doc.guid, reason)
            continue

        if dry_run:
            logger.info("[dry-run] %s | %s | %s", doc.language.value, doc.guid, doc.title[:80])
            entries.append(
                entry_from_wds(
                    doc,
                    payload=b"dry-run-placeholder" * 200,
                    relative_path=f"{slugify(doc.title)}-{doc.guid}{doc.extension}",
                    license_name=license_name,
                    license_url=license_url,
                    doc_type=doc_type,
                    source_name=source_name,
                )
            )
            continue

        time.sleep(delay)
        try:
            payload = download(doc.download_url, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Tải hỏng %s: %s", doc.guid, exc)
            stats.failed += 1
            continue

        rejection = looks_like_document(payload, expect_text=bool(doc.text_url))
        if rejection is not None:
            logger.info("Loại %s: %s", doc.guid, rejection)
            stats.rejected += 1
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in known_hashes:
            logger.info("Loại %s: nội dung trùng tài liệu đã có", doc.guid)
            stats.rejected += 1
            continue

        filename = f"{slugify(doc.title)}-{doc.guid}{doc.extension}"
        (out_dir / filename).write_bytes(payload)
        known_hashes.add(digest)
        known_ids.add(doc_id)
        stats.downloaded += 1
        logger.info("✓ %-3s %7.1f KB  %s", doc.language.value, len(payload) / 1024, doc.title[:70])
        entries.append(
            entry_from_wds(
                doc,
                payload=payload,
                relative_path=filename,
                license_name=license_name,
                license_url=license_url,
                doc_type=doc_type,
                source_name=source_name,
            )
        )

    return entries


def fetch_seed_list(
    block: dict[str, Any],
    *,
    out_dir: Path,
    known_ids: set[str],
    known_hashes: set[str],
    delay: float,
    timeout: float,
    dry_run: bool,
    stats: FetchStats,
    already_have: int = 0,
) -> list[CorpusEntry]:
    """Tải theo danh sách URL ghi tay — dùng cho nguồn không có API mở."""
    del already_have  # danh sách ghi tay vốn đã hữu hạn, không cần chặn thêm
    entries: list[CorpusEntry] = []
    for item in block.get("documents", []):
        doc_id = str(item["doc_id"])
        if doc_id in known_ids:
            stats.skipped_existing += 1
            continue
        url = str(item["url"])
        expect_text = not url.lower().endswith(".pdf")

        if dry_run:
            logger.info("[dry-run] %s | %s", doc_id, url)
            continue

        time.sleep(delay)
        try:
            payload = download(url, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Tải hỏng %s: %s", doc_id, exc)
            stats.failed += 1
            continue

        rejection = looks_like_document(payload, expect_text=expect_text)
        if rejection is not None:
            logger.info("Loại %s: %s", doc_id, rejection)
            stats.rejected += 1
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in known_hashes:
            stats.rejected += 1
            continue

        filename = f"{slugify(doc_id)}{'.txt' if expect_text else '.pdf'}"
        (out_dir / filename).write_bytes(payload)
        known_hashes.add(digest)
        known_ids.add(doc_id)
        stats.downloaded += 1
        entries.append(
            CorpusEntry(
                doc_id=doc_id,
                relative_path=filename,
                source_url=url,
                landing_url=str(item.get("landing_url", "")),
                license=str(item.get("license", block["license"])),
                license_url=str(item.get("license_url", block.get("license_url", ""))),
                title=str(item.get("title", doc_id)),
                lang=Language(item.get("lang", "unknown")),
                doc_type=DocType(item.get("doc_type", block.get("doc_type", "other"))),
                source=str(block.get("name", "seed_list")),
                published_at=str(item.get("published_at", "")),
                sha256=digest,
                bytes=len(payload),
                fetched_at=CorpusEntry.now_iso(),
                notes=str(item.get("notes", "")),
            )
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.csv"))
    parser.add_argument("--delay", type=float, default=1.0, help="giây nghỉ giữa 2 lần tải")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, help="ghi đè `limit` của mọi nguồn")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    existing = load_manifest(args.manifest)
    known_ids = {entry.doc_id for entry in existing}
    known_hashes = {entry.sha256 for entry in existing}
    have_by_source = Counter(entry.source for entry in existing)
    logger.info("Manifest hiện có %d tài liệu", len(existing))

    stats = FetchStats()
    new_entries: list[CorpusEntry] = []
    handlers = {"worldbank_wds": fetch_worldbank, "seed_list": fetch_seed_list}

    for block in config["sources"]:
        kind = block["type"]
        handler = handlers.get(kind)
        if handler is None:
            raise SystemExit(f"Nguồn không hỗ trợ: {kind!r}. Chỉ có {sorted(handlers)}")
        if args.limit is not None:
            block = {**block, "limit": args.limit}
        name = str(block.get("name", kind))
        logger.info("── Nguồn %s (%s)", name, kind)
        # `source` của mỗi entry là **tên block**, nên kế toán chính xác theo từng
        # nguồn: nâng `limit` của một block về sau vẫn tải thêm đúng phần còn thiếu.
        already = have_by_source.get(name, 0)
        new_entries.extend(
            handler(
                block,
                out_dir=args.out,
                known_ids=known_ids,
                known_hashes=known_hashes,
                delay=args.delay,
                timeout=args.timeout,
                dry_run=args.dry_run,
                stats=stats,
                already_have=already,
            )
        )

    if args.dry_run:
        logger.info("[dry-run] sẽ tải %d tài liệu, không ghi gì", len(new_entries))
        return 0

    total = write_manifest(args.manifest, [*existing, *new_entries])
    logger.info(
        "Xong: +%d mới · %d bỏ qua (đã có) · %d bị loại · %d lỗi tải · manifest %d dòng",
        stats.downloaded,
        stats.skipped_existing,
        stats.rejected,
        stats.failed,
        total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
