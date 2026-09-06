"""Đúc bản phát hành từ một ứng viên đã qua gate — `W5-10`, trả `TD-71`.

## ⭐⭐ Vì sao promote phải **đúc bản mới** chứ không ghi vào bundle cũ

`RagBundle.gate` có mặt từ `W4-01` và tới hết `W5-09` vẫn luôn là `NOT_RUN`:
phán quyết sống trong `gate-*.json`/`.html` cạnh bundle, không nằm **trong** nó.
Hệ quả đúng như `TD-71` ghi — một bundle không tự khai được nó đã qua gate nào,
nên không có gì để một đường promote tự động kiểm trước khi đổi con trỏ.

Cách hiển nhiên để sửa là mở manifest ra, điền `gate`, ký lại. Nó sai:
`save_bundle` có đúng một luật số 1 — *không ghi đè một version đã tồn tại* —
và luật ấy tồn tại vì "số đo này thuộc về `v0.2.1`" chỉ có nghĩa khi `v0.2.1`
không đổi được. Một manifest sửa được sau khi ký thì chữ ký chỉ còn là trang trí.

Nên promote **đúc một bản patch mới**: cùng components, cùng eval, khác đúng ba
thứ — số hiệu, phán quyết gate, và một dòng `notes` nói nó sinh ra từ đâu.

## ⭐⭐ `git_sha` **không** được làm mới, và đó là điểm dễ sai nhất

Phản xạ là ghi commit của lần chạy promote. Nhưng docstring của trường ấy nói rõ
nó là *"đường duy nhất đi ngược từ một artifact đang chạy về mã đã tạo ra nó"* —
và mã đã tạo ra **những con số** trong bundle này là commit của ứng viên, không
phải commit của cái đêm mà một job CI đổi con trỏ. Ghi đè nó là đổi một trường
truy nguyên lấy một trường nhật ký, im lặng.

Commit của lần promote vẫn được ghi lại — ở `notes`, nơi nó là nhật ký thật.

## Ranh giới với "phát hành"

Promote ghi **hai** thứ vào working tree: manifest mới và con trỏ `CURRENT`. Nó
không merge gì cả. Trong luồng `W5-10`, cả hai đi vào một PR, và **merge PR** mới
là lần phát hành. Tức phán quyết máy đưa ra là một đề nghị có bằng chứng, còn
chữ ký cuối vẫn là của người — đúng chỗ nên đặt nó.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.bundle import (
    BundleValidationError,
    GateRecord,
    GateStatus,
    RagBundle,
    bundle_dir_name,
    list_bundles,
    parse_semver,
    save_bundle,
    write_pointer,
)

if TYPE_CHECKING:
    from pipeline.eval.gate import GateVerdict

__all__ = [
    "PromotionRefused",
    "PromotionResult",
    "next_patch_version",
    "promote",
    "rollback",
]

logger = logging.getLogger(__name__)

#: Những trường mà promote **sao chép nguyên văn**. Danh sách này là hợp đồng:
#: một bản phát hành mang đúng những số đo của ứng viên, không hơn không kém.
#: `test_promote_copies_the_measurements_verbatim` đọc chính nó.
CARRIED_OVER = ("components", "eval", "git_sha")


class PromotionRefused(RuntimeError):
    """Không đúc bản phát hành. Luôn kèm lý do đọc được, vì nó chạy trong CI."""


@dataclass(frozen=True)
class PromotionResult:
    bundle: RagBundle
    manifest: Path
    pointer: Path
    previous_pointer: str | None


def next_patch_version(root: Path, candidate: RagBundle) -> str:
    """Số patch trống tiếp theo trên cùng `major.minor` với ứng viên.

    Quét thay vì `patch + 1` vì lần promote thứ hai trong cùng một đêm — hoặc
    một `0.2.2` đúc tay từ trước — làm `patch + 1` đâm vào luật không-ghi-đè và
    job đêm tự kẹt. Quét thì nó đi tiếp, và số nó chọn nằm trong log.
    """
    major, minor, patch, _ = parse_semver(candidate.bundle_version)
    taken = {b.bundle_version for b in list_bundles(root, verify=False)}
    nxt = patch + 1
    while f"{major}.{minor}.{nxt}" in taken:
        nxt += 1
    return f"{major}.{minor}.{nxt}"


def _promote_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:  # pragma: no cover - máy không có git
        return "unknown"
    return out.stdout.strip()[:12]


def promote(
    candidate: RagBundle,
    verdict: GateVerdict,
    *,
    root: Path,
    version: str | None = None,
    report: str | None = None,
    move_pointer: bool = True,
) -> PromotionResult:
    """Ứng viên + phán quyết PASS → bundle phát hành đã ký, con trỏ đã dời.

    Ba phép từ chối, và cả ba đều là những cách một đường promote tự động lặng
    lẽ phát hành nhầm thứ:

    * **Phán quyết không PASS.** Hiển nhiên, nhưng phải nằm ở đây chứ không chỉ
      ở nhánh `if` của người gọi: hàm này là thứ ghi vào đĩa.
    * **Phán quyết nói về một bundle khác.** `evaluate_gate(a, ...)` rồi
      `promote(b, verdict)` là cách dán một dấu PASS lên một artifact chưa ai
      chấm — và nó trông hoàn toàn bình thường trong log.
    * **Phán quyết PASS mà không có champion.** `--no-champion` là chế độ chẩn
      đoán: nó bỏ hẳn nhóm luật hồi quy. Một bản phát hành đúc từ đó là một bản
      chưa ai hỏi "có tụt so với bản đang chạy không". `GateRecord` cũng đã tự
      đòi điều này ở tầng schema; hàm này đòi sớm hơn, kèm lý do dài hơn.
    """
    from pipeline.eval.gate import GateStatus as VerdictStatus

    if verdict.status is not VerdictStatus.PASS:
        raise PromotionRefused(
            f"gate cho {verdict.candidate} là {verdict.status.value}, không phải PASS — "
            "không đúc bản phát hành."
        )
    if verdict.candidate != candidate.bundle_version:
        raise PromotionRefused(
            f"phán quyết nói về bundle {verdict.candidate!r} nhưng đang promote "
            f"{candidate.bundle_version!r}. Một dấu PASS chỉ thuộc về đúng cái "
            "artifact đã được chấm."
        )
    if verdict.champion is None:
        raise PromotionRefused(
            f"gate PASS cho {verdict.candidate} nhưng không có champion để so. "
            "`--no-champion` bỏ hẳn nhóm luật hồi quy, nên nó chẩn đoán được mà "
            "phát hành thì không: không ai đã hỏi bản này có tụt so với bản đang chạy."
        )

    target = version or next_patch_version(root, candidate)
    record = GateRecord(status=GateStatus.PASS, champion_compared=verdict.champion, report=report)
    # ⭐⭐ Dựng lại qua `model_validate`, KHÔNG qua `model_copy`. `model_copy` bỏ
    # qua toàn bộ validator — đúng cái bẫy mà `make_bundle_retrieval_only` trong
    # `tests/unit/test_bundle.py` đã ghi lại một lần. Ở đây hậu quả nặng hơn một
    # fixture sai: đường này **ghi ra đĩa và ký**, nên một bundle mà schema sẽ
    # từ chối vẫn ra được một manifest hợp lệ về chữ ký. Bundle phát hành phải
    # đi qua đúng cánh cửa mà mọi bundle khác đi qua.
    payload = candidate.model_dump(mode="json")
    payload.update(
        {
            "bundle_version": target,
            "created_at": datetime.now(UTC).isoformat(),
            "gate": record.model_dump(mode="json"),
            "notes": _notes(candidate, verdict),
            "checksum": None,  # ký lại ở `save_bundle`; giữ chữ ký cũ là ký cho nội dung cũ
        }
    )
    promoted = RagBundle.model_validate(payload)
    manifest = save_bundle(promoted, root)
    logger.info("đã đúc bản phát hành %s từ ứng viên %s", target, candidate.bundle_version)

    previous = None
    pointer = root / "CURRENT"
    if move_pointer:
        from rag_core.bundle import read_pointer

        previous = read_pointer(root)
        pointer = write_pointer(root, target)
        logger.info("con trỏ CURRENT: %s → %s", previous or "(chưa có)", target)

    return PromotionResult(
        bundle=promoted, manifest=manifest, pointer=pointer, previous_pointer=previous
    )


def _notes(candidate: RagBundle, verdict: GateVerdict) -> str:
    counts = verdict.counts()
    lines = [
        f"Phát hành tự động (`W5-10`) từ ứng viên {candidate.bundle_version}, "
        f"gate PASS so với champion {verdict.champion}.",
        f"Luật: {counts.get('PASS', 0)} PASS · {counts.get('SKIP', 0)} SKIP.",
        f"Commit lúc promote: {_promote_sha()} "
        f"(`git_sha` giữ nguyên của ứng viên — nó trỏ về mã đã sinh ra SỐ ĐO).",
    ]
    if candidate.notes:
        lines.append(f"Ghi chú của ứng viên: {candidate.notes}")
    return "\n".join(lines)


def rollback(root: Path, version: str) -> Path:
    """Dời con trỏ về một bundle cũ. Không xoá gì, không đúc gì.

    Có mặt ở đây thay vì ở một script rời vì nó là **cặp** của `promote`: một
    đường tiến không có đường lùi cùng chỗ, cùng phép kiểm, là một đường tiến
    mà người ta ngại đi.
    """
    if not (root / bundle_dir_name(version) / "manifest.json").is_file():
        raise BundleValidationError(f"không có bundle {version} trong {root} để lùi về")
    return write_pointer(root, version)
