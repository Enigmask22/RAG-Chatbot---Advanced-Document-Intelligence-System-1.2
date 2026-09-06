"""Vòng eval đêm: chấm ứng viên → gate → đề nghị phát hành. `W5-10`.

    uv run python -m pipeline.eval.nightly --out-dir plans/reports/runs

## Ba câu hỏi mà một job đêm phải trả lời riêng rẽ

1. **Có ứng viên nào mới không?** — bundle semver cao hơn bản mà `CURRENT` đang
   trỏ tới. Không có thì dừng, exit 0, và nói ra là nó dừng vì không có việc chứ
   không phải vì mọi thứ đều tốt.
2. **Ứng viên có qua gate không?** — `pipeline.eval.gate`, không sao chép lại
   một luật nào. Module này **không** biết ngưỡng là bao nhiêu, và đó là điều
   kiện để nó không trở thành bản sao thứ hai của gate.
3. **Nếu qua thì đề nghị gì?** — một bundle phát hành đã ký cộng một con trỏ đã
   dời, gói trong một PR. Không tự merge.

## ⭐⭐ Champion mặc định là bản **đang phục vụ**, không phải bản kề dưới

`gate.py` chọn champion là bản semver cao nhất thấp hơn ứng viên. Đúng cho một
lần chấm thủ công. Sai cho một quyết định phát hành: câu hỏi ở đây là *"có tụt
so với thứ người dùng đang nhận không"*, và sau một lần rollback thì thứ người
dùng đang nhận **không** phải bản kề dưới ứng viên.

Hai hàm, hai câu hỏi, hai câu trả lời khác nhau — nên nightly truyền champion
tường minh thay vì mượn mặc định của gate.

## ⭐ Vì sao "không đề nghị gì" cũng phải sinh ra artifact

Một job đêm im lặng khi không có gì để làm là một job không phân biệt được với
một job đã chết. Mọi lượt chạy đều ghi `nightly-*.json` và một thân PR, kể cả
lượt kết luận "không có ứng viên": thứ đọc được ngày hôm sau là *tại sao* hôm
qua không có bản phát hành nào.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.bundle import RagBundle, list_bundles, read_pointer

from .gate import (
    DEFAULT_THRESHOLDS,
    GateStatus,
    GateVerdict,
    RuleOutcome,
    evaluate_gate,
    load_thresholds,
    render_html,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pipeline.bundle.promote import PromotionResult

__all__ = ["NightlyResult", "render_pr_body", "run_nightly"]

logger = logging.getLogger(__name__)

#: Exit code riêng cho "gate PASS nhưng không đúc được bản phát hành". Không gộp
#: vào 1: FAIL nghĩa là *hệ thống* chưa đủ tốt, còn cái này nghĩa là hệ thống đủ
#: tốt mà **đường phát hành** hỏng. Hai người khác nhau phải thức dậy.
EXIT_PROMOTION_REFUSED = 3

#: "Không có ứng viên" — exit 0, vì không có gì hỏng cả.
EXIT_NOTHING_TO_DO = 0


@dataclass
class NightlyResult:
    candidate: RagBundle | None
    champion: RagBundle | None
    verdict: GateVerdict | None
    promoted: PromotionResult | None
    reason: str
    exit_code: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran_at": datetime.now(UTC).isoformat(),
            "candidate": self.candidate.bundle_version if self.candidate else None,
            "champion": self.champion.bundle_version if self.champion else None,
            "verdict": self.verdict.as_dict() if self.verdict else None,
            "promoted": self.promoted.bundle.bundle_version if self.promoted else None,
            "pointer_moved_from": self.promoted.previous_pointer if self.promoted else None,
            "reason": self.reason,
            "exit_code": self.exit_code,
        }


def pick_candidate(root: Path, serving: str | None) -> RagBundle | None:
    """Bundle semver cao nhất. `None` nếu thư mục rỗng.

    Không lọc theo `serving` ở đây: "ứng viên trùng bản đang phục vụ" là một
    tình huống cần được **chấm và ghi lại**, không phải một tình huống cần được
    lọc mất. Xem `run_nightly`.
    """
    found = list_bundles(root, verify=True)
    if not found:
        return None
    logger.info("ứng viên = %s (đang phục vụ: %s)", found[-1].bundle_version, serving or "—")
    return found[-1]


def _champion_for(root: Path, candidate: RagBundle, serving: str | None) -> RagBundle | None:
    """Bản đang phục vụ, trừ khi chính nó là ứng viên — khi đó là bản kề dưới.

    Nhánh thứ hai không phải để lách: nó giữ cho lượt chạy "chưa có ứng viên
    mới" vẫn sinh ra một phán quyết đọc được, thay vì một bundle tự so với chính
    mình rồi báo mọi luật hồi quy đều xanh.
    """
    from .gate import _pick_champion

    if serving is not None and serving != candidate.bundle_version:
        from rag_core.bundle import bundle_dir_name, load_bundle

        return load_bundle(root / bundle_dir_name(serving))
    return _pick_champion(candidate, root)


def run_nightly(
    *,
    root: Path,
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    candidate_version: str | None = None,
    do_promote: bool = True,
) -> NightlyResult:
    serving = read_pointer(root)
    candidate: RagBundle | None
    if candidate_version is not None:
        from rag_core.bundle import bundle_dir_name, load_bundle

        candidate = load_bundle(root / bundle_dir_name(candidate_version))
    else:
        candidate = pick_candidate(root, serving)

    if candidate is None:
        return NightlyResult(None, None, None, None, f"không có bundle nào trong {root}", 0)

    champion = _champion_for(root, candidate, serving)
    verdict = evaluate_gate(candidate, champion, load_thresholds(thresholds_path))

    if verdict.status is not GateStatus.PASS:
        return NightlyResult(
            candidate,
            champion,
            verdict,
            None,
            f"gate {verdict.status.value} — không đề nghị phát hành",
            verdict.status.exit_code,
        )

    if serving is not None and candidate.bundle_version == serving:
        # ⭐ PASS mà vẫn không phát hành: ứng viên **chính là** bản đang chạy.
        # Đúc thêm một patch chỉ để dán dấu PASS lên nó là tăng số hiệu mà không
        # tăng thông tin, và mỗi lần chạy đêm lại tăng thêm một lần nữa.
        return NightlyResult(
            candidate,
            champion,
            verdict,
            None,
            f"gate PASS nhưng {candidate.bundle_version} đã là bản đang phục vụ — "
            "không có gì để phát hành",
            EXIT_NOTHING_TO_DO,
        )

    if not do_promote:
        return NightlyResult(
            candidate, champion, verdict, None, "gate PASS — bỏ qua promote theo yêu cầu", 0
        )

    from pipeline.bundle.promote import PromotionRefused, promote

    try:
        promoted = promote(candidate, verdict, root=root, report=f"gate-{candidate.bundle_version}")
    except PromotionRefused as exc:
        logger.error("gate PASS nhưng promote bị từ chối: %s", exc)
        return NightlyResult(candidate, champion, verdict, None, str(exc), EXIT_PROMOTION_REFUSED)

    return NightlyResult(candidate, champion, verdict, promoted, "", 0)


# ------------------------------------------------------------------ thân PR


_MARK = {RuleOutcome.PASS: "✅", RuleOutcome.FAIL: "❌", RuleOutcome.SKIP: "⏭"}


def _name(bundle: RagBundle | None) -> str:
    return bundle.bundle_version if bundle is not None else "—"


def render_pr_body(result: NightlyResult, *, thresholds_path: str = "") -> str:
    """Markdown cho thân PR / comment. Đọc được **mà không cần mở artifact nào**.

    Người đọc thân PR lúc 9 giờ sáng cần đúng ba thứ theo thứ tự này: đề nghị là
    gì, phán quyết ra sao, và luật nào trượt. Bảng đầy đủ để cuối — một thân PR
    mở ra là 25 dòng bảng thì phần kết luận bị đẩy xuống dưới nếp gấp.
    """
    lines: list[str] = []
    if result.promoted is not None:
        promoted = result.promoted
        lines += [
            f"# Đề nghị phát hành `{promoted.bundle.bundle_version}`",
            "",
            f"Ứng viên `{_name(result.candidate)}` qua gate so với "
            f"champion `{_name(result.champion)}`.",
            "",
            f"- Bundle mới: `{promoted.manifest.as_posix()}`",
            f"- Con trỏ `CURRENT`: `{promoted.previous_pointer or '(chưa có)'}` → "
            f"`{promoted.bundle.bundle_version}`",
            "- `git_sha` giữ nguyên của ứng viên — nó trỏ về mã đã sinh ra **số đo**, "
            "không phải commit của lần promote.",
            "",
            "**Merge PR này là lần phát hành.** Trước khi merge, kiểm hai thứ mà máy "
            "không kiểm được: số đo có được đo trên đúng corpus đang phục vụ không, "
            "và có thay đổi vận hành nào ngoài bundle cần đi cùng không.",
        ]
    else:
        lines += [
            "# Eval đêm — không đề nghị phát hành",
            "",
            f"**Lý do**: {result.reason}",
        ]

    if result.verdict is None:
        return "\n".join(lines) + "\n"

    verdict = result.verdict
    counts = verdict.counts()
    lines += [
        "",
        f"## Gate: **{verdict.status.value}** (exit {verdict.status.exit_code})",
        "",
        f"Ứng viên `{verdict.candidate}` · champion `{verdict.champion or '—'}` · "
        f"{counts['PASS']} PASS / {counts['FAIL']} FAIL / {counts['SKIP']} SKIP",
    ]
    if thresholds_path:
        lines.append(f"Ngưỡng: `{thresholds_path}`")

    failures = verdict.failures
    if failures:
        lines += ["", "### Luật trượt", ""]
        lines += [f"- ❌ **{rule.name}** ({rule.group}) — {rule.detail}" for rule in failures]
        if verdict.status is GateStatus.INCOMPARABLE:
            lines += [
                "",
                "> ⚠️ `INCOMPARABLE` bảo sửa **phép đo**, không phải sửa hệ thống. "
                "Hai lần đo chưa đặt cạnh nhau được thì mọi so sánh phía sau đều rỗng.",
            ]

    lines += [
        "",
        "<details><summary>Toàn bộ luật</summary>",
        "",
        "| | nhóm | luật | chi tiết |",
        "|---|---|---|---|",
    ]
    for rule in verdict.rules:
        detail = rule.detail.replace("|", "\\|")
        lines.append(f"| {_MARK[rule.outcome]} | {rule.group} | `{rule.name}` | {detail} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pipeline.eval.nightly",
        description="W5-10 — eval đêm: gate ứng viên rồi đề nghị phát hành nếu PASS",
    )
    parser.add_argument("--root", type=Path, default=Path("bundles"))
    parser.add_argument(
        "--candidate",
        default=None,
        help="version cụ thể; mặc định là bản semver cao nhất trong --root",
    )
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--out-dir", type=Path, default=Path("plans/reports/runs"))
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="chạy gate và sinh báo cáo, KHÔNG ghi bundle mới cũng không dời con trỏ",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = run_nightly(
        root=args.root,
        thresholds_path=args.thresholds,
        candidate_version=args.candidate,
        do_promote=not args.no_promote,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    summary = args.out_dir / f"nightly-{stamp}.json"
    summary.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    body = args.out_dir / "nightly-pr-body.md"
    body.write_text(
        render_pr_body(result, thresholds_path=args.thresholds.as_posix()), encoding="utf-8"
    )
    if result.verdict is not None:
        html = args.out_dir / f"gate-{result.verdict.candidate}.html"
        html.write_text(
            render_html(result.verdict, thresholds_path=args.thresholds.as_posix()),
            encoding="utf-8",
        )
        logger.info("đã ghi %s", html)
    logger.info("đã ghi %s và %s", summary, body)

    if result.promoted is not None:
        logger.info("ĐỀ NGHỊ PHÁT HÀNH %s", result.promoted.bundle.bundle_version)
    else:
        logger.info("không đề nghị phát hành: %s", result.reason)
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
