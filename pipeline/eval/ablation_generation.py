"""So hai **model sinh** trên cùng một retrieval stack, có kiểm định. `W5-11`.

    uv run python -m pipeline.eval.ablation_generation \\
        --run plans/reports/runs/w511-deepseek-generation-per-query.json \\
        --run plans/reports/runs/w511-glm-generation-per-query.json \\
        --baseline w511-deepseek --out plans/reports/runs/w511-ablation.md

## ⭐⭐ Vì sao không đọc thẳng hai file `*-generation.json` rồi trừ hai con số

Đó đúng là thứ `W2-09` đã cấm, và lý do nó cấm vẫn nguyên: `hit_rate@5` "tụt
6,7%" hoá ra là **45 câu xuống 42 câu** — chênh ba câu. Một bảng ablation xếp
hạng bằng những mức chênh cỡ ấy đang xếp hạng nhiễu.

Nên module này chỉ nhận **điểm cấp truy vấn** (`*-per-query.json`), ghép cặp
theo `query_id`, rồi lấy khoảng tin cậy của **hiệu** bằng bootstrap cặp — cùng
hàm `paired_bootstrap` mà mọi con số đã công bố từ `W2-01` đi qua, không phải
một bản sao của nó.

## ⭐ Hai hàng rào mượn nguyên từ `compare.py`

1. **Chỉ so trên tập truy vấn giao nhau**, và nói ra nếu hai lần chạy không cùng
   tập. So 242 câu với 200 câu rồi kết luận là một dạng tự chọn mẫu.
2. **Từ chối so khi judge khác nhau.** `TD-66` đo được: đổi model judge làm
   faithfulness dịch **7,5 điểm** — lớn hơn mọi cải thiện của cả `W2` cộng lại.
   Hai nhánh chấm bởi hai judge khác nhau thì hiệu số giữa chúng mang lẫn cả
   phần dịch ấy, và không có cách nào tách ra sau khi đã trộn.

## ⚠️ Micro-average và macro-average là hai con số khác nhau

`generation_metrics` báo **micro** (mọi mệnh đề nặng như nhau). Bootstrap ở đây
chạy trên **macro** (mọi truy vấn nặng như nhau) vì đơn vị lấy mẫu lại độc lập
là truy vấn, không phải mệnh đề. Hai con số ấy không bằng nhau và bảng in cả
hai, đúng tên: cột `micro` để đọc mức, cột `macro ± CI` để đọc **hiệu**.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .compare import DEFAULT_ALPHA, DEFAULT_BOOTSTRAP, DEFAULT_SEED, paired_bootstrap

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Branch", "MetricRow", "compare_branches", "format_table", "load_branch"]

logger = logging.getLogger(__name__)

#: Metric mà **cao hơn là tốt hơn**. Những cái còn lại (`misattribution`,
#: `false_refusal_rate`) ngược, và một bảng không phân biệt được hai chiều ấy sẽ
#: in mũi tên xanh cho một lần xấu đi.
HIGHER_IS_BETTER = {
    "answer_relevancy",
    "citation_coverage",
    "citation_validity",
    "context_precision@5",
    "context_recall@5",
    "faithfulness",
    "refusal_accuracy",
    "refusal_recall",
    "uncited_grounding",
}


@dataclass(frozen=True)
class Branch:
    name: str
    models: tuple[str, ...]
    judge_model: str
    scores: dict[str, dict[str, float]]
    """`metric -> {query_id: điểm}`."""

    def query_ids(self, metric: str) -> set[str]:
        return set(self.scores.get(metric, {}))


@dataclass(frozen=True)
class MetricRow:
    metric: str
    n: int
    baseline_macro: float
    candidate_macro: float
    diff: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        """Khoảng tin cậy của hiệu **không chứa 0**.

        Không đọc thành "hai hệ thống như nhau" khi nó chứa 0: nó nghĩa là *tập
        này không phân biệt được hai hệ thống*, và với n vài trăm câu thì đó là
        phát biểu về cỡ mẫu ít nhất ngang bằng phát biểu về hệ thống.
        """
        return self.low > 0 or self.high < 0

    @property
    def better(self) -> str:
        if not self.significant:
            return "—"
        gain = self.diff > 0
        if self.metric not in HIGHER_IS_BETTER:
            gain = not gain
        return "ứng viên" if gain else "mốc"

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n": self.n,
            "baseline_macro": round(self.baseline_macro, 4),
            "candidate_macro": round(self.candidate_macro, 4),
            "diff": round(self.diff, 4),
            "ci95": [round(self.low, 4), round(self.high, 4)],
            "significant": self.significant,
            "better": self.better,
        }


def load_branch(path: Path) -> Branch:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = {"run", "models", "judge_model"}
    return Branch(
        name=str(raw["run"]),
        models=tuple(raw.get("models", ())),
        judge_model=str(raw.get("judge_model", "")),
        scores={k: v for k, v in raw.items() if k not in meta and isinstance(v, dict)},
    )


def compare_branches(
    baseline: Branch,
    candidate: Branch,
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[list[MetricRow], list[str]]:
    """Trả về `(hàng, cảnh báo)`. Cảnh báo **không** làm hàm ném.

    Một lần chạy lệch 3 câu vẫn so được, miễn là con số 3 ấy được in ra cạnh
    bảng. Thứ phải ném là hai judge khác nhau — xem docstring module.
    """
    if baseline.judge_model != candidate.judge_model:
        raise ValueError(
            f"hai nhánh được chấm bởi hai judge khác nhau ({baseline.judge_model!r} vs "
            f"{candidate.judge_model!r}). `TD-66` đo được đổi judge làm faithfulness dịch "
            "7,5 điểm — hiệu số giữa hai nhánh sẽ mang lẫn phần dịch ấy và không tách ra được."
        )

    warnings: list[str] = []
    rows: list[MetricRow] = []
    for metric in sorted(set(baseline.scores) & set(candidate.scores)):
        left, right = baseline.query_ids(metric), candidate.query_ids(metric)
        shared = sorted(left & right)
        if not shared:
            warnings.append(f"`{metric}`: không có truy vấn chung, bỏ qua")
            continue
        if left != right:
            warnings.append(
                f"`{metric}`: mốc {len(left)} câu, ứng viên {len(right)} câu, "
                f"so trên {len(shared)} câu giao nhau"
            )
        a = [baseline.scores[metric][q] for q in shared]
        b = [candidate.scores[metric][q] for q in shared]
        diffs = [y - x for x, y in zip(a, b, strict=True)]
        low, high = paired_bootstrap(diffs, iterations=iterations, seed=seed, alpha=alpha)
        rows.append(
            MetricRow(
                metric=metric,
                n=len(shared),
                baseline_macro=sum(a) / len(a),
                candidate_macro=sum(b) / len(b),
                diff=sum(diffs) / len(diffs),
                low=low,
                high=high,
            )
        )
    missing = sorted(set(baseline.scores) ^ set(candidate.scores))
    if missing:
        warnings.append(f"metric chỉ có ở một nhánh, không so: {', '.join(missing)}")
    return rows, warnings


def format_table(
    baseline: Branch, candidate: Branch, rows: Sequence[MetricRow], warnings: Sequence[str]
) -> str:
    head = [
        f"# Ablation tầng sinh — `{candidate.name}` so với `{baseline.name}`",
        "",
        f"Mốc: `{', '.join(baseline.models) or '?'}` · "
        f"Ứng viên: `{', '.join(candidate.models) or '?'}` · "
        f"Judge: `{baseline.judge_model}`",
        "",
        "Cột **macro** là trung bình theo truy vấn (đơn vị lấy mẫu lại của bootstrap), "
        "khác micro-average mà `generation_metrics` báo. `CI95` là khoảng tin cậy "
        "của **hiệu**; chứa 0 nghĩa là tập này **không phân biệt được** hai nhánh — "
        "một phát biểu về cỡ mẫu ít nhất ngang bằng một phát biểu về hệ thống.",
        "",
        "| metric | n | mốc | ứng viên | hiệu | CI95 | tốt hơn |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    body = [
        f"| `{r.metric}` | {r.n} | {r.baseline_macro:.4f} | {r.candidate_macro:.4f} | "
        f"{r.diff:+.4f} | [{r.low:+.4f}, {r.high:+.4f}] | {r.better} |"
        for r in rows
    ]
    tail = ["", "## Cảnh báo", "", *[f"* {w}" for w in warnings]] if warnings else []
    return "\n".join([*head, *body, *tail]) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pipeline.eval.ablation_generation",
        description="W5-11 — so hai model sinh trên cùng retrieval stack, có kiểm định",
    )
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--baseline", required=True, help="tên run làm mốc")
    parser.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    branches = {b.name: b for b in (load_branch(p) for p in args.run)}
    if args.baseline not in branches:
        parser.error(f"không thấy mốc {args.baseline!r} trong {sorted(branches)}")
    baseline = branches.pop(args.baseline)

    chunks: list[str] = []
    payload: dict[str, Any] = {"baseline": baseline.name, "comparisons": {}}
    for name, candidate in sorted(branches.items()):
        rows, warnings = compare_branches(
            baseline, candidate, iterations=args.iterations, seed=args.seed, alpha=args.alpha
        )
        chunks.append(format_table(baseline, candidate, rows, warnings))
        payload["comparisons"][name] = {
            "models": list(candidate.models),
            "rows": [r.as_dict() for r in rows],
            "warnings": warnings,
        }

    text = "\n---\n\n".join(chunks)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("đã ghi %s", args.out)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
