"""Hiệu chỉnh bộ dò từ chối theo **từng model sinh** — `W5-11`, trả `TD-77`.

## Vì sao con số này phải chạy lại mỗi lần đổi model

`rag_core.generation.refusal.looks_like_refusal` là một danh sách từ khoá. Nó nuôi
ô `rag_refusals_suspected` trên bảng Grafana, và `W5-07` đã đo được nó lệch bao
nhiêu so với nhãn judge: F1 **0,721** ở bản đầu (báo **thiếu** 24,5% số lần từ
chối — hướng chệch tệ nhất, vì nó làm hệ thống trông tốt hơn thực tế), rồi
**0,889** sau khi bổ sung từ khoá, đo trên nửa giữ ngoài.

Nhưng con số ấy là của **một** model trả lời trên **một** corpus. Cách nói *"tôi
không tìm thấy"* là một thói quen văn phong, và nó đổi theo model. Một danh sách
từ khoá hiệu chỉnh trên `deepseek-v4-flash` có thể mù hoàn toàn với cách GLM
viết cùng ý ấy — và triệu chứng sẽ là ô Grafana tụt xuống, trông y hệt *"hệ
thống bớt từ chối hơn"*.

Nên `TD-77` đòi F1 là một **cột trong bảng ablation**, không phải một hằng số
trong báo cáo `W5-07`.

## Vì sao nó tốn $0

Nhãn judge đã nằm trong cache địa chỉ theo nội dung của `W5-03`. Module này chạy
với `frozen_cache=True`: mọi lượt tra trượt là **lỗi**, không phải một lời gọi
mới. Tức nó không thể lặng lẽ biến thành một phép đo tốn tiền, và nếu ai đó sửa
rubric thì nó đỏ thay vì âm thầm chấm lại dưới một câu hỏi khác.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.generation.refusal import looks_like_refusal

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DetectorScore", "calibrate", "main"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorScore:
    """Bộ dò từ khoá so với nhãn judge, trên **một** tập câu trả lời."""

    n: int
    tp: int
    fp: int
    fn: int
    judge_rate: float
    detector_rate: float
    missed_examples: tuple[str, ...] = ()

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def bias(self) -> float:
        """Lệch tương đối của tỉ lệ ước lượng so với tỉ lệ judge.

        Dấu quan trọng hơn độ lớn: âm nghĩa là bảng báo **thiếu** số lần từ
        chối, tức nó làm hệ thống trông tốt hơn thực tế — cùng họ với `TD-55`.
        """
        return (self.detector_rate - self.judge_rate) / self.judge_rate if self.judge_rate else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "judge_rate": round(self.judge_rate, 4),
            "detector_rate": round(self.detector_rate, 4),
            "bias": round(self.bias, 4),
            "missed_examples": list(self.missed_examples),
        }


def calibrate(
    answers: Sequence[tuple[str, str]],
    labels: dict[str, str | None],
    *,
    n_examples: int = 5,
) -> DetectorScore:
    """`answers` là `(query_id, text)`; `labels` là nhãn judge theo `query_id`.

    Câu có `label is None` (judge không đọc được) bị **loại khỏi cả tử số lẫn
    mẫu số** — cùng quy ước với `score_relevancy`. Tính chúng thành "không từ
    chối" là ghi lỗi của judge thành một thuộc tính của bộ dò.
    """
    tp = fp = fn = 0
    judged = 0
    detected = 0
    missed: list[str] = []
    for query_id, text in answers:
        label = labels.get(query_id)
        if label is None:
            continue
        judged += 1
        truth = label == "REFUSAL"
        guess = looks_like_refusal(text)
        detected += int(guess)
        if truth and guess:
            tp += 1
        elif guess and not truth:
            fp += 1
        elif truth and not guess:
            fn += 1
            if len(missed) < n_examples:
                missed.append(text.strip().replace("\n", " ")[:160])
    return DetectorScore(
        n=judged,
        tp=tp,
        fp=fp,
        fn=fn,
        judge_rate=(tp + fn) / judged if judged else 0.0,
        detector_rate=detected / judged if judged else 0.0,
        missed_examples=tuple(missed),
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .answer_run import load_answer_run
    from .generation_metrics import score_relevancy
    from .judge import DEEPSEEK_BASE_URL, DEFAULT_JUDGE_MODEL, JudgeConfig, build_judge

    parser = argparse.ArgumentParser(
        prog="python -m pipeline.eval.refusal_calibration",
        description="W5-11 — F1 của bộ dò từ chối theo từng model sinh (TD-77)",
    )
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--cache", type=Path, default=Path(".cache/judge.sqlite3"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-judge-calls",
        action="store_true",
        help="Bỏ `frozen_cache`. Mặc định TẮT: module này phải luôn tốn $0.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from rag_core.llm import GLM_BASE_URL

    judge = build_judge(
        JudgeConfig(
            model=args.judge_model,
            base_url=GLM_BASE_URL if args.judge_model.startswith("glm-") else DEEPSEEK_BASE_URL,
            cache_path=args.cache,
            frozen_cache=not args.allow_judge_calls,
        )
    )

    out: dict[str, Any] = {"judge_model": args.judge_model, "branches": {}}
    for path in args.run:
        run = load_answer_run(path)
        _, labels = score_relevancy(run.records, judge)
        score = calibrate([(r.query_id, r.answer) for r in run.records], labels)
        out["branches"][run.name] = {"models": run.models, **score.as_dict()}
        logger.info(
            "%s (%s): P=%.3f R=%.3f F1=%.3f · judge %.1f%% vs dò %.1f%% (lệch %+.1f%%)",
            run.name,
            ", ".join(run.models),
            score.precision,
            score.recall,
            score.f1,
            score.judge_rate * 100,
            score.detector_rate * 100,
            score.bias * 100,
        )

    text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        logger.info("đã ghi %s", args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
