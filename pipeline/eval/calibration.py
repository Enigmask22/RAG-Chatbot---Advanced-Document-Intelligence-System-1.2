"""Hiệu chỉnh judge với nhãn tay. `W5-04`.

`W5-01` báo faithfulness `0,9877` và gọi đó là qua ngưỡng `0,92`. Con số ấy do
`deepseek-v4-flash` sinh ra. Câu hỏi mà `W5-03` cố tình để lại — *"12 mẫu tự
soạn chứng minh suy luận là thừa trên ca dễ; nó không nói gì về ca khó"* — bây
giờ mới trả lời được, vì bây giờ mới có ca khó thật.

Module này làm ba việc, mỗi việc là một lệnh con, và **thứ tự giữa chúng là một
phần của phương pháp**:

    sample → (gán nhãn tay) → cross → score

## ⭐⭐ 1. Mẫu phải phân tầng, và điều đó buộc phải công bố hai con số

Phân bố nhãn faithfulness của `w5-answers-v1` là 402/26/5/0. Lấy 50 mẫu ngẫu
nhiên đều thì **kỳ vọng 0,58 mẫu `NOT_FOUND`** — nhánh judge dễ sai nhất thường
không có mẫu nào, và toàn bộ phép hiệu chỉnh sẽ chỉ nói được một câu: "judge gán
`SUPPORTED` cho những thứ vốn là `SUPPORTED`".

Nên tầng hiếm bị bơm lên có chủ đích, và hệ quả là kappa tính trên 50 mẫu ấy
**không** phải kappa của quần thể. `kappa.py` giữ cả hai con số; xem docstring ở
đó. Ở đây chỉ ghi lại điều buộc phải nhớ: một báo cáo in duy nhất một con số
kappa trên mẫu phân tầng là một báo cáo sai, dù con số ấy tính đúng.

## ⭐⭐ 2. Người gán nhãn không được nhìn phán quyết của judge

Đây là điều dễ làm hỏng nhất và dễ giả vờ đã làm nhất. Nếu file gán nhãn có sẵn
nhãn của judge ở cột bên cạnh, thì cái đo được không còn là "người nghĩ gì" mà
là "người có phản đối judge không" — và mọi thiên lệch đều đi về phía đồng thuận.

Nên `sample` ghi **hai** file:

* `*-blind.jsonl` — chỉ có `ref`, `context`, `claim`. Không có nhãn nào.
* `*-sealed.jsonl` — phán quyết của judge, trọng số, tầng.

và ghi `sha256` của file sealed vào header của file blind. `score` kiểm lại hash
ấy trước khi tính. Điều đó chứng minh được rằng phán quyết **không bị sửa sau
khi thấy nhãn tay**; nó không chứng minh được rằng người gán nhãn không đọc trộm.
Giới hạn ấy phải nằm trong báo cáo chứ không nằm im ở đây.

## ⭐ 3. Câu hỏi gửi cho judge chéo phải **giống hệt** câu hỏi gốc

`cross` không dựng lại câu hỏi từ mẫu; nó dựng lại từ **chính** `answer_run` +
sidecar bằng cùng đường mã như `score_faithfulness`, rồi lọc theo `ref`. Nếu
dựng lại từ file blind thì một khác biệt khoảng trắng cũng đủ làm hai judge trả
lời hai câu hỏi hơi khác nhau — và bất đồng ấy sẽ bị đọc thành bất đồng về nội
dung.

## ⭐ 4. Hai nhánh judge **không** so được ở điều kiện suy luận giống nhau

`glm-5.3-flash` trả HTTP 400 khi bị yêu cầu tắt suy luận (`MIN_REASONING` trong
`rag_core.llm` ghi lại phép đo). Mức thấp nhất nó nhận là `low`, còn DeepSeek
tắt hẳn được. Nên mọi bất đồng giữa hai nhánh đều mang lẫn một phần khác biệt về
điều kiện, không thuần là khác biệt về model. Không sửa được — chỉ ghi ra.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_run import AnswerRecord, load_answer_run, load_chunk_sidecar
from .generation_metrics import (
    FAITHFULNESS_LABELS,
    RELEVANCY_LABELS,
    faithfulness_questions,
)
from .judge import Judge, JudgeQuestion, JudgeVerdict
from .kappa import (
    Pair,
    accuracy_of,
    bootstrap_ci,
    cohen_kappa,
    confusion,
    pabak,
    per_label,
    rate_of,
)

__all__ = [
    "DEFAULT_ALLOCATION",
    "RUBRICS",
    "Rubric",
    "Sampled",
    "allocate",
    "build_questions",
    "cmd_cross",
    "cmd_sample",
    "cmd_score",
    "main",
    "parse_allocation",
    "sha256_of",
    "stratified_sample",
]


@dataclass(frozen=True)
class Rubric:
    prompt_id: str
    labels: tuple[str, ...]
    numerator: tuple[str, ...]
    denominator: tuple[str, ...]
    """Tử/mẫu của chính metric mà rubric này nuôi — để `score` tính lại được
    con số headline từ nhãn tay bằng **cùng một công thức**, chỉ đổi nguồn nhãn.
    """


RUBRICS: dict[str, Rubric] = {
    "faithfulness": Rubric(
        prompt_id="judge-faithfulness",
        labels=FAITHFULNESS_LABELS,
        numerator=("SUPPORTED",),
        denominator=("SUPPORTED", "CONTRADICTED", "NOT_FOUND"),
    ),
    "relevancy": Rubric(
        prompt_id="judge-answer-relevancy",
        labels=RELEVANCY_LABELS,
        numerator=("RELEVANT",),
        denominator=RELEVANCY_LABELS,
    ),
}

DEFAULT_ALLOCATION: dict[str, dict[str, int]] = {
    # Phân bổ là một **quyết định được khai**, không phải kết quả nổi lên từ dữ
    # liệu. Ghi thẳng ở đây rồi ghi lại vào header của mẫu, để ai đọc báo cáo
    # cũng tính lại được trọng số mà không phải đoán.
    "faithfulness": {"SUPPORTED": 30, "NO_CLAIM": 15, "NOT_FOUND": 5, "CONTRADICTED": 5},
    "relevancy": {"RELEVANT": 20, "REFUSAL": 15, "PARTIAL": 11, "IRRELEVANT": 4},
}

UNPARSEABLE = "UNPARSEABLE"
"""Tầng riêng cho `label=None`. Chấm tay được — và nếu người đọc ra nhãn dứt
khoát ở những mục judge bó tay thì đó là một phát hiện, không phải nhiễu."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ dựng câu hỏi


def build_questions(
    rubric_name: str,
    records: Sequence[AnswerRecord],
    chunks: Mapping[str, str],
) -> list[JudgeQuestion]:
    """Dựng **đúng** tập câu hỏi mà `generation_metrics` đã hỏi ở lần chạy gốc.

    Không sao chép logic: faithfulness đi qua chính `faithfulness_questions`, nên
    khoá cache trùng khít và `sample` chạy được ở `frozen_cache` với chi phí $0.
    Nếu chép lại thì một khác biệt nhỏ sẽ biến thành hàng trăm lượt trượt cache —
    tức là trả tiền để chấm lại đúng thứ đã chấm, và ra một tập phán quyết thứ
    hai cho cùng một câu hỏi.
    """
    if rubric_name == "faithfulness":
        out: list[JudgeQuestion] = []
        for record in records:
            if record.route != "retrieve":
                continue
            out.extend(
                question
                for _, question in faithfulness_questions(record, chunks)
                if question is not None
            )
        return out
    if rubric_name == "relevancy":
        return [
            JudgeQuestion(
                prompt_id="judge-answer-relevancy",
                labels=RELEVANCY_LABELS,
                variables={"question": record.query, "answer": record.answer},
                ref=record.query_id,
            )
            for record in records
            if record.answer.strip()
        ]
    raise ValueError(f"rubric lạ: {rubric_name!r}. Có: {sorted(RUBRICS)}")


# --------------------------------------------------------------------- lấy mẫu


def parse_allocation(text: str) -> dict[str, int]:
    """`"SUPPORTED=30,NOT_FOUND=5"` → dict. Sai cú pháp là lỗi, không phải bỏ qua."""
    out: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        label, _, count = part.partition("=")
        if not count.strip().isdigit():
            raise ValueError(f"phân bổ hỏng ở {part!r} — dạng đúng là NHÃN=SỐ")
        out[label.strip()] = int(count)
    if not out:
        raise ValueError("phân bổ rỗng")
    return out


def allocate(
    population: Mapping[str, int],
    wanted: Mapping[str, int],
) -> dict[str, int]:
    """Số mẫu thực lấy ở mỗi tầng = `min(muốn, có)`.

    Không bù sang tầng khác khi thiếu. Bù tự động sẽ lặng lẽ đổi trọng số của
    tầng nhận bù, và cỡ mẫu cuối cùng sẽ không còn khớp con số ghi trong báo cáo
    — trong khi thiếu mẫu ở một tầng là chuyện phải **thấy**, không phải chuyện
    phải che.
    """
    return {
        label: min(count, population.get(label, 0))
        for label, count in wanted.items()
        if population.get(label, 0) > 0
    }


@dataclass(frozen=True)
class Sampled:
    ref: str
    stratum: str
    weight: float
    variables: Mapping[str, str]
    judge_label: str | None
    judge_reason: str


def stratified_sample(
    questions: Sequence[JudgeQuestion],
    verdicts: Sequence[JudgeVerdict],
    wanted: Mapping[str, int],
    *,
    seed: int,
) -> list[Sampled]:
    """Lấy mẫu phân tầng theo nhãn của judge, tất định theo `seed`.

    Sắp theo `ref` trước khi bốc: `Judge.ask_many` giữ thứ tự đầu vào, nhưng thứ
    tự ấy đến từ thứ tự dòng trong file run. Sắp lại làm mẫu **không** đổi khi
    ai đó thêm một truy vấn vào cuối golden set.
    """
    by_stratum: dict[str, list[tuple[JudgeQuestion, JudgeVerdict]]] = {}
    for question, verdict in zip(questions, verdicts, strict=True):
        by_stratum.setdefault(verdict.label or UNPARSEABLE, []).append((question, verdict))

    population = {label: len(items) for label, items in by_stratum.items()}
    taken = allocate(population, wanted)

    out: list[Sampled] = []
    for stratum in sorted(taken):
        items = sorted(by_stratum[stratum], key=lambda pair: pair[0].ref)
        rng = random.Random(f"{seed}:{stratum}")
        picked = rng.sample(items, taken[stratum])
        weight = population[stratum] / taken[stratum]
        out.extend(
            Sampled(
                ref=question.ref,
                stratum=stratum,
                weight=weight,
                variables=dict(question.variables),
                judge_label=verdict.label,
                judge_reason=verdict.reason,
            )
            for question, verdict in sorted(picked, key=lambda pair: pair[0].ref)
        )
    # ⭐⭐ Trộn thứ tự — và đây là một bản vá do chính lần dùng đầu tiên bắt được.
    #
    # Bản đầu ghi ra theo tầng: 5 mục `NOT_FOUND`, rồi 15 mục `NO_CLAIM`, rồi 30
    # mục `SUPPORTED`. File mù không chứa nhãn nào, nhưng **vị trí dòng** thì
    # chứa: đọc tới mục thứ 18 là suy ra được ranh giới, và từ đó trở đi người
    # gán nhãn biết trước judge đã nói gì. Một file "mù" rò rỉ qua thứ tự vẫn là
    # rò rỉ, chỉ khó thấy hơn.
    #
    # Sắp theo `ref` cũng không cứu được: `ref` là `query_id#chỉ_số_câu`, mà
    # `unanswerable-*` sinh ra phần lớn câu meta còn `#s0` phần lớn là câu dẫn —
    # thứ tự ấy vẫn tương quan với nhãn. Chỉ có trộn mới làm thứ tự **không mang
    # thông tin** theo đúng nghĩa xây dựng được.
    random.Random(f"{seed}:order").shuffle(out)
    return out


# ------------------------------------------------------------------------- I/O


def _write_jsonl(path: Path, header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_header": dict(header)}, ensure_ascii=False)]
    lines += [json.dumps(dict(row), ensure_ascii=False) for row in rows]
    path.write_text("".join(line + chr(10) for line in lines), encoding="utf-8")


def _read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or "_header" not in rows[0]:
        raise ValueError(f"{path} thiếu dòng header")
    return rows[0]["_header"], rows[1:]


# ------------------------------------------------------------------ lệnh: sample


def _load_run(args: Any) -> tuple[Any, dict[str, str]]:
    run = load_answer_run(args.run)
    chunks = load_chunk_sidecar(args.run)
    if not chunks:
        raise SystemExit(f"thiếu sidecar nội dung chunk cho {args.run}")
    return run, chunks


def _build_judge(
    args: Any, *, frozen: bool, model: str, base_url: str, cache: Path, reasoning: bool = False
) -> Judge:
    from .judge import JudgeConfig, build_judge

    return build_judge(
        JudgeConfig(
            model=model,
            base_url=base_url,
            cache_path=cache,
            cap_usd=args.cap_usd,
            concurrency=args.concurrency,
            frozen_cache=frozen,
            reasoning=reasoning,
        )
    )


def cmd_sample(args: Any) -> int:
    """Bốc mẫu phân tầng, phát lại phán quyết từ cache — **$0, không gọi mạng**.

    `frozen_cache=True` là bắt buộc chứ không phải tối ưu: nếu một câu hỏi ở đây
    trượt cache thì nghĩa là mẫu này không phải mẫu của lần chạy đã báo cáo, và
    tiếp tục sẽ hiệu chỉnh một tập phán quyết khác với tập đã sinh ra `0,9877`.
    """
    run, chunks = _load_run(args)
    rubric = RUBRICS[args.rubric]
    questions = build_questions(args.rubric, run.records, chunks)
    judge = _build_judge(
        args, frozen=True, model=args.model, base_url=args.base_url, cache=args.cache
    )
    verdicts = judge.ask_many(questions)

    wanted = parse_allocation(args.allocate) if args.allocate else DEFAULT_ALLOCATION[args.rubric]
    picked = stratified_sample(questions, verdicts, wanted, seed=args.seed)

    population: dict[str, int] = {}
    for verdict in verdicts:
        key = verdict.label or UNPARSEABLE
        population[key] = population.get(key, 0) + 1

    stem = args.out
    sealed_path = stem.with_name(f"{stem.stem}-sealed.jsonl")
    blind_path = stem.with_name(f"{stem.stem}-blind.jsonl")

    sealed_header = {
        "run": run.name,
        "rubric": judge.registry.get(rubric.prompt_id).spec,
        "rubric_sha256": judge.registry.get(rubric.prompt_id).sha256,
        "judge_model": judge.config.model,
        "judge_reasoning": judge.config.reasoning,
        "cache_digest": judge.cache.digest(),
        "population": dict(sorted(population.items())),
        "allocation": dict(sorted(wanted.items())),
        "seed": args.seed,
    }
    _write_jsonl(
        sealed_path,
        sealed_header,
        [
            {
                "ref": s.ref,
                "stratum": s.stratum,
                "weight": s.weight,
                "judge_label": s.judge_label,
                "judge_reason": s.judge_reason,
            }
            for s in picked
        ],
    )
    _write_jsonl(
        blind_path,
        {
            "run": run.name,
            "rubric": sealed_header["rubric"],
            "labels": list(rubric.labels),
            "n": len(picked),
            "seed": args.seed,
            # ⭐ Cam kết bằng hash: `score` từ chối chạy nếu file sealed đã đổi.
            # Nó chứng minh phán quyết không bị sửa **sau khi** thấy nhãn tay.
            # Nó không chứng minh người gán nhãn đã không đọc trộm — giới hạn ấy
            # thuộc về báo cáo, không thuộc về mã.
            "sealed_sha256": sha256_of(sealed_path),
            "sealed_file": sealed_path.name,
        },
        [{"ref": s.ref, "gold": None, "why": "", **s.variables} for s in picked],
    )
    print(json.dumps({"population": population, "taken": len(picked)}, ensure_ascii=False))
    print(f"đã ghi {blind_path} (mù) và {sealed_path} (niêm phong)")
    print(f"gán nhãn tay vào {stem.with_name(f'{stem.stem}-human.jsonl')}: mỗi dòng {{ref, gold}}")
    return 0


# ------------------------------------------------------------------- lệnh: cross


def cmd_cross(args: Any) -> int:
    """Chấm lại **đúng** những câu hỏi ấy bằng một judge khác họ."""
    run, chunks = _load_run(args)
    _, rows = _read_jsonl(args.blind)
    refs = {row["ref"] for row in rows}

    questions = [q for q in build_questions(args.rubric, run.records, chunks) if q.ref in refs]
    missing = refs - {q.ref for q in questions}
    if missing:
        raise SystemExit(
            f"{len(missing)} ref trong mẫu không dựng lại được từ run: {sorted(missing)[:5]}"
        )

    judge = _build_judge(
        args,
        frozen=False,
        model=args.model,
        base_url=args.base_url,
        cache=args.cache,
        reasoning=args.reasoning,
    )
    verdicts = judge.ask_many(questions)
    _write_jsonl(
        args.out,
        {
            "run": run.name,
            "judge_model": judge.config.model,
            "judge_family": judge.config.family,
            "judge_reasoning": judge.config.reasoning,
            # Không phải "tắt": `glm-5.3-flash` trả HTTP 400 khi bị yêu cầu tắt.
            "reasoning_body": judge.config.reasoning_body,
            "served_models": dict(sorted(judge.stats.served_models.items())),
            "spent_usd": round(judge.stats.spent_usd, 6),
            "unparseable": judge.stats.unparseable,
        },
        [
            {"ref": v.ref, "label": v.label, "reason": v.reason, "served_model": v.served_model}
            for v in sorted(verdicts, key=lambda v: v.ref)
        ],
    )
    print(json.dumps(judge.stats.as_dict(), ensure_ascii=False, indent=2))
    print(f"đã ghi {args.out}")
    return 0


# ------------------------------------------------------------------- lệnh: score


def _labels_of(rows: Sequence[Mapping[str, Any]], field_name: str) -> dict[str, str]:
    """Nhãn `None`/rỗng thành `UNPARSEABLE` — một tầng thật, không phải khoảng trống.

    Ba nguồn nhãn (người, judge, judge chéo) đi qua đúng hàm này, nên "judge bó
    tay" và "người bỏ trống" được đối xử như nhau và cùng hiện lên trong ma trận
    nhầm lẫn thay vì biến mất khỏi mẫu số.
    """
    out: dict[str, str] = {}
    for row in rows:
        value = row.get(field_name)
        out[str(row["ref"])] = UNPARSEABLE if value in (None, "") else str(value)
    return out


def _agreement(
    pairs: Sequence[Pair], labels: Sequence[str], *, weighted: bool, seed: int
) -> dict[str, Any]:
    """Một nhánh đồng thuận, tính hai lần: trên mẫu và quy về quần thể."""
    view = [p if weighted else Pair(p.a, p.b, stratum=p.stratum, ref=p.ref) for p in pairs]
    cm = confusion(view, labels)
    ci = bootstrap_ci(
        view, lambda batch: cohen_kappa(confusion(batch, labels)), n_resamples=2000, seed=seed
    )
    return {
        "n_items": cm.n_items,
        "observed_agreement": accuracy_of(view),
        "kappa": cohen_kappa(cm),
        "kappa_ci": {k: ci[k] for k in ("lo", "hi", "n_undefined")},
        "pabak": pabak(cm),
        "confusion": cm.as_dict(),
        "per_label": per_label(cm),
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def cmd_score(args: Any) -> int:
    blind_header, blind_rows = _read_jsonl(args.blind)
    sealed_path = args.blind.with_name(str(blind_header["sealed_file"]))
    actual = sha256_of(sealed_path)
    if actual != blind_header["sealed_sha256"]:
        raise SystemExit(
            f"{sealed_path} đã đổi kể từ lúc lấy mẫu ({actual[:12]} ≠ "
            f"{str(blind_header['sealed_sha256'])[:12]}). Phán quyết của judge phải cố "
            "định TRƯỚC khi có nhãn tay — nếu không thì con số kappa dưới đây không "
            "nói lên điều gì."
        )
    sealed_header, sealed_rows = _read_jsonl(sealed_path)
    _, human_rows = _read_jsonl(args.human)

    rubric = RUBRICS[args.rubric]
    valid = {*rubric.labels, UNPARSEABLE}
    judge = _labels_of(sealed_rows, "judge_label")
    human = _labels_of(human_rows, "gold")
    if set(judge) != set(human):
        only_j, only_h = sorted(set(judge) - set(human)), sorted(set(human) - set(judge))
        raise SystemExit(
            f"tập ref lệch nhau — thiếu nhãn tay: {only_j[:5]}, thừa: {only_h[:5]}. "
            "Bỏ qua mục chưa gán nhãn sẽ lặng lẽ đổi cỡ mẫu của một tầng, và trọng "
            "số của tầng ấy sẽ sai mà không có gì báo."
        )
    bad = {ref: label for ref, label in human.items() if label not in valid}
    if bad:
        raise SystemExit(f"nhãn tay ngoài tập khai báo: {dict(list(bad.items())[:5])}")

    meta = {str(row["ref"]): row for row in sealed_rows}
    why = {str(row["ref"]): str(row.get("why", "")) for row in human_rows}
    text = {str(row["ref"]): row for row in blind_rows}
    cross: dict[str, str] = {}
    cross_header: dict[str, Any] = {}
    if args.cross:
        cross_header, cross_rows = _read_jsonl(args.cross)
        cross = _labels_of(cross_rows, "label")
        if set(cross) != set(judge):
            raise SystemExit("tập ref của judge chéo lệch với mẫu")

    labels = (*rubric.labels, UNPARSEABLE)

    def pairs_of(a: Mapping[str, str], b: Mapping[str, str]) -> list[Pair]:
        return [
            Pair(
                a=a[ref],
                b=b[ref],
                stratum=str(meta[ref]["stratum"]),
                weight=float(meta[ref]["weight"]),
                ref=ref,
            )
            for ref in sorted(judge)
        ]

    def both_views(pairs: Sequence[Pair]) -> dict[str, Any]:
        return {
            "sample": _agreement(pairs, labels, weighted=False, seed=args.seed),
            "population": _agreement(pairs, labels, weighted=True, seed=args.seed),
        }

    jh = pairs_of(human, judge)
    agreement: dict[str, Any] = {"judge_vs_human": both_views(jh)}
    if cross:
        agreement["cross_vs_human"] = both_views(pairs_of(human, cross))
        agreement["judge_vs_cross"] = both_views(pairs_of(judge, cross))

    def headline(pairs: Sequence[Pair], side: str) -> dict[str, Any]:
        def rate(batch: Sequence[Pair]) -> float | None:
            return rate_of(
                batch, side=side, numerator=rubric.numerator, denominator=rubric.denominator
            )

        return bootstrap_ci(pairs, rate, n_resamples=2000, seed=args.seed)

    report: dict[str, Any] = {
        "task": "W5-04",
        "run": blind_header["run"],
        "rubric": blind_header["rubric"],
        "rubric_sha256": sealed_header.get("rubric_sha256"),
        "judge_model": sealed_header.get("judge_model"),
        "judge_reasoning": sealed_header.get("judge_reasoning"),
        "cross": (
            {k: cross_header.get(k) for k in ("judge_model", "judge_family", "reasoning_body")}
            if cross
            else None
        ),
        "sealed_sha256": blind_header["sealed_sha256"],
        "cache_digest": sealed_header.get("cache_digest"),
        "n": len(judge),
        "population": sealed_header.get("population"),
        "allocation": sealed_header.get("allocation"),
        "weights": {str(row["stratum"]): round(float(row["weight"]), 4) for row in sealed_rows},
        "label_counts": {
            "human": _count(human.values()),
            "judge": _count(judge.values()),
            **({"cross": _count(cross.values())} if cross else {}),
        },
        "agreement": agreement,
        "headline": {
            # ⭐ Tự kiểm gắn sẵn: tầng **chính là** nhãn của judge, nên tỉ lệ có
            # trọng số ở phía judge bắt buộc trùng khít con số đã báo cáo ở
            # `W5-01`. Lệch một chút nghĩa là trọng số sai — và khi ấy con số
            # phía người, thứ duy nhất ta thật sự muốn biết, cũng sai theo.
            "judge_reweighted": headline(jh, "b"),
            "human_reweighted": headline(jh, "a"),
            # ⭐⭐ Con số quan trọng nhất của cả `W5-04`: metric headline sẽ ra
            # bao nhiêu nếu **chỉ** đổi model judge, giữ nguyên hệ thống, nguyên
            # rubric, nguyên câu trả lời. Không tính ra thì không ai biết phần
            # nào của `0,9877` thuộc về hệ thống và phần nào thuộc về giám khảo.
            **({"cross_reweighted": headline(pairs_of(human, cross), "b")} if cross else {}),
        },
        "disagreements": [
            {
                "ref": ref,
                "stratum": meta[ref]["stratum"],
                "human": human[ref],
                "judge": judge[ref],
                **({"cross": cross[ref]} if cross else {}),
                "why_human": why[ref],
                "judge_reason": meta[ref]["judge_reason"],
                **{k: v for k, v in text[ref].items() if k not in ("ref", "gold", "why")},
            }
            for ref in sorted(judge)
            if human[ref] != judge[ref]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "n": report["n"],
        "agreement": {
            name: {
                "sample_kappa": arm["sample"]["kappa"],
                "sample_po": arm["sample"]["observed_agreement"],
                "population_kappa": arm["population"]["kappa"],
                "population_po": arm["population"]["observed_agreement"],
                "population_pabak": arm["population"]["pabak"],
            }
            for name, arm in agreement.items()
        },
        "headline": report["headline"],
        "n_disagreements": len(report["disagreements"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"đã ghi {args.out}")
    return 0


# -------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import logging

    from rag_core.llm import GLM_BASE_URL

    from .judge import DEEPSEEK_BASE_URL, DEFAULT_JUDGE_MODEL

    parser = argparse.ArgumentParser(description="W5-04 — hiệu chỉnh judge với nhãn tay")
    parser.add_argument("--rubric", choices=sorted(RUBRICS), default="faithfulness")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--cap-usd", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=6)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="bốc mẫu phân tầng, phát lại từ cache ($0)")
    p_sample.add_argument("--run", type=Path, required=True)
    p_sample.add_argument("--out", type=Path, required=True, help="tiền tố, ví dụ .../w5-04")
    p_sample.add_argument("--cache", type=Path, default=Path(".cache/judge-w5.sqlite3"))
    p_sample.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    p_sample.add_argument("--base-url", default=DEEPSEEK_BASE_URL)
    p_sample.add_argument("--allocate", default="", help="NHÃN=SỐ,... (mặc định trong mã)")
    p_sample.set_defaults(func=cmd_sample)

    p_cross = sub.add_parser("cross", help="chấm lại đúng mẫu ấy bằng judge khác họ")
    p_cross.add_argument("--run", type=Path, required=True)
    p_cross.add_argument("--blind", type=Path, required=True)
    p_cross.add_argument("--out", type=Path, required=True)
    p_cross.add_argument("--cache", type=Path, default=Path(".cache/judge-w5-cross.sqlite3"))
    p_cross.add_argument("--model", default="glm-5.3-flash")
    p_cross.add_argument("--base-url", default=GLM_BASE_URL)
    p_cross.add_argument(
        "--reasoning",
        action="store_true",
        help=(
            "Bật suy luận. `W5-03` chốt mặc định TẮT trên 12 mẫu dễ tự soạn và ghi rõ "
            "rằng kết luận ấy chưa nói gì về ca khó — cờ này là cách hỏi lại câu đó "
            "trên 50 mệnh đề thật đã có nhãn tay."
        ),
    )
    p_cross.set_defaults(func=cmd_cross)

    p_score = sub.add_parser("score", help="ghép nhãn tay + judge + judge chéo, tính kappa")
    p_score.add_argument("--blind", type=Path, required=True)
    p_score.add_argument("--human", type=Path, required=True)
    p_score.add_argument("--cross", type=Path, default=None)
    p_score.add_argument("--out", type=Path, required=True)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
