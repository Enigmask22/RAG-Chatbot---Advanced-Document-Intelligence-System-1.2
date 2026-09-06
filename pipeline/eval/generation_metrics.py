"""Metric cho tầng sinh: faithfulness, answer relevancy, context precision/recall. `W5-01`.

Đọc artifact của `answer_run.py` — **không** gọi lại hệ thống, **không** cần stack.
Phần cần LLM đi qua `judge.py`, nơi có cache; nên chấm lại bằng rubric cũ là miễn phí.

## ⭐⭐ 1. Tách câu bằng luật, không bằng LLM

RAGAS phân rã câu trả lời thành "mệnh đề nguyên tử" bằng một lời gọi LLM nữa. Làm
vậy đặt **hai** model không tất định vào trong cùng một phép đo, và khi con số
dịch chuyển thì không quy được cho model nào.

Ở đây phân rã là **tách câu bằng luật**: tất định, miễn phí, và chỉ còn đúng một
LLM trong chuỗi — cái đó lại có cache.

⚠️ Giá phải trả, nói thẳng: câu ghép chứa một mệnh đề đúng và một mệnh đề sai bị
tính là **cả câu không được chống đỡ**. Tức phép đo này **nghiêm hơn** mức mệnh
đề, và nó lệch về phía báo faithfulness *thấp hơn* thực tế. Với một metric an
toàn thì đó là chiều lệch đúng: nó không bao giờ giấu một chỗ bịa bên trong một
câu phần lớn là đúng.

## ⭐⭐ 2. Chấm mệnh đề với **đúng nguồn nó trích**, không phải với hợp mọi nguồn

Luật 2 của `chat-system` bắt mỗi ý phải mang `[n]`. Nên mỗi câu tự khai nó dựa
vào nguồn nào, và phép đo đúng là hỏi *nguồn ấy*.

Chấm với hợp của mọi chunk đã truy hồi trả lời một câu **yếu hơn**: "văn bản này
có suy ra được từ thứ gì đó ta đã lấy về không". Câu ấy cho điểm cao hơn và
**giấu mất lỗi gán nhầm nguồn** — trích `[3]` cho một ý chỉ có trong `[1]` dẫn
người đọc tới nhầm tài liệu, và đó chính là thứ `W4-09` tồn tại để bắt.

Nên **gán nhầm nguồn trở thành một con số**, không phải một lo ngại:
`score_misattribution` hỏi lại hợp mọi chunk, nhưng **chỉ với những mệnh đề đã
trượt** ở vòng theo-nguồn-trích. Chạy cả hai vòng đầy đủ tốn gấp 2,6× (ngữ cảnh
hợp có 5 chunk thay vì 1–2) để đo một thứ mà định nghĩa đã nói là chỉ tồn tại
trong tập con ấy.

## ⭐ 3. Câu không trích nguồn nào KHÔNG bị tính là không trung thực

Nó là một lỗi khác — vi phạm luật 2 — và có phép đo riêng: `citation_coverage`.
Gộp hai thứ vào một số làm mất khả năng phân biệt "model bịa" với "model quên
trích", trong khi cách chữa hai bên hoàn toàn khác nhau.

Cặp số ấy luôn được công bố cùng nhau, nên không có đường nào để một model không
trích gì đạt điểm cao: faithfulness của nó không xác định và coverage bằng 0.

## ⭐ 4. Mọi tổng hợp đều khai `n_unjudged`

Phán quyết judge không đọc được bị loại khỏi **cả tử số lẫn mẫu số**. Xem điểm 3
ở docstring `judge.py`: quy nó thành "không được chống đỡ" là tính lỗi của judge
thành lỗi của hệ thống bị chấm.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .answer_run import AnswerRecord
from .golden import GoldenQuery
from .judge import Judge, JudgeQuestion, JudgeVerdict
from .metrics import precision_at_k, recall_at_k

__all__ = [
    "FAITHFULNESS_LABELS",
    "RELEVANCY_LABELS",
    "Aggregate",
    "SentenceClaim",
    "citation_accuracy",
    "citation_coverage",
    "citation_validity",
    "context_precision",
    "context_recall",
    "derived",
    "faithfulness_questions",
    "score_faithfulness",
    "score_misattribution",
    "score_refusal",
    "score_relevancy",
    "score_uncited_grounding",
    "split_sentences",
    "strip_markers",
]

FAITHFULNESS_LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_FOUND", "NO_CLAIM")
"""`NO_CLAIM` có mặt từ rubric v2, và nó đến từ một phép đo suýt sai.

Vòng chấm đầu cho `uncited_grounding = 0,427` và tôi định viết "22% nội dung
không có căn cứ". Đọc ví dụ thật thì gần như toàn bộ là câu **meta**:
`"Dựa trên ngữ cảnh được cung cấp:"`, `"tôi không đủ thông tin để trả lời"`,
`"Lưu ý: Các nguồn không nêu rõ…"`. Đó là model đang **tuân thủ luật 3** của
`chat-system`, và judge chấm `NOT_FOUND` là đúng theo rubric v1 — ngữ cảnh
quả thật không chứa lời khẳng định nào về việc chính nó thiếu thông tin.

Tức một lời từ chối trung thực đang bị đếm thành một lỗi trung thực. Rubric
v2 tách nó ra; `NO_CLAIM` bị loại khỏi cả tử số lẫn mẫu số và được đếm riêng."""
RELEVANCY_LABELS = ("RELEVANT", "PARTIAL", "IRRELEVANT", "REFUSAL")


# --------------------------------------------------------------- tách câu

#: Từ viết tắt kết thúc bằng dấu chấm mà **không** kết thúc câu.
#: Danh sách ngắn có chủ ý: mỗi mục là một ca đã gặp trong corpus World Bank
#: tiếng Việt hoặc trong câu trả lời thật, không phải một bộ sưu tập phòng xa.
ABBREVIATIONS = frozenset(
    {
        "tp.",
        "tt.",
        "q.",
        "p.",
        "gs.",
        "pgs.",
        "ts.",
        "ths.",
        "bs.",
        "vd.",
        "vs.",
        "v.v.",
        "no.",
        "st.",
        "mr.",
        "ms.",
        "dr.",
        "fig.",
        "usd.",
    }
)

_SENTENCE_END = re.compile(r"([.!?…])(\s+)")
_CITE = re.compile(r"\[(\d+)\]")


def _is_real_boundary(text: str, dot_index: int) -> bool:
    """Dấu chấm ở `dot_index` có thật sự kết thúc câu không.

    ## ⭐ Cái bảo vệ `1.234.567` là `\\s+` trong `_SENTENCE_END`, không phải hàm này

    Bản đầu có thêm một phép kiểm `\\d[.,]\\d` quanh dấu chấm, với lý lẽ: tiếng
    Việt dùng dấu chấm làm phân cách hàng nghìn, tách ở đó thì `1.234.567` vỡ
    thành hai "câu" vô nghĩa.

    Phép tiêm `G1` (bỏ phép kiểm ấy) **sống sót**, và lý do là nó không thể chạy:
    `_SENTENCE_END` đòi dấu chấm phải theo sau bởi khoảng trắng, còn phép kiểm
    đòi nó phải theo sau bởi một chữ số. Hai điều kiện loại trừ nhau.

    Nên phần bảo vệ ấy là **mã chết** — cùng loại với `J2` của `W5-03`, và cũng
    được xử lý như thế: gỡ đi. Một lớp bảo vệ không có tác dụng còn tệ hơn không
    có, vì người đọc sau sẽ tưởng nó đang giữ một bất biến.

    Còn lại đúng một việc: từ viết tắt (`v.v.`, `TP.`) — chúng **có** khoảng
    trắng theo sau nên chúng thật sự lọt qua regex.
    """
    head = text[:dot_index].rsplit(" ", 1)[-1].lower()
    return f"{head}." not in ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Tách câu bằng luật. Tất định, không gọi model — xem điểm 1 ở docstring."""
    stripped = text.strip()
    if not stripped:
        return []
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(stripped):
        end = match.start()
        if stripped[end] == "." and not _is_real_boundary(stripped, end):
            continue
        piece = stripped[start : match.end(1)].strip()
        if piece:
            out.append(piece)
        start = match.end()
    tail = stripped[start:].strip()
    if tail:
        out.append(tail)
    # Dòng xuống hàng cũng là ranh giới ý trong câu trả lời có gạch đầu dòng —
    # `- ý một [1]\n- ý hai [2]` là hai mệnh đề, không phải một.
    final: list[str] = []
    for piece in out:
        for line in piece.splitlines():
            cleaned = line.strip().lstrip("-•*").strip()
            if cleaned:
                final.append(cleaned)
    return final


def markers(sentence: str) -> tuple[int, ...]:
    """Các số nguồn `[n]` mà câu này tự khai, theo thứ tự xuất hiện, khử trùng."""
    seen: list[int] = []
    for raw in _CITE.findall(sentence):
        value = int(raw)
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def strip_markers(sentence: str) -> str:
    """Bỏ `[n]` trước khi đưa cho judge — chúng là siêu dữ liệu, không phải nội
    dung mệnh đề, và để lại thì judge sẽ đi tìm `[1]` trong ngữ cảnh."""
    return re.sub(r"\s*\[\d+\]", "", sentence).strip()


#: Câu quá ngắn sau khi bỏ marker thì không phải một mệnh đề kiểm được
#: (`"Cụ thể:"`, `"Tóm lại."`). Chấm chúng là bơm nhiễu vào cả tử lẫn mẫu.
MIN_CLAIM_CHARS = 25


@dataclass(frozen=True)
class SentenceClaim:
    query_id: str
    index: int
    text: str
    cited_ns: tuple[int, ...]

    @property
    def ref(self) -> str:
        return f"{self.query_id}#s{self.index}"


def claims_of(record: AnswerRecord) -> list[SentenceClaim]:
    """Mệnh đề kiểm được của một câu trả lời."""
    out: list[SentenceClaim] = []
    for index, sentence in enumerate(split_sentences(record.answer)):
        text = strip_markers(sentence)
        if len(text) < MIN_CLAIM_CHARS:
            continue
        out.append(SentenceClaim(record.query_id, index, text, markers(sentence)))
    return out


# --------------------------------------------------------- tổng hợp có nhãn


@dataclass
class Aggregate:
    """Một con số, kèm **mọi lý do** khiến mẫu số nhỏ hơn tổng số mục.

    Không có `value` trần trụi ở đâu trong module này: một tỉ lệ mà không biết
    mẫu số là gì thì không đọc được, và hai lý do bị loại dưới đây có ý nghĩa
    hoàn toàn khác nhau.

    Giữ **danh sách giá trị** chứ không giữ tử/mẫu: metric nhị phân
    (`faithfulness`) và metric liên tục (`context_precision`) dùng chung một lớp,
    và một cặp tử/mẫu nguyên không biểu diễn được cái thứ hai mà không phải nhân
    hệ số — tức là một chỗ làm tròn không ai nhớ.
    """

    name: str
    n_unjudged: int = 0
    """Judge không đọc được — **lỗi của phép đo**, không phải của hệ thống."""
    n_no_evidence: int = 0
    """Không có bằng chứng để chấm: câu không trích nguồn nào, chunk thiếu trong
    sidecar, hoặc truy vấn `unanswerable` (precision/recall không xác định)."""
    n_not_a_claim: int = 0
    """Judge trả `NO_CLAIM`: câu dẫn, đề mục, hoặc lời từ chối. Không phải một
    mệnh đề để kiểm — xem docstring `FAITHFULNESS_LABELS`."""
    per_category: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_query: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    """⭐⭐ `W5-11`: cùng những giá trị ấy, khoá theo **truy vấn**.

    `per_category` gộp mất danh tính truy vấn, nên hai lần chạy không **ghép
    cặp** được — mà ghép cặp đúng là thứ làm cho một bảng ablation có nghĩa.
    `W2-09` đã trả giá một lần cho bài học này ở tầng truy hồi: `hit_rate@5`
    "tụt 6,7%" hoá ra là 45 câu xuống 42 câu, chênh **ba câu**. Không có kiểm
    định thì thứ thắng chỉ là thứ may.

    Là `list` chứ không phải một số: `faithfulness` chấm theo **mệnh đề**, nên
    một truy vấn đóng góp nhiều giá trị. Điểm cấp truy vấn là trung bình của
    danh sách ấy — một **macro-average**, khác micro-average mà `value` trả về.
    Hai con số ấy trả lời hai câu hỏi khác nhau và báo cáo phải nói rõ đang đọc
    cái nào (xem `paired_values`).
    """

    @property
    def values(self) -> list[float]:
        return [value for values in self.per_category.values() for value in values]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def value(self) -> float | None:
        values = self.values
        return statistics.fmean(values) if values else None

    def paired_values(self) -> dict[str, float]:
        """Điểm **một số cho một truy vấn** — đầu vào của bootstrap ghép cặp.

        Trung bình trong nội bộ truy vấn trước, rồi mới ghép cặp: nếu không thì
        một truy vấn 12 mệnh đề nặng gấp 12 lần một truy vấn 1 mệnh đề, và mẫu
        bootstrap sẽ lấy lại theo mệnh đề chứ không theo **đơn vị độc lập** là
        truy vấn.
        """
        return {qid: statistics.fmean(vals) for qid, vals in self.per_query.items() if vals}

    def add(self, hit: bool, category: str, query_id: str | None = None) -> None:
        self.per_category[category].append(float(hit))
        if query_id is not None:
            self.per_query[query_id].append(float(hit))

    def add_value(self, score: float, category: str, query_id: str | None = None) -> None:
        self.per_category[category].append(float(score))
        if query_id is not None:
            self.per_query[query_id].append(float(score))

    def breakdown(self) -> dict[str, dict[str, float | int]]:
        return {
            category: {"n": len(values), "value": round(statistics.fmean(values), 4)}
            for category, values in sorted(self.per_category.items())
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 4) if self.value is not None else None,
            "n": self.n,
            "n_unjudged": self.n_unjudged,
            "n_no_evidence": self.n_no_evidence,
            "n_not_a_claim": self.n_not_a_claim,
            "by_category": self.breakdown(),
        }


# ------------------------------------------------------- metric tất định


def context_precision(
    records: Sequence[AnswerRecord], golden: Mapping[str, GoldenQuery], k: int = 5
) -> Aggregate:
    """Bao nhiêu phần của ngữ cảnh **đưa cho model** thật sự liên quan.

    Khác con số `W1-13`/`W2-*`: ở đây là truy hồi **sau** khi đi qua bộ định
    tuyến và bản viết lại của `W4-07`, tức đúng thứ người dùng nhận. Hai con số
    lệch nhau là thông tin, không phải mâu thuẫn.
    """
    return _label_metric(records, golden, k, precision_at_k, "context_precision")


def context_recall(
    records: Sequence[AnswerRecord], golden: Mapping[str, GoldenQuery], k: int = 5
) -> Aggregate:
    return _label_metric(records, golden, k, recall_at_k, "context_recall")


def _label_metric(
    records: Sequence[AnswerRecord],
    golden: Mapping[str, GoldenQuery],
    k: int,
    fn: Any,
    name: str,
) -> Aggregate:
    agg = Aggregate(f"{name}@{k}")
    for record in records:
        query = golden.get(record.query_id)
        if query is None:
            continue
        score = fn(record.source_chunk_ids, query.relevant_chunk_ids, k)
        if score is None:
            # Câu `unanswerable`: không có tài liệu liên quan nên precision/recall
            # không xác định. `metrics.py` đã chốt quy ước này từ `W1-08` — trả
            # `None` chứ không trả `0.0`; nhóm ấy đo riêng bằng refusal
            # correctness (`W5-02`).
            agg.n_no_evidence += 1
            continue
        agg.add_value(score, record.category, record.query_id)
    return agg


def citation_coverage(records: Sequence[AnswerRecord]) -> Aggregate:
    """Tỉ lệ mệnh đề có mang ít nhất một `[n]` — luật 2 của `chat-system`.

    Chỉ tính các lượt **có** truy hồi: nhánh `no_retrieval` (chào hỏi) và
    `clarify` không có nguồn nào để trích, nên đòi chúng trích là đòi sai.
    """
    agg = Aggregate("citation_coverage")
    for record in records:
        if record.route != "retrieve":
            agg.n_no_evidence += 1
            continue
        for claim in claims_of(record):
            agg.add(bool(claim.cited_ns), record.category, record.query_id)
    return agg


def citation_validity(records: Sequence[AnswerRecord]) -> Aggregate:
    """Tỉ lệ trích dẫn có `verified=True` — cơ chế `W4-09`, không cần judge.

    Đây là phần **tất định** của độ chính xác trích dẫn: quote có nằm nguyên văn
    trong đúng chunk được chỉ hay không. Phần cần judge (chunk có chống đỡ mệnh
    đề hay không) là `faithfulness`.
    """
    agg = Aggregate("citation_validity")
    for record in records:
        if not record.citations:
            agg.n_no_evidence += 1
            continue
        for citation in record.citations:
            agg.add(bool(citation.get("verified")), record.category, record.query_id)
    return agg


# ---------------------------------------------------------- metric có judge


def faithfulness_questions(
    record: AnswerRecord,
    chunks: Mapping[str, str],
    *,
    union: bool = False,
) -> list[tuple[SentenceClaim, JudgeQuestion | None]]:
    """Ghép mỗi mệnh đề với ngữ cảnh sẽ dùng để chấm nó.

    `None` nghĩa là **không chấm được** và lý do phải đi ra ngoài: câu không
    trích nguồn nào (khi `union=False`), hoặc chunk được trích không có trong
    sidecar (index đã đổi kể từ lần chạy).
    """
    by_n = {int(source["n"]): str(source["chunk_id"]) for source in record.sources}
    all_text = [chunks[cid] for cid in by_n.values() if cid in chunks]
    out: list[tuple[SentenceClaim, JudgeQuestion | None]] = []
    for claim in claims_of(record):
        if union:
            texts = all_text
        else:
            wanted = [by_n.get(n) for n in claim.cited_ns]
            texts = [chunks[cid] for cid in wanted if cid and cid in chunks]
        if not texts:
            out.append((claim, None))
            continue
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(texts, start=1))
        out.append(
            (
                claim,
                JudgeQuestion(
                    prompt_id="judge-faithfulness",
                    labels=FAITHFULNESS_LABELS,
                    variables={"context": context, "claim": claim.text},
                    ref=claim.ref + ("+union" if union else ""),
                ),
            )
        )
    return out


def score_faithfulness(
    records: Sequence[AnswerRecord],
    chunks: Mapping[str, str],
    judge: Judge,
    *,
    union: bool = False,
) -> tuple[Aggregate, list[tuple[SentenceClaim, JudgeVerdict]]]:
    """`SUPPORTED` / (SUPPORTED + CONTRADICTED + NOT_FOUND), theo mệnh đề.

    Trả kèm danh sách phán quyết để báo cáo trích được **ví dụ thật** — một con
    số faithfulness không có ví dụ đi kèm thì không ai kiểm được.
    """
    agg = Aggregate("faithfulness_union" if union else "faithfulness")
    pairs: list[tuple[SentenceClaim, JudgeQuestion | None]] = []
    for record in records:
        if record.route != "retrieve":
            continue
        pairs.extend(faithfulness_questions(record, chunks, union=union))

    askable = [(claim, question) for claim, question in pairs if question is not None]
    agg.n_no_evidence = len(pairs) - len(askable)
    verdicts = judge.ask_many([question for _, question in askable])

    category_of = {record.query_id: record.category for record in records}
    detail: list[tuple[SentenceClaim, JudgeVerdict]] = []
    for claim, verdict in zip((c for c, _ in askable), verdicts, strict=True):
        detail.append((claim, verdict))
        if verdict.label is None:
            agg.n_unjudged += 1
            continue
        if verdict.label == "NO_CLAIM":
            agg.n_not_a_claim += 1
            continue
        agg.add(verdict.label == "SUPPORTED", category_of[claim.query_id], claim.query_id)
    return agg, detail


def score_uncited_grounding(
    records: Sequence[AnswerRecord],
    chunks: Mapping[str, str],
    judge: Judge,
) -> tuple[Aggregate, list[SentenceClaim]]:
    """Những mệnh đề **không trích nguồn nào** có căn cứ trong ngữ cảnh không?

    ## ⭐⭐ Đây là điểm mù, và điểm mù phải thành một con số

    `faithfulness` chỉ chấm được mệnh đề **có** `[n]`. Trên lần chạy đầu đó là
    433/700 — tức **38% nội dung câu trả lời nằm ngoài mọi phép kiểm**, cả cơ
    chế `W4-09` lẫn judge. Báo `faithfulness = 0,95` mà không nói ra điều đó là
    báo một con số đúng về một phần ba câu trả lời rồi để người đọc tưởng nó nói
    về toàn bộ.

    Nên phần không trích được hỏi lại với **hợp mọi chunk đã truy hồi**: câu hỏi
    yếu hơn ("có suy ra được từ thứ gì ta đã lấy về không") nhưng đúng câu cần
    hỏi ở đây — mệnh đề không tự khai nguồn thì không có nguồn nào để đối chiếu
    riêng.

    Kết quả tách được hai chế độ hỏng hoàn toàn khác nhau:

    * có căn cứ nhưng quên trích ⇒ lỗi **hình thức**, sửa bằng prompt;
    * không có căn cứ ⇒ **bịa**, và nó đã lọt qua mọi hàng rào.
    """
    agg = Aggregate("uncited_grounding")
    category_of = {record.query_id: record.category for record in records}
    questions: list[JudgeQuestion] = []
    asked: list[SentenceClaim] = []
    for record in records:
        if record.route != "retrieve":
            continue
        texts = [
            chunks[str(source["chunk_id"])]
            for source in record.sources
            if str(source["chunk_id"]) in chunks
        ]
        for claim in claims_of(record):
            if claim.cited_ns:
                continue
            if not texts:
                agg.n_no_evidence += 1
                continue
            context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(texts, start=1))
            questions.append(
                JudgeQuestion(
                    prompt_id="judge-faithfulness",
                    labels=FAITHFULNESS_LABELS,
                    variables={"context": context, "claim": claim.text},
                    ref=claim.ref + "+uncited",
                )
            )
            asked.append(claim)

    groundless: list[SentenceClaim] = []
    for claim, verdict in zip(asked, judge.ask_many(questions), strict=True):
        if verdict.label is None:
            agg.n_unjudged += 1
            continue
        if verdict.label == "NO_CLAIM":
            agg.n_not_a_claim += 1
            continue
        grounded = verdict.label == "SUPPORTED"
        if not grounded:
            groundless.append(claim)
        agg.add(grounded, category_of[claim.query_id], claim.query_id)
    return agg, groundless


def score_relevancy(
    records: Sequence[AnswerRecord], judge: Judge
) -> tuple[Aggregate, dict[str, str | None]]:
    """`RELEVANT` / tổng. `REFUSAL` **không** vào tử số nhưng vẫn ở mẫu số.

    Lý do: với một câu trả lời được, từ chối là một lần trượt — người dùng hỏi và
    không nhận được gì. Với câu `unanswerable` thì ngược lại, và đó là việc của
    `W5-02` (`refusal correctness`), chấm trên **cùng những nhãn này**. Judge trả
    nhãn; quy ước cộng điểm nằm ở đây, nơi test được và đổi lại được.
    """
    agg = Aggregate("answer_relevancy")
    questions = [
        JudgeQuestion(
            prompt_id="judge-answer-relevancy",
            labels=RELEVANCY_LABELS,
            variables={"question": record.query, "answer": record.answer},
            ref=record.query_id,
        )
        for record in records
        if record.answer.strip()
    ]
    answered = [record for record in records if record.answer.strip()]
    agg.n_no_evidence = len(records) - len(answered)
    verdicts = judge.ask_many(questions)

    labels: dict[str, str | None] = {}
    for record, verdict in zip(answered, verdicts, strict=True):
        labels[record.query_id] = verdict.label
        if verdict.label is None:
            agg.n_unjudged += 1
            continue
        agg.add(verdict.label == "RELEVANT", record.category, record.query_id)
    return agg, labels


def score_misattribution(
    records: Sequence[AnswerRecord],
    chunks: Mapping[str, str],
    judge: Judge,
    detail: Sequence[tuple[SentenceClaim, JudgeVerdict]],
) -> tuple[Aggregate, list[SentenceClaim]]:
    """Gán nhầm nguồn: mệnh đề **hợp mọi chunk chống đỡ** nhưng **chunk nó trích
    thì không**.

    ## ⭐ Chỉ chấm lại đúng những mệnh đề đã trượt ở vòng trước

    Cám dỗ là chạy cả hai vòng trên toàn bộ 700 mệnh đề rồi lấy hiệu hai tỉ lệ.
    Làm vậy tốn gấp 2,6× (ngữ cảnh hợp có 5 chunk thay vì 1–2) để đo một thứ
    **định nghĩa** đã nói là chỉ tồn tại trong tập con: nếu chunk được trích đã
    chống đỡ mệnh đề thì không có gì để gán nhầm cả.

    Nên vòng này chỉ hỏi lại những mệnh đề mà vòng "theo nguồn trích" trả
    `CONTRADICTED`/`NOT_FOUND`. Mẫu số là **toàn bộ** mệnh đề đã chấm được, để
    con số vẫn là một tỉ lệ trên cùng quần thể.

    Trả kèm danh sách mệnh đề gán nhầm — báo cáo cần ví dụ thật, không cần một
    tỉ lệ trần trụi.
    """
    agg = Aggregate("misattribution")
    category_of = {record.query_id: record.category for record in records}
    by_query = {record.query_id: record for record in records}

    failed = [
        claim for claim, verdict in detail if verdict.label not in (None, "SUPPORTED", "NO_CLAIM")
    ]
    supported = [claim for claim, verdict in detail if verdict.label == "SUPPORTED"]
    for claim in supported:
        # Được chunk nó trích chống đỡ ⇒ theo định nghĩa không thể gán nhầm.
        agg.add(False, category_of[claim.query_id], claim.query_id)

    questions: list[JudgeQuestion] = []
    asked: list[SentenceClaim] = []
    for claim in failed:
        record = by_query[claim.query_id]
        texts = [
            chunks[str(source["chunk_id"])]
            for source in record.sources
            if str(source["chunk_id"]) in chunks
        ]
        if not texts:
            agg.n_no_evidence += 1
            continue
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(texts, start=1))
        questions.append(
            JudgeQuestion(
                prompt_id="judge-faithfulness",
                labels=FAITHFULNESS_LABELS,
                variables={"context": context, "claim": claim.text},
                ref=claim.ref + "+union",
            )
        )
        asked.append(claim)

    misattributed: list[SentenceClaim] = []
    for claim, verdict in zip(asked, judge.ask_many(questions), strict=True):
        if verdict.label is None:
            agg.n_unjudged += 1
            continue
        if verdict.label == "NO_CLAIM":
            agg.n_not_a_claim += 1
            continue
        hit = verdict.label == "SUPPORTED"
        if hit:
            misattributed.append(claim)
        agg.add(hit, category_of[claim.query_id], claim.query_id)
    return agg, misattributed


def score_refusal(
    records: Sequence[AnswerRecord], labels: Mapping[str, str | None]
) -> dict[str, Aggregate]:
    """Từ chối đúng chỗ. `W5-02`.

    Dùng lại **đúng những nhãn** mà `score_relevancy` đã lấy — không một lời gọi
    judge nào thêm. `REFUSAL` là một nhãn của rubric `judge-answer-relevancy` chứ
    không phải một phép dò từ khoá, vì "không đủ thông tin", "các nguồn không nêu
    rõ", "I could not find" và mười cách nói khác đều là cùng một hành vi, và một
    danh sách từ khoá sẽ bắt được đúng những cách nói tôi nghĩ ra được.

    ## ⭐ Ba con số, không phải một

    Một tỉ lệ duy nhất che mất đánh đổi trung tâm của hạng mục này. Hệ thống từ
    chối **mọi** câu đạt 100% trên nhóm `unanswerable` và vô dụng; hệ thống không
    bao giờ từ chối đạt 100% trên nhóm còn lại và bịa cho 33 câu không có đáp án.

    * `refusal_recall` — trong 33 câu `unanswerable`, bao nhiêu câu được từ chối.
    * `false_refusal_rate` — trong các câu **trả lời được**, bao nhiêu câu bị từ
      chối oan. Đây là cái giá, và nó phải nằm cạnh con số kia.
    * `refusal_accuracy` — quyết định từ chối/không từ chối đúng trên toàn tập.
      Đây là con số đối chiếu với ngưỡng của bảng mục tiêu.
    """
    recall = Aggregate("refusal_recall")
    false_rate = Aggregate("false_refusal_rate")
    accuracy = Aggregate("refusal_accuracy")
    for record in records:
        label = labels.get(record.query_id)
        if label is None:
            recall.n_unjudged += 1
            false_rate.n_unjudged += 1
            accuracy.n_unjudged += 1
            continue
        refused = label == "REFUSAL"
        should_refuse = record.category == "unanswerable"
        accuracy.add(refused == should_refuse, record.category, record.query_id)
        if should_refuse:
            recall.add(refused, record.category, record.query_id)
        else:
            false_rate.add(refused, record.category, record.query_id)
    return {agg.name: agg for agg in (recall, false_rate, accuracy)}


def citation_accuracy(validity: Aggregate, faith: Aggregate) -> dict[str, Any]:
    """Hai định nghĩa của "độ chính xác trích dẫn", và cả hai đều cần thiết.

    Bảng mục tiêu viết một dòng `Citation accuracy ≥ 0,85` mà không nói *cấp nào*,
    và hai cấp cho hai con số khác hẳn nhau:

    * **cấp trích dẫn** (`citation_validity`, tất định, cơ chế `W4-09`): quote có
      nằm **nguyên văn** trong đúng chunk nó chỉ vào không. Đây là câu hỏi mà một
      người dùng bấm vào citation đang hỏi.
    * **cấp mệnh đề** (`faithfulness`, judge): chunk được trích có **chống đỡ** ý
      của câu không. Một quote nguyên văn nhưng không liên quan vẫn qua được cấp
      trên và trượt ở cấp này.

    Trả cả hai và nói rõ cái nào đối chiếu với ngưỡng. Gộp chúng thành một con số
    trung bình là tạo ra một đại lượng không ai kiểm lại được.
    """
    return {
        "quote_level": {
            "value": validity.value,
            "n": validity.n,
            "note": "quote nguyên văn trong đúng chunk được chỉ (W4-09, tất định)",
        },
        "claim_level": {
            "value": faith.value,
            "n": faith.n,
            "note": "chunk được trích chống đỡ ý của câu (judge)",
        },
        "gate_metric": "quote_level",
    }


def derived(faith: Aggregate, uncited: Aggregate, coverage: Aggregate) -> dict[str, Any]:
    """Hai con số chỉ tính được **sau khi** judge đã tách `NO_CLAIM` ra.

    ## ⭐⭐ Mẫu số là toàn bộ câu chuyện

    `faithfulness` chấm mệnh đề **có trích nguồn**; `uncited_grounding` chấm phần
    còn lại. Công bố riêng cái thứ nhất là công bố một con số đúng về hai phần ba
    câu trả lời rồi để người đọc tưởng nó nói về toàn bộ. `overall_groundedness`
    gộp cả hai trên đúng một mẫu số: **mọi mệnh đề thật**.

    `citation_coverage` tất định (ở trên) lấy mẫu số là *mọi câu*, kể cả câu dẫn
    và lời từ chối — mà luật 2 của `chat-system` không đòi những câu ấy phải trích
    nguồn. `citation_coverage_on_claims` sửa mẫu số ấy, và chênh lệch giữa hai
    con số chính là tỉ trọng câu meta trong câu trả lời.
    """
    real_claims = faith.n + uncited.n
    if not real_claims:
        return {}
    grounded = (faith.value or 0.0) * faith.n + (uncited.value or 0.0) * uncited.n
    return {
        "overall_groundedness": {
            "value": round(grounded / real_claims, 4),
            "n": real_claims,
            "note": (
                "mọi mệnh đề THẬT: có trích nguồn (chấm với nguồn đã trích) + "
                "không trích nguồn (chấm với hợp mọi chunk). Loại NO_CLAIM."
            ),
        },
        "citation_coverage_on_claims": {
            "value": round(faith.n / real_claims, 4),
            "n": real_claims,
            "note": (
                f"mẫu số bỏ {faith.n_not_a_claim + uncited.n_not_a_claim} câu meta "
                f"(câu dẫn, đề mục, lời từ chối) mà luật 2 không đòi phải trích nguồn; "
                f"bản tất định ở trên lấy mẫu số {coverage.n} câu"
            ),
        },
        "not_a_claim_share": {
            "value": round((faith.n_not_a_claim + uncited.n_not_a_claim) / coverage.n, 4),
            "n": coverage.n,
            "note": "tỉ trọng câu meta trong câu trả lời",
        },
    }


def summarise(aggregates: Iterable[Aggregate]) -> dict[str, Any]:
    return {agg.name: agg.as_dict() for agg in aggregates}


def main(argv: Sequence[str] | None = None) -> int:
    """Chấm một answer run. Không chạm hệ thống — chỉ đọc file và gọi judge."""
    import argparse
    import json
    import logging
    import sys
    from pathlib import Path

    from rag_core.llm import GLM_BASE_URL

    from .answer_run import load_answer_run, load_chunk_sidecar
    from .golden import load_golden_set
    from .judge import DEEPSEEK_BASE_URL, DEFAULT_JUDGE_MODEL, JudgeConfig, build_judge

    parser = argparse.ArgumentParser(description="W5-01 — chấm tầng sinh")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cap-usd", type=float, default=1.0)
    parser.add_argument("--cache", type=Path, default=Path(".cache/judge.sqlite3"))
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--frozen-cache",
        action="store_true",
        help="Mọi lượt trượt cache là lỗi — chế độ tái lập lại con số đã báo cáo",
    )
    # ⭐ `W5-11`: judge phải chọn được model. Trước đây provider hardcode DeepSeek,
    # tức một bảng ablation có DeepSeek trong danh sách ứng viên sẽ do CHÍNH một
    # ứng viên chấm, và không có cờ nào để kiểm chéo. `TD-66` đã đo được đổi
    # model judge làm faithfulness dịch 7,5 điểm — lớn hơn mọi cải thiện của cả
    # `W2` cộng lại, tức đủ để đảo thứ hạng.
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="slug judge (`deepseek-v4-flash` | `glm-5.3-flash`). Họ suy ra từ slug.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="mặc định: endpoint của họ model. Nêu tường minh khi dùng gateway riêng.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run = load_answer_run(args.run)
    chunks = load_chunk_sidecar(args.run)
    if not chunks:
        parser.error(
            f"thiếu sidecar nội dung chunk cho {args.run}. Chạy `answer_run --only-chunks`."
        )
    golden = {query.query_id: query for query in load_golden_set(args.golden)}

    base_url = args.judge_base_url or (
        GLM_BASE_URL if args.judge_model.startswith("glm-") else DEEPSEEK_BASE_URL
    )
    judge = build_judge(
        JudgeConfig(
            model=args.judge_model,
            base_url=base_url,
            cache_path=args.cache,
            cap_usd=args.cap_usd,
            concurrency=args.concurrency,
            frozen_cache=args.frozen_cache,
        )
    )

    deterministic = [
        context_precision(run.records, golden, k=args.k),
        context_recall(run.records, golden, k=args.k),
        citation_coverage(run.records),
        citation_validity(run.records),
    ]
    faith, detail = score_faithfulness(run.records, chunks, judge)
    mis, misattributed = score_misattribution(run.records, chunks, judge, detail)
    uncited, groundless = score_uncited_grounding(run.records, chunks, judge)
    relevancy, labels = score_relevancy(run.records, judge)
    refusal = score_refusal(run.records, labels)

    report = {
        "run": run.name,
        "golden_sha256": run.golden_sha256,
        "bundle_versions": run.bundle_versions,
        "prompt_specs": run.prompt_specs,
        # ⭐ Model **thực tế đã sinh** ra những câu trả lời đang bị chấm. Thiếu nó
        # thì `build_bundle` phải nhận `evaluated_with_generator` gõ tay — và đó
        # đúng là cách hai bundle hiện có mang bí danh `deepseek-chat@2026-09`
        # trong chính trường sinh ra để bảo đảm danh tính ổn định (`W5-05`).
        "models": run.models,
        # ⭐ Độ trễ **end-to-end**, đo trên chính 242 request đã sinh ra các con
        # số ở trên. `W4-13` báo `5312 ms` từ một smoke test vài câu; đây là cùng
        # phép đo trên toàn golden set, qua HTTP, kể cả một lượt trúng cache.
        "latency_ms": _latency(run.records),
        "judge": {
            "model": judge.config.model,
            "family": judge.config.family,
            "reasoning": judge.config.reasoning,
            "rubrics": sorted(
                prompt.spec for prompt in judge.registry.all() if prompt.id.startswith("judge-")
            ),
            "cache_digest": judge.cache.digest(),
            **judge.stats.as_dict(),
        },
        "metrics": summarise([*deterministic, faith, mis, uncited, relevancy]),
        "derived": derived(faith, uncited, deterministic[2]),
        "refusal": {name: agg.as_dict() for name, agg in refusal.items()},
        "citation_accuracy": citation_accuracy(deterministic[3], faith),
        "relevancy_labels": _count(labels.values()),
        "faithfulness_labels": _count(v.label for _, v in detail),
        # Ví dụ thật, không phải chỉ một tỉ lệ: một con số faithfulness không có
        # ví dụ đi kèm thì không ai kiểm lại được nó.
        "examples": {
            "unfaithful": [
                {"ref": claim.ref, "claim": claim.text, "reason": verdict.reason}
                for claim, verdict in detail
                if verdict.label in ("CONTRADICTED", "NOT_FOUND")
            ][:12],
            "misattributed": [
                {"ref": claim.ref, "claim": claim.text, "cited": list(claim.cited_ns)}
                for claim in misattributed
            ][:12],
            "groundless_uncited": [{"ref": claim.ref, "claim": claim.text} for claim in groundless][
                :12
            ],
        },
    }
    target = args.out or args.run.with_name(f"{args.run.stem}-generation.json")
    Path(target).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ⭐ `W5-11`: điểm cấp truy vấn ra **file riêng**, không nhét vào report.
    # 242 truy vấn × 11 metric làm report chính phồng lên gấp nhiều lần và đẩy
    # phần người đọc cần xuống dưới — trong khi thứ duy nhất tiêu thụ nó là
    # `pipeline.eval.ablation_generation`, một máy đọc.
    paired = {
        agg.name: agg.paired_values()
        for agg in (*deterministic, faith, mis, uncited, relevancy, *refusal.values())
    }
    per_query_path = Path(target).with_name(f"{Path(target).stem}-per-query.json")
    per_query_path.write_text(
        json.dumps(
            {"run": run.name, "models": run.models, "judge_model": judge.config.model, **paired},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    headline = {"metrics": report["metrics"], "judge": report["judge"]}
    sys.stdout.write(json.dumps(headline, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.write(f"đã ghi {target}\n")
    return 0


def _latency(records: Sequence[AnswerRecord]) -> dict[str, float]:
    """p50/p95/p99 của `wall_ms`. Bao gồm cả lượt trúng cache — đó là độ trễ
    người dùng thật sự thấy, và loại chúng ra là tự cho mình một con số đẹp hơn
    production."""
    values = sorted(r.wall_ms for r in records if r.wall_ms)
    if not values:
        return {}

    def quantile(fraction: float) -> float:
        return values[min(len(values) - 1, int(fraction * len(values)))]

    return {
        "n": float(len(values)),
        "mean": round(statistics.fmean(values), 1),
        "p50": round(quantile(0.50), 1),
        "p95": round(quantile(0.95), 1),
        "p99": round(quantile(0.99), 1),
        "max": round(max(values), 1),
    }


def _count(labels: Iterable[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label in labels:
        key = label or "UNPARSEABLE"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
