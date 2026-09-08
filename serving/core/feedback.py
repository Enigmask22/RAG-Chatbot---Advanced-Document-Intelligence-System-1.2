"""Vòng phản hồi: 👍/👎 → Postgres → điểm Langfuse → ứng viên golden set.

`W5-08`. Đây là chỗ **duy nhất** trong hệ thống mà tín hiệu đi *ngược* chiều:
mọi thứ khác chảy từ corpus ra câu trả lời, còn cái này chảy từ người dùng
trở lại tập đo.

## ⭐⭐ Khoá nối không được đến từ người gọi

Điểm số Langfuse gắn theo `traceId`. Cách hiển nhiên là để client gửi kèm
`trace_id` — nó đã có sẵn trong khung `meta`. Nhưng `TD-73` đã ghi rằng tenant
trong Langfuse là một **nhãn**, không phải một **hàng rào**: bên ấy không kiểm
gì cả, nên một tenant gửi `trace_id` của tenant khác sẽ gắn điểm 👎 lên trace
của người khác, và hàng đợi review của họ nhận rác mà không có gì trong log nói
ra.

Nên endpoint **không nhận** `trace_id`. Nó nhận `message_id`, đọc hàng ấy qua
`atenant_session` (RLS lọc theo tenant của token), rồi lấy `trace_id` **từ
hàng đó**. Khoá nối trở thành một thứ người gọi phải *chứng minh sở hữu* thay
vì một thứ họ *khai*.

Cùng lý lẽ với `tenant_filter()` của `W4-04` và với policy RLS của `W4-05`, chỉ
khác chỗ: hai lần trước hàng rào chặn *đọc*, lần này nó chặn *ghi sang một hệ
thống khác*.

## ⭐ Hai kho, một luật idempotent

Một cú double-click là hai request. Postgres chặn bằng `UNIQUE (tenant_id,
message_id)` + upsert; Langfuse chặn bằng `score_id()` tất định. Nếu chỉ có một
trong hai thì đổi 👎 thành 👍 để lại một hàng đúng trong Postgres và hai điểm
mâu thuẫn trong Langfuse — và cái người ta nhìn là Langfuse.

## ⚠️ Điểm số là việc phụ, và nó được phép trượt

`submit_score()` xếp hàng, không gửi. Nếu Langfuse chết thì feedback **vẫn nằm
trong Postgres** và hàng đợi review vẫn chạy; phản hồi HTTP nói rõ điểm đã được
xếp hay chưa (`scored`) để người gọi không suy ra rằng im lặng nghĩa là xong.
Chiều ngược lại thì không chấp nhận được: mất hàng Postgres mà vẫn có điểm
Langfuse nghĩa là một tín hiệu người dùng chỉ còn sống trong một hệ quan sát có
TTL.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from rag_core.generation.guardrails import redact_pii
from serving.core.auth import Principal
from serving.core.langfuse import Score
from serving.db.engine import atenant_session
from serving.db.models import FEEDBACK_REASONS, Conversation, Feedback, Message

__all__ = [
    "FEEDBACK_REASONS",
    "SCORE_NAME",
    "FeedbackResult",
    "GoldenCandidate",
    "MessageNotFound",
    "NotAnAnswer",
    "export_candidates",
    "record_feedback",
    "review_queue",
    "write_candidates",
]

logger = logging.getLogger(__name__)

SCORE_NAME = "user_feedback"
"""Tên điểm trong Langfuse.

Một tên cố định chứ không phải `f"feedback_{reason}"`: tên là chiều mà Langfuse
gộp và vẽ biểu đồ, nên tách theo lý do sẽ cho bảy đường rời rạc mà không đường
nào là "tỉ lệ hài lòng". Lý do đi vào `comment`.
"""

_MAX_COMMENT = 2_000


class MessageNotFound(LookupError):
    """Không có message ấy — **hoặc** nó thuộc tenant khác (RLS không phân biệt)."""


class NotAnAnswer(ValueError):
    """Chấm một message của người dùng, hoặc của hệ thống."""


@dataclass(frozen=True)
class FeedbackResult:
    id: str
    message_id: str
    rating: int
    reason: str | None
    trace_id: str | None
    scored: bool
    """Điểm đã được **xếp hàng** sang Langfuse chưa. Không phải "đã tới nơi" —
    xem `/admin/tracing` cho câu đó."""

    created: bool
    """`False` = đã ghi đè một lần chấm trước đó của cùng tenant."""


async def record_feedback(
    sessions: async_sessionmaker[AsyncSession],
    principal: Principal,
    *,
    message_id: str,
    rating: int,
    reason: str | None = None,
    comment: str | None = None,
    sink: Any | None = None,
) -> FeedbackResult:
    """Ghi một lượt chấm, rồi xếp điểm sang Langfuse nếu có sink."""
    if rating not in (-1, 1):
        raise ValueError(f"rating phải là -1 hoặc 1, nhận {rating!r}")
    if reason is not None and reason not in FEEDBACK_REASONS:
        raise ValueError(f"reason không hợp lệ: {reason!r}")
    if comment is not None:
        # ⚠️ `NEW-08`/`AU-05`: redact TẠI NGUỒN, vì cột này chảy đi ba ngả —
        # review queue, điểm Langfuse, và file xuất ứng viên golden (thứ sẽ
        # sống trong git). Email/số điện thoại người dùng gõ vào ô góp ý không
        # được phép theo dòng ấy ra ngoài. Redact trước khi cắt: placeholder
        # dài hơn chuỗi gốc, cắt trước là có thể cắt đôi một placeholder.
        comment = redact_pii(comment).strip()[:_MAX_COMMENT] or None

    async with atenant_session(sessions, principal.tenant_id) as session:
        message = await session.scalar(select(Message).where(Message.id == message_id))
        if message is None:
            raise MessageNotFound(
                # `TD-78` đã đóng ca "chưa ghi xong": hàng trợ lý tồn tại từ
                # `_open_turn`, trước cả khung SSE đầu tiên. Còn lại là id sai,
                # tenant khác (RLS không phân biệt), hoặc hội thoại đã bị xoá.
                f"không có message {message_id!r}"
            )
        if message.role != "assistant":
            # ⭐ Chặn ở đây chứ không để nó thành một hàng hợp lệ: một feedback
            # gắn vào câu HỎI không nói được câu trả lời nào bị chấm — cùng một
            # câu hỏi hỏi lại lần hai cho hai câu trả lời khác nhau.
            raise NotAnAnswer(
                f"message {message_id!r} có role={message.role!r}; chỉ chấm được câu trả lời"
            )

        statement: Any = (
            pg_insert(Feedback)
            .values(
                tenant_id=principal.tenant_id,
                message_id=message_id,
                rating=rating,
                reason=reason,
                comment=comment,
            )
            .on_conflict_do_update(
                index_elements=["tenant_id", "message_id"],
                set_={
                    "rating": rating,
                    "reason": reason,
                    "comment": comment,
                    # ⚠️ Đẩy `created_at` lên: hàng đợi review sắp theo thời
                    # gian, và một lần đổi ý là một tín hiệu MỚI. Giữ mốc cũ thì
                    # người review không bao giờ thấy nó nổi lên.
                    "created_at": func.now(),
                },
            )
            # ⭐ `xmax = 0` là cách Postgres nói "hàng này vừa được INSERT, không
            # phải UPDATE" trong một `ON CONFLICT ... RETURNING`. Không có nó thì
            # phản hồi không phân biệt được "đã ghi nhận" với "đã ghi đè", và
            # người dùng bấm 👎 lần thứ hai không biết lần đầu có vào không.
            .returning(Feedback.id, literal_column("(xmax = 0)").label("inserted"))
        )
        row = (await session.execute(statement)).one()
        await session.commit()
        feedback_id, inserted = row[0], bool(row[1])

    trace_id = message.trace_id
    scored = False
    if sink is not None and trace_id:
        submit = getattr(sink, "submit_score", None)
        if callable(submit):
            submit(
                Score(
                    trace_id=trace_id,
                    name=SCORE_NAME,
                    value=float(rating),
                    comment=_score_comment(reason, comment),
                )
            )
            scored = True
    elif sink is not None and not trace_id:
        # Hàng ghi trước `0004` không có `trace_id`. Nói ra chứ không im lặng:
        # "không có điểm nào trong Langfuse" có hai nguyên nhân rất khác nhau.
        logger.info("feedback: message %s không có trace_id — không gắn điểm được", message_id)

    logger.info(
        "feedback %+d cho message %s (tenant %s, reason=%s, scored=%s)",
        rating,
        message_id,
        principal.tenant_id,
        reason,
        scored,
    )
    return FeedbackResult(
        id=feedback_id,
        message_id=message_id,
        rating=rating,
        reason=reason,
        trace_id=trace_id,
        scored=scored,
        created=inserted,
    )


def _score_comment(reason: str | None, comment: str | None) -> str | None:
    parts = [p for p in (reason, comment) if p]
    return " · ".join(parts) if parts else None


# ------------------------------------------------------------ hàng đợi review


@dataclass(frozen=True)
class ReviewItem:
    """Một lượt bị chấm, kèm đủ thứ để trả lời *tại sao* mà không mở DB lần nữa."""

    feedback_id: str
    created_at: datetime
    rating: int
    reason: str | None
    comment: str | None
    conversation_id: str
    message_id: str
    question: str | None
    rewritten_query: str | None
    answer: str
    model: str | None
    finish_reason: str | None
    route: str | None
    latency_ms: int | None
    trace_id: str | None
    bundle_version: str | None
    retrieved_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    citations_verified: int | None
    citations_claimed: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "created_at": self.created_at.isoformat(),
            "rating": self.rating,
            "reason": self.reason,
            "comment": self.comment,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "question": self.question,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "route": self.route,
            "latency_ms": self.latency_ms,
            "trace_id": self.trace_id,
            "bundle_version": self.bundle_version,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "cited_chunk_ids": self.cited_chunk_ids,
            "citations_verified": self.citations_verified,
            "citations_claimed": self.citations_claimed,
        }


async def review_queue(
    sessions: async_sessionmaker[AsyncSession],
    principal: Principal,
    *,
    rating: int | None = -1,
    reason: str | None = None,
    limit: int = 50,
) -> list[ReviewItem]:
    """Những lượt bị chấm, mới nhất trước.

    ⭐ Mặc định `rating=-1`. Hàng đợi review là một danh sách **việc phải làm**,
    và một lượt 👍 không phải việc phải làm — trộn chúng vào làm cho danh sách
    dài ra theo tỉ lệ hài lòng, tức càng tốt càng khó dùng. `rating=None` lấy
    tất cả, cho phép đếm mẫu số.

    ⚠️ RLS vẫn áp: một key `admin` thuộc về **một** tenant, nên đây là hàng đợi
    của tenant ấy, không phải của cả hệ thống. Đó là hành vi đúng — nhưng nó
    nghĩa là chưa có góc nhìn vận hành toàn cục nào, và không được nhầm cái
    này với cái đó.
    """
    async with atenant_session(sessions, principal.tenant_id) as session:
        query = (
            select(Feedback, Message, Conversation.bundle_version)
            .join(Message, Message.id == Feedback.message_id)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .order_by(Feedback.created_at.desc(), Feedback.id)
            .limit(limit)
        )
        if rating is not None:
            query = query.where(Feedback.rating == rating)
        if reason is not None:
            query = query.where(Feedback.reason == reason)
        rows = (await session.execute(query)).all()
        questions = await _questions_for(session, [row[1] for row in rows])

    return [
        _to_item(feedback, answer, bundle_version, questions.get(answer.id))
        for feedback, answer, bundle_version in rows
    ]


async def _questions_for(session: AsyncSession, answers: Sequence[Message]) -> dict[str, Message]:
    """Câu hỏi mà mỗi câu trả lời trả lời, ghép trong Python.

    ⭐ Hai truy vấn thay vì một `LATERAL` cho mỗi hàng: hàng đợi review có trần
    50 hàng, và một `LATERAL` ở đây là thứ chạy đúng cho tới ngày ai đó bỏ trần.

    ## `NEW-08`/`AU-07`: join theo `user_message_id`, không đoán theo đồng hồ

    Bản đầu chọn user message **muộn nhất** có `created_at <=` answer — đúng
    khi các lượt tuần tự, sai khi hai lượt cùng hội thoại chồng nhau (hàng user
    ghi ngay ở `_open_turn`, hàng assistant ghi trong task nền sau khi stream
    xong): user_B chen vào giữa là ứng viên golden mang câu hỏi B dán lên câu
    trả lời của A. Từ `0005`, hàng assistant mang khoá thật; suy luận thời
    gian chỉ còn là **fallback cho hàng ghi trước `0005`** — với đúng các hàng
    ấy, rủi ro cũ là thuộc tính của dữ liệu, không xoá được bằng code mới.
    """
    if not answers:
        return {}
    by_key = [a for a in answers if a.user_message_id is not None]
    legacy = [a for a in answers if a.user_message_id is None]

    paired: dict[str, Message] = {}
    if by_key:
        wanted = {a.user_message_id for a in by_key}
        rows = (await session.scalars(select(Message).where(Message.id.in_(wanted)))).all()
        questions_by_id = {row.id: row for row in rows}
        for answer in by_key:
            question = questions_by_id.get(answer.user_message_id or "")
            if question is not None:
                paired[answer.id] = question

    if legacy:
        conversation_ids = {a.conversation_id for a in legacy}
        rows = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id.in_(conversation_ids), Message.role == "user")
                .order_by(Message.created_at, Message.id)
            )
        ).all()
        by_conversation: dict[str, list[Message]] = {}
        for row in rows:
            by_conversation.setdefault(row.conversation_id, []).append(row)
        for answer in legacy:
            earlier = [
                m
                for m in by_conversation.get(answer.conversation_id, [])
                if m.created_at <= answer.created_at
            ]
            if earlier:
                paired[answer.id] = earlier[-1]
    return paired


def _to_item(
    feedback: Feedback,
    answer: Message,
    bundle_version: str | None,
    question: Message | None,
) -> ReviewItem:
    frame = answer.citations_verified or {}
    cited = frame.get("citations") if isinstance(frame, dict) else None
    cited_ids = (
        [c["chunk_id"] for c in cited if isinstance(c, dict) and c.get("chunk_id")]
        if isinstance(cited, list)
        else []
    )
    verified = frame.get("verified") if isinstance(frame, dict) else None
    return ReviewItem(
        feedback_id=feedback.id,
        created_at=feedback.created_at,
        rating=feedback.rating,
        reason=feedback.reason,
        comment=feedback.comment,
        conversation_id=answer.conversation_id,
        message_id=answer.id,
        question=question.content if question is not None else None,
        rewritten_query=question.rewritten_query if question is not None else None,
        answer=answer.content,
        model=answer.model,
        finish_reason=answer.finish_reason,
        route=question.route if question is not None else None,
        latency_ms=answer.latency_ms,
        trace_id=answer.trace_id,
        bundle_version=bundle_version,
        retrieved_chunk_ids=[
            str(s["chunk_id"])
            for s in (answer.retrieved_sources or [])
            if isinstance(s, dict) and s.get("chunk_id")
        ],
        cited_chunk_ids=[str(c) for c in cited_ids],
        citations_verified=verified if isinstance(verified, int) else None,
        citations_claimed=len(cited_ids) if isinstance(cited, list) else None,
    )


# ------------------------------------------------------- ứng viên golden set


class GoldenCandidate(BaseModel):
    """Một câu 👎 đã đóng gói để **người** xem xét — chưa phải một `GoldenQuery`.

    ## ⭐⭐ Ứng viên KHÔNG được mang hình dạng của một câu golden

    Cám dỗ là xuất thẳng ra `GoldenQuery`: điền `relevant_chunk_ids` bằng những
    chunk hệ thống đã truy hồi, `reference_answer` bằng câu hệ thống đã trả lời,
    rồi `category` đoán từ độ dài. File chạy được ngay, và nó **phá hỏng chính
    thứ nó bổ sung vào**.

    Lý do: hàng này tồn tại *bởi vì* hệ thống đã sai. Lấy đầu ra của hệ thống
    làm nhãn nghĩa là chấm hệ thống bằng chính lỗi của nó — `ndcg@10` sẽ tăng
    khi truy hồi giữ nguyên hành vi sai, và giảm khi nó được sửa. Một bộ eval có
    tính chất ấy tệ hơn không có bộ eval nào, vì nó vẫn ra số.

    Nên: **không có** trường nào tên `relevant_*`, không `reference_answer`,
    không `category`. Cái hệ thống đã làm nằm ở `retrieved_chunk_ids` /
    `system_answer` và tên của chúng nói rõ đó là *hành vi cần xét*, không phải
    *nhãn đúng*. Có test khoá điều này bằng cách đòi
    `GoldenQuery.model_validate(candidate)` phải **thất bại**.

    Ba trường một người review phải điền để nó thành `GoldenQuery`:
    `category`, `relevant_spans`, `reference_answer`. Không tự động hoá được cái
    nào — đó chính là công việc.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    query: str
    """Chuỗi người dùng **thật sự gõ** (`Message.content` của hàng user)."""

    rewritten_query: str | None = None
    system_answer: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    """Hệ thống đã đưa gì cho model. Bằng chứng cho người review, **không** phải nhãn."""

    cited_chunk_ids: list[str] = Field(default_factory=list)
    citations_verified: int | None = None
    citations_claimed: int | None = None
    rating: int
    reason: str | None = None
    comment: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    route: str | None = None
    bundle_version: str | None = None
    trace_id: str | None = None
    conversation_id: str
    message_id: str
    created_at: str
    reviewed_by_human: bool = False
    """Luôn `False` khi xuất ra. Có mặt để bước promote không phải nhớ đặt nó."""


def to_candidate(item: ReviewItem) -> GoldenCandidate:
    return GoldenCandidate(
        candidate_id=f"fb-{item.feedback_id[:12]}",
        query=item.question or item.rewritten_query or "",
        rewritten_query=item.rewritten_query,
        system_answer=item.answer,
        retrieved_chunk_ids=item.retrieved_chunk_ids,
        cited_chunk_ids=item.cited_chunk_ids,
        citations_verified=item.citations_verified,
        citations_claimed=item.citations_claimed,
        rating=item.rating,
        reason=item.reason,
        comment=item.comment,
        model=item.model,
        finish_reason=item.finish_reason,
        route=item.route,
        bundle_version=item.bundle_version,
        trace_id=item.trace_id,
        conversation_id=item.conversation_id,
        message_id=item.message_id,
        created_at=item.created_at.isoformat(),
    )


async def export_candidates(
    sessions: async_sessionmaker[AsyncSession],
    principal: Principal,
    *,
    rating: int | None = -1,
    limit: int = 500,
) -> list[GoldenCandidate]:
    items = await review_queue(sessions, principal, rating=rating, limit=limit)
    # ⚠️ Bỏ hàng không có câu hỏi: một ứng viên không có `query` thì không review
    # được, và nó sẽ nằm trong file như một hàng "sẽ sửa sau" vĩnh viễn.
    return [to_candidate(item) for item in items if item.question]


def write_candidates(candidates: Iterable[GoldenCandidate], path: Path) -> int:
    """JSONL, một ứng viên một dòng — cùng định dạng với `data/golden/*.jsonl`.

    Cùng **định dạng**, khác **schema**: một người mở hai file cạnh nhau thấy
    ngay cái nào đã có nhãn, vì file này không có cột nhãn nào.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


# ------------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI mỏng
    import argparse
    import asyncio

    from serving.core.auth import Principal as _Principal
    from serving.db.engine import async_session_factory, make_async_engine

    parser = argparse.ArgumentParser(
        prog="python -m serving.core.feedback",
        description="Xuất câu 👎 thành file ứng viên golden set.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    export = sub.add_parser("export", help="Ghi JSONL ứng viên.")
    export.add_argument("--tenant", required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--limit", type=int, default=500)
    export.add_argument(
        "--rating",
        type=int,
        default=-1,
        choices=(-1, 1, 0),
        help="-1 = 👎 (mặc định), 1 = 👍, 0 = tất cả",
    )
    args = parser.parse_args(argv)

    async def run() -> int:
        engine = make_async_engine()
        try:
            sessions = async_session_factory(engine)
            principal = _Principal(
                key_id="cli", tenant_id=args.tenant, scopes=frozenset(), rate_limit_per_minute=0
            )
            candidates = await export_candidates(
                sessions,
                principal,
                rating=None if args.rating == 0 else args.rating,
                limit=args.limit,
            )
            written = write_candidates(candidates, args.out)
            print(f"đã ghi {written} ứng viên → {args.out}")
            if written:
                print(
                    "\nĐây CHƯA phải golden query: thiếu `category`, `relevant_spans`,\n"
                    "`reference_answer` — ba thứ chỉ người điền được. Xem docstring\n"
                    "`GoldenCandidate` để biết vì sao không tự sinh chúng."
                )
            return 0
        finally:
            await engine.dispose()

    # ⚠️ Cùng lý do với `serving/__main__.py`: driver async của psycopg không
    # chạy trên `ProactorEventLoop`, và `asyncio.run` **có** tôn trọng
    # `loop_factory` (uvicorn thì không — đó là chỗ `W4-06` mất một buổi).
    from serving.__main__ import needs_selector_loop

    if needs_selector_loop():
        return asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
