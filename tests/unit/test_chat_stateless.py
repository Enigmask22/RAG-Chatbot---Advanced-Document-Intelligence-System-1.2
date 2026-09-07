"""`ChatService` chạy **không trạng thái** (`sessions=None`). `W6-02`.

## ⭐⭐ Vì sao bộ test này tồn tại

Trường `sessions` khai kiểu `async_sessionmaker | None` từ `W4-06`, còn dòng
thứ hai của `_prepare` thì `raise GenerationUnavailable("chưa cấu hình
Postgres")`. Kiểu hứa một chế độ mà mã từ chối phục vụ — và ai đọc chữ ký cũng
dựng được đúng cấu hình ấy, rồi biết mình sai ở *runtime*. Đó là chuyện đã xảy
ra khi lắp HF Space.

Đáng chú ý hơn: gỡ dòng chặn ấy đi thì **2.453 bài test vẫn xanh**. Không bài
nào canh nó, theo cả hai chiều. Nên chế độ này cần bộ test của riêng nó, và
bộ test phải nói được điều mà một `assert not raises` không nói: *không lưu
gì* phải khác *lặng lẽ giả vờ có lưu*.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from rag_core.llm import ChatMessage, LLMChunk, LLMResponse
from rag_core.schemas import Chunk, DocumentMetadata, RetrievalMode, RetrievedChunk, TokenUsage
from serving.core.auth import Principal
from serving.core.chat import ChatService, ConversationNotFound

PRINCIPAL = Principal(tenant_id="public", key_id="space-demo", scopes=frozenset())
META = DocumentMetadata(source_url="https://example.org/x", license="CC BY 3.0 IGO")


class _Retriever:
    name = "gia-lap"

    def retrieve(self, query: str, top_k: int = 10, **kw: Any) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="c1",
                    doc_id="d1",
                    content="Tăng trưởng GDP đạt 5,3%.",
                    chunk_index=0,
                    metadata=META,
                ),
                score=0.9,
                rank=1,
                mode=RetrievalMode.HYBRID,
            )
        ]


class _Bundle:
    bundle_version = "0.2.1"

    class components:
        class prompt:
            id = "chat-system"
            version = 2


class _Active:
    version = "0.2.1"
    bundle = _Bundle()
    retriever = _Retriever()
    reranker = None


class _Registry:
    active = _Active()


class _LLM:
    """Nhánh sinh tối thiểu. Cùng hình dạng `FakeLLM` của `test_chat_service.py`
    nhưng **không** dùng lại nó: bộ test kia đi kèm `CapturingService`, thứ
    override `_save` — tức nó cố ý không chạm đúng đoạn mã mà bài test này đang
    hỏi về."""

    name = "gia-lap"
    model = "gia-lap-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Any = None,
    ) -> AsyncIterator[LLMChunk]:
        self.calls += 1
        text = "Xong. [1] “Tăng trưởng GDP đạt 5,3%.”"
        yield LLMChunk(delta=text)
        yield LLMChunk(
            final=LLMResponse(
                text=text,
                model="gia-lap-v1",
                model_requested="gia-lap-v1",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2, cost_usd=0.0001),
                finish_reason="stop",
            )
        )


def _service(**kw: Any) -> ChatService:
    opts: dict[str, Any] = {"registry": _Registry(), "sessions": None, "llm": _LLM()}
    opts.update(kw)
    return ChatService(**opts)


class TestKhongPostgresVanChayDuocMotLuot:
    @pytest.mark.asyncio
    async def test_prepare_khong_con_tu_choi_vi_thieu_Postgres(self) -> None:
        turn = await _service().prepare(PRINCIPAL, question="GDP tăng bao nhiêu?")
        assert turn.contexts, "phải truy hồi được như thường"

    @pytest.mark.asyncio
    async def test_van_phat_ra_id_hop_le_du_khong_co_hang_nao_dang_sau(self) -> None:
        """Khung `meta` là hợp đồng với client, không phải hệ quả của Postgres."""
        turn = await _service().prepare(PRINCIPAL, question="GDP tăng bao nhiêu?")
        for value in (turn.conversation_id, turn.user_message_id, turn.answer_message_id):
            assert uuid.UUID(str(value))

    @pytest.mark.asyncio
    async def test_hai_luot_lien_tiep_cho_hai_hoi_thoai_KHAC_nhau(self) -> None:
        """Không trạng thái nghĩa là không có mạch — và điều đó phải nhìn thấy."""
        service = _service()
        a = await service.prepare(PRINCIPAL, question="câu một")
        b = await service.prepare(PRINCIPAL, question="câu hai")
        assert a.conversation_id != b.conversation_id


class TestTruyenConversationIdVaoMotServiceKhongTrangThai:
    @pytest.mark.asyncio
    async def test_la_ConversationNotFound_chu_KHONG_phai_im_lang_bo_qua(self) -> None:
        """⭐ Im lặng bỏ qua để client tin nó có mạch hội thoại trong khi mỗi
        lượt là độc lập — đúng kiểu hỏng câm mà dự án này từ chối ở mọi tầng."""
        with pytest.raises(ConversationNotFound):
            await _service().prepare(
                PRINCIPAL, question="tiếp đi", conversation_id=str(uuid.uuid4())
            )


class TestKhongLuuGiCaVaKhongVoLucCoGangLuu:
    @pytest.mark.asyncio
    async def test_stream_chay_tron_ven(self) -> None:
        service = _service()
        turn = await service.prepare(PRINCIPAL, question="GDP tăng bao nhiêu?")
        events = [event async for event in service.stream_turn(turn)]
        assert [e.event for e in events][-1] == "done"
        assert any(e.event == "delta" for e in events)

    @pytest.mark.asyncio
    async def test_task_ghi_nen_KHONG_ghi_mot_dong_loi_nao(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """⭐⭐ Bài này thay một bài không thể đỏ.

        `_save` chạy ở một task nền và nuốt mọi exception vào
        `logger.exception` — đúng như nó nên làm, vì task ấy không có ai bắt.
        Hệ quả cho việc kiểm: một `_save` **quên** kiểm `sessions is None` sẽ
        gọi `atenant_session(None, …)`, nổ, và **không đổi một khung SSE nào**.
        Phép tiêm `M19` sống sót đúng vì thế.

        Nên chỗ duy nhất nhìn thấy được là log. "Không trạng thái" phải nghĩa
        là *không thử ghi*, không phải *thử rồi thất bại êm*.
        """
        import asyncio
        import logging

        service = _service()
        turn = await service.prepare(PRINCIPAL, question="GDP tăng bao nhiêu?")
        with caplog.at_level(logging.ERROR, logger="serving.core.chat"):
            async for _ in service.stream_turn(turn):
                pass
            # Nhường vòng lặp hai nhịp để task nền chạy xong và kịp ghi log.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], [
            r.getMessage() for r in caplog.records
        ]

    @pytest.mark.asyncio
    async def test_nhanh_sinh_van_duoc_goi_dung_mot_lan(self) -> None:
        service = _service()
        turn = await service.prepare(PRINCIPAL, question="GDP tăng bao nhiêu?")
        async for _ in service.stream_turn(turn):
            pass
        assert service.llm.calls == 1  # type: ignore[union-attr]
