"""App factory cho test SSE — chạy trong **tiến trình uvicorn thật**.

Không phải một file test. Nó tồn tại vì `TestClient` của Starlette **đệm** phản
hồi dạng dòng (`W4-03` đã ghim điều đó bằng
`test_the_test_client_cannot_prove_streaming_works`), nên câu DoD *"nhận ≥ 2
chunk SSE"* không kiểm được bằng nó: một cài đặt gom hết token rồi gửi một cục
cũng làm test ấy xanh.

Chạy trong tiến trình khác nghĩa là mọi thứ phải đi qua **biến môi trường** —
đó là cái giá, và nó rẻ hơn một test xanh không chứng minh gì.

Giả lập đúng hai thứ: Qdrant (retriever) và LLM. Postgres là **thật**, vì nửa
sau của DoD (*"chat history sống sót qua restart container"*) không có nghĩa gì
với một DB giả.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from rag_core.bundle import RagBundle
from rag_core.llm import (
    ChatMessage,
    DailyBudget,
    LLMChunk,
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMRouter,
    Route,
)
from rag_core.retrieval.filters import FilterSpec
from rag_core.schemas import Chunk, DocumentMetadata, RetrievedChunk, TokenUsage
from rag_core.settings import Settings
from serving.api.app import create_app
from serving.core.probes import ReadinessProbes
from serving.core.registry import BundleRegistry
from serving.core.understanding import QueryUnderstanding

ENV_BUNDLES = "CHAT_TEST_BUNDLES"
ENV_KEYS = "CHAT_TEST_KEYS"
ENV_DELTA_MS = "CHAT_TEST_DELTA_MS"
ENV_RETRIEVAL_MS = "CHAT_TEST_RETRIEVAL_MS"
ENV_MODE = "CHAT_TEST_MODE"
ENV_ROUTER = "CHAT_TEST_ROUTER"
"""`fallback` = nhánh chính chết trước token đầu · `midstream` = chết sau token thứ hai · `broke` = trần chi phí đã cạn."""
ENV_REWRITE = "CHAT_TEST_REWRITE"
"""Chuỗi mà bộ viết lại của `W4-07` trả về. Không đặt = không cấu hình bộ viết lại."""
ENV_RETRIEVAL_BOOM = "CHAT_TEST_RETRIEVAL_BOOM"
"""`1` = truy hồi ném lỗi hạ tầng mang chi tiết nội bộ — cho test `AU-03`."""

BOOM_MESSAGE = "ResponseHandlingException: qdrant:6333/collections/rag_noi_bo timed out"
"""Giả đúng hình dạng lỗi thật của client Qdrant: tên service, port, collection.
Test canh rằng KHÔNG mảnh nào của chuỗi này lọt ra body 503."""


@dataclass
class SlowRetriever:
    """Đứng thay Qdrant + BGE-M3 + cross-encoder.

    ⭐ `time.sleep` chứ không `asyncio.sleep`, và đó là toàn bộ điểm của nó:
    `retrieve()` thật là **đồng bộ** và tiêu CPU/GPU. Nếu `ChatService` gọi
    thẳng nó trong `async def` thì suốt khoảng đó vòng lặp sự kiện đứng im, và
    `/health` — thứ điều khiển việc container có bị khởi động lại hay không —
    không trả lời. Test `test_health_answers_while_a_chat_is_retrieving` đo đúng
    điều đó, và nó chỉ đo được nếu chỗ này chặn thật.
    """

    name: str = "fake-slow"
    seen_filters: list[Any] | None = None

    def retrieve(
        self, query: str, top_k: int = 10, *, filters: FilterSpec = None
    ) -> list[RetrievedChunk]:
        if os.environ.get(ENV_RETRIEVAL_BOOM) == "1":
            raise RuntimeError(BOOM_MESSAGE)
        time.sleep(float(os.environ.get(ENV_RETRIEVAL_MS, "0")) / 1000.0)
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=f"chunk-{n}",
                    doc_id=f"doc-{n}",
                    content=f"Đoạn {n} nói về {query}.",
                    chunk_index=n - 1,
                    metadata=DocumentMetadata(
                        source_url=f"https://example.test/doc-{n}",
                        license="CC-BY-4.0",
                        title=f"Tài liệu {n}",
                    ),
                ),
                score=1.0 / n,
                rank=n,
            )
            for n in range(1, min(top_k, 2) + 1)
        ]


class ScriptedLLM:
    """Sinh token theo nhịp, hoặc chết giữa chừng — theo `CHAT_TEST_MODE`."""

    name = "scripted"
    model = "scripted-model"

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Any = None,
    ) -> AsyncIterator[LLMChunk]:
        import asyncio

        delay = float(os.environ.get(ENV_DELTA_MS, "120")) / 1000.0
        mode = os.environ.get(ENV_MODE, "ok")
        pieces = ["Theo ", "tài liệu ", "[1], ", "câu trả lời ", "là vậy."]
        if mode == "echo_prompt":
            # ⭐ Cửa sổ duy nhất nhìn được vào prompt thật từ tiến trình test.
            # Nó chứng minh cùng lúc ba thứ của `W4-07`: chuỗi nào đã đi vào
            # truy hồi (nằm trong khối NGỮ CẢNH mà `SlowRetriever` chép lại),
            # model được cho xem câu hỏi **gốc**, và chỉ thị ngôn ngữ có ở cuối.
            pieces = [messages[-1].content]
        elif mode == "citations":
            # `W4-09`: một quote THẬT + một quote BỊA. Quote thật lấy từ chính
            # prompt — dòng ngay sau mốc mở của nguồn 1 — nên test không phải
            # đoán lại chuỗi mà `SlowRetriever` đã sinh. Marker cố ý cắt đôi
            # giữa hai delta: đường giữ-lại phải chạy qua tiến trình thật.
            #
            # ⚠️ `W4-12` đổi hình dạng khối từ `[1] nội dung` sang
            # `<<<NGUON 1 {nonce}>>>\nnội dung`, và hai test này đỏ ngay — đúng
            # điều mong muốn: app giả bám vào **hợp đồng prompt thật**, nên đổi
            # hợp đồng thì nó phải đỏ chứ không được lặng lẽ trích nhầm.
            prompt = messages[-1].content
            lines = prompt.splitlines()
            opener = next(i for i, ln in enumerate(lines) if ln.startswith("<<<NGUON 1 "))
            line = lines[opener + 1]
            block = json.dumps(
                [
                    {"n": 1, "quote": line.strip()},
                    {"n": 2, "quote": "một câu bịa hoàn toàn"},
                ],
                ensure_ascii=False,
            )
            pieces = ["Trả lời ", "[1] và [2].", "\nCITA", "TIONS: " + block]
        elif mode == "echo_history":
            # Đọc ngược lại đúng phần lịch sử mà `ChatService` đã đưa vào prompt.
            # Không có đường nào khác: LLM sống trong tiến trình con, nên thứ duy
            # nhất test nhìn thấy là dòng token nó trả về.
            # Chỉ **kích thước** và **mép đầu** của cửa sổ, không phải toàn bộ nội
            # dung: bản đầu nối cả lịch sử bằng "|", mà chính các câu trả lời echo
            # cũng chứa "|" — nên test tự tách ra 363 mảnh. Số đo phải là thứ
            # không lồng vào chính nó.
            history = messages[1:-1]
            pieces = [f"n={len(history)};first={history[0].content if history else ''}"]
        for i, piece in enumerate(pieces):
            if mode == "fail_mid" and i == 2:
                raise LLMError("provider ngắt kết nối giữa chừng")
            await asyncio.sleep(delay)
            yield LLMChunk(delta=piece)
        yield LLMChunk(
            final=LLMResponse(
                text="".join(pieces),
                model="scripted-model-served",
                model_requested="scripted-model",
                usage=TokenUsage(prompt_tokens=120, completion_tokens=11, cost_usd=0.000045),
                finish_reason="stop",
            )
        )


class ScriptedRewriter(LLMProvider):
    """Bộ viết lại câu hỏi của `W4-07`, chạy trong tiến trình con.

    Trả một chuỗi cố định đọc từ môi trường: test cần biết **chính xác** chuỗi
    nào lẽ ra phải đi vào truy hồi, và một bộ viết lại "thông minh" ở đây sẽ
    biến phép kiểm thành một phép đoán.
    """

    name = "scripted-rewriter"

    def __init__(self) -> None:
        self.model = "scripted-rewriter-model"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        extra_body: Any = None,
    ) -> LLMResponse:
        text = os.environ.get(ENV_REWRITE, "")
        return LLMResponse(
            text=text,
            model=self.model,
            model_requested=self.model,
            usage=TokenUsage(prompt_tokens=90, completion_tokens=10, cost_usd=0.00003),
            finish_reason="stop",
        )


class BrokenLLM:
    """Nhánh chính chết — trước hoặc sau mẩu đầu tiên, theo `CHAT_TEST_ROUTER`."""

    name = "broken"
    model = "broken-model"

    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Any = None,
    ) -> AsyncIterator[LLMChunk]:
        if self.fail_after is None:
            raise LLMError("nhánh chính trả HTTP 503")
        for i in range(self.fail_after):
            yield LLMChunk(delta=f"CHÍNH-{i} ")
        raise LLMError("nhánh chính đứt giữa chừng")

    def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise LLMError("nhánh chính trả HTTP 503")


ENV_CACHE = "CHAT_TEST_CACHE"


class _TopicEmbedder:
    """Embedder giả cho test cache: vector chỉ phụ thuộc TỪ CUỐI của câu.

    Hai câu cùng từ cuối → cosine 1,0 (mô phỏng paraphrase vượt ngưỡng), khác
    từ cuối → hai vector ngẫu nhiên độc lập 32 chiều (cosine lè tè dưới 0,5).
    Dùng sha256 chứ không `hash()`: server chạy ở TIẾN TRÌNH KHÁC với test, mà
    `hash()` của Python đổi theo tiến trình (PYTHONHASHSEED).
    """

    def embed_query(self, text: str) -> Any:
        import hashlib
        import re as _re

        import numpy as np

        words = _re.findall(r"\w+", text.lower())
        topic = words[-1] if words else ""
        seed = int.from_bytes(hashlib.sha256(topic.encode()).digest()[:4], "big")
        vector = np.random.default_rng(seed).normal(size=32)
        return (vector / np.linalg.norm(vector)).astype(np.float32)


class _TopicStore:
    embeddings = _TopicEmbedder()


def _build(bundle: RagBundle) -> tuple[Any, None]:
    retriever = SlowRetriever()
    if os.environ.get(ENV_CACHE) == "1":
        # `embedder_of` đào `retriever.store.embeddings` — gắn store giả là đủ
        # để đường cache thật chạy, không cần BGE-M3 trong test.
        retriever.store = _TopicStore()  # type: ignore[attr-defined]
    return retriever, None


def _probes(registry: BundleRegistry) -> ReadinessProbes:
    return ReadinessProbes(checks={}, ttl_s=0.0)


def make() -> FastAPI:
    settings = Settings(
        bundle_root=Path(os.environ[ENV_BUNDLES]),
        api_keys_file=Path(os.environ[ENV_KEYS]),
        log_level="WARNING",
        # `none` để `build_llm` không đọc `DEEPSEEK_API_KEY` thật: một test
        # không bao giờ được đi ra Internet, kể cả khi máy có key trong `.env`.
        chat_provider="none",
        # `W4-10`: cache chỉ bật khi test yêu cầu — Redis là trạng thái CHUNG
        # giữa các test, và một câu trả lời cache từ test này trả cho test kia
        # là loại đỏ giả khó truy nhất.
        chat_cache=os.environ.get(ENV_CACHE) == "1",
    )
    app = create_app(settings=settings, build_runtime=_build, probe_factory=_probes)
    app.state.chat.llm = ScriptedLLM()
    # `chat_provider="none"` làm `primary_generator` trả "" (⇒ cache tắt), nên
    # app test phải tự khai danh tính bộ sinh của mình. Phần sau dấu `:` phải
    # khớp `model_requested` của `ScriptedLLM` — lệch là cache tắt câm (`W5-11`).
    app.state.chat.generator = GENERATOR
    app.state.chat.endpoint = ENDPOINT
    mode = os.environ.get(ENV_ROUTER, "")
    if mode:
        # ⭐ Cùng `LLMRouter` của production, chỉ đổi nhà cung cấp giả — nên test
        # này đo đúng đoạn mã sẽ chạy thật, không đo một bản sao của nó.
        budget = DailyBudget(0.01)
        if mode == "broke":
            # Ngân sách hôm nay **đã** cạn từ trước — đúng ca thật, và là ca duy
            # nhất `prepare()` trả lời được bằng một HTTP status: ở thời điểm ấy
            # prompt chưa tồn tại nên không ước được giá của lời gọi sắp tới.
            budget.charge(0.02)
        primary = BrokenLLM(fail_after=2 if mode == "midstream" else None)
        app.state.chat.llm = LLMRouter(
            routes=[
                Route(provider=primary, label="primary"),  # type: ignore[arg-type]
                Route(provider=ScriptedLLM(), label="fallback"),  # type: ignore[arg-type]
            ],
            budget=budget,
        )
    if ENV_REWRITE in os.environ:
        app.state.chat.understanding = QueryUnderstanding(llm=ScriptedRewriter())
    return app


GENERATOR = "scripted:scripted-model"
"""Danh tính bộ sinh của app test — cùng giá trị mà bài kiểm namespace dựng lại."""

ENDPOINT = "scripted://in-process"
"""Máy chủ giả đã phục vụ `GENERATOR` (`W6-01`). Không rỗng có chủ đích: một
chuỗi rỗng sẽ để bài kiểm namespace xanh cả khi trục endpoint bị bỏ khỏi khoá."""


def write_keys(path: Path, keys: dict[str, dict[str, Any]]) -> None:
    path.write_text(json.dumps(keys), encoding="utf-8")
