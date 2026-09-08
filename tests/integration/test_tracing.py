"""Một lượt `/chat` thật sinh ra đủ span — `W5-06`.

## Vì sao là `TestClient` ở đây, khác `test_chat_stream.py`

`W4-06` không dùng được `TestClient` vì câu hỏi của nó là *"khung SSE có tới
theo dòng không"*, và `TestClient` đệm phản hồi nên nó không phân biệt được.

Câu hỏi ở đây khác hẳn: *"cây span có đủ nhánh không, và mỗi nhánh có mọc đúng
chỗ không"*. Đó là một tính chất của **cấu trúc dữ liệu trong tiến trình**, và
để đọc được nó thì phải ở cùng tiến trình. Một server con sẽ buộc cây span phải
đi qua một file trung gian — tức bài test đo cái file ấy, không đo cái cây.

Postgres là **thật**: `prepare()` mở hội thoại trong DB trước khi trả `ChatTurn`,
và `trace.session_id` chỉ có giá trị nếu nó là `conversation_id` thật.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_core.llm import ChatMessage, LLMChunk, LLMResponse
from rag_core.retrieval.filters import FilterSpec
from rag_core.schemas import Chunk, DocumentMetadata, RetrievalMode, RetrievedChunk, TokenUsage
from rag_core.settings import Settings
from serving.api.app import create_app
from serving.core.auth import digest_of
from serving.core.tracing import NONCE_MASK, Span, Trace
from tests.integration.chat_app import write_keys
from tests.integration.test_bundle_reload import write_bundle

pytestmark = pytest.mark.integration

KEY = "rag_acme_tracing_key"
ADMIN_KEY = "rag_acme_tracing_admin"
LANGFUSE_URL = os.environ.get("LANGFUSE_HOST", "http://127.0.0.1:3000")
LANGFUSE_AUTH = ("pk-lf-rag-platform-local", "sk-lf-rag-platform-local")


# ---------------------------------------------------------------------------
# Giả lập ĐÚNG hình dạng chuỗi truy hồi thật
# ---------------------------------------------------------------------------


def _hit(n: int, *, mode: RetrievalMode, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"c{n}",
            doc_id=f"d{n}",
            content=f"Nội dung chunk {n}. RRF là reciprocal rank fusion.",
            chunk_index=0,
            metadata=DocumentMetadata(
                source_url=f"https://example.test/{n}", license="CC-BY-4.0", title=f"Tài liệu {n}"
            ),
        ),
        score=score,
        rank=n,
        mode=mode,
        dense_score=score,
        sparse_score=score / 2,
    )


class FakeHybrid:
    """Đứng thay `QdrantHybridRetriever`.

    ⭐ Có thuộc tính `k`, vì `instrument_retriever` nhận diện lớp **bằng thuộc
    tính** chứ không bằng `isinstance`. Một fake không mang `k` sẽ nhận span tên
    `retrieve` thay vì `retrieve.hybrid`, và bài test sẽ đo một nhánh mã mà
    production không đi qua.
    """

    name = "fake-hybrid"
    k = 60

    def retrieve(
        self, query: str, top_k: int = 10, *, filters: FilterSpec = None
    ) -> list[RetrievedChunk]:
        return [_hit(n, mode=RetrievalMode.HYBRID, score=1.0 / n) for n in range(1, 6)]


class FakeReranker:
    name = "fake-reranker"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        # Đảo ngược thứ hạng: chunk cuối của nhánh nền thành chunk đầu. Cụ thể
        # có ý nghĩa — nó chứng minh span `rerank` mang một thứ tự KHÁC span
        # `retrieve.hybrid`, tức trace đọc ra được đóng góp của reranker.
        return [float(i) for i in range(len(texts))]


class FakeReranked:
    """Đứng thay `RerankedRetriever` — cùng hai thuộc tính công khai."""

    name = "reranked[fake-hybrid]:fake-reranker:n50"

    def __init__(self) -> None:
        self.base: Any = FakeHybrid()
        self.reranker: Any = FakeReranker()

    def retrieve(
        self, query: str, top_k: int = 10, *, filters: FilterSpec = None
    ) -> list[RetrievedChunk]:
        pool = self.base.retrieve(query, 50, filters=filters)
        scores = self.reranker.score(query, [hit.chunk.content for hit in pool])
        order = sorted(range(len(pool)), key=lambda i: -scores[i])
        return [
            RetrievedChunk(
                chunk=pool[i].chunk,
                score=scores[i],
                rank=rank,
                mode=RetrievalMode.RERANKED,
                dense_score=pool[i].dense_score,
                sparse_score=pool[i].sparse_score,
                rerank_score=scores[i],
            )
            for rank, i in enumerate(order[:top_k], start=1)
        ]


class FakeLLM:
    name = "fake"
    model = "fake-model"

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Any = None,
    ) -> Any:
        for piece in ("RRF ", "hợp nhất ", "thứ hạng."):
            yield LLMChunk(delta=piece)
        yield LLMChunk(
            final=LLMResponse(
                text="RRF hợp nhất thứ hạng.",
                model="fake-model-served",
                model_requested="fake-model",
                usage=TokenUsage(prompt_tokens=1613, completion_tokens=48, cost_usd=0.000412),
                finish_reason="stop",
            )
        )


class Recorder:
    """Sink thu tại chỗ. Cái cây là thứ được kiểm, không phải cái gói tin."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def submit(self, trace: Trace) -> None:
        self.traces.append(trace)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tracing_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tracing")
    write_bundle(root / "bundles", "0.2.0")
    write_keys(
        root / "api-keys.json",
        {
            digest_of(KEY): {
                "tenant_id": "acme",
                "key_id": "acme-trace",
                "scopes": [],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(ADMIN_KEY): {
                "tenant_id": "acme",
                "key_id": "acme-trace-admin",
                "scopes": ["admin"],
                "rate_limit_per_minute": 10_000,
            },
        },
    )
    return root


@pytest.fixture
def app(tracing_workspace: Path, database: Any) -> Iterator[tuple[TestClient, Recorder]]:
    settings = Settings(
        bundle_root=tracing_workspace / "bundles",
        bundle_version="0.2.0",
        api_keys_file=tracing_workspace / "api-keys.json",
        chat_cache=False,
        # Bộ đếm hạn mức cục bộ, bất kể Docker: xem chú thích ở `chat_app.make`.
        quota_shared=False,
        chat_rewrite=False,
    )
    api = create_app(
        settings=settings,
        build_runtime=_fake_runtime,
        probe_factory=lambda registry: _always_ready(),
    )
    recorder = Recorder()
    api.state.trace_sink = recorder
    with TestClient(api) as client:
        api.state.chat.llm = FakeLLM()
        api.state.chat.sink = recorder
        yield client, recorder


def _fake_runtime(bundle: Any) -> tuple[Any, Any]:
    """Chuỗi truy hồi giả mang **đúng hình dạng** chuỗi thật (`.base` +
    `.reranker`), nên `instrument_retriever` chạy đúng nhánh của production."""
    return FakeReranked(), None


def _always_ready() -> Any:
    from serving.core.probes import ReadinessProbes

    return ReadinessProbes(checks={})


def _ask(
    client: TestClient, message: str = "RRF là gì?", *, key: str = KEY, **body: Any
) -> httpx.Response:
    """Một lượt `/chat` đọc tới khung cuối.

    `key` mở ra vì `test_metrics_endpoint.py` dùng lại hàm này với kho khoá của
    nó — hai module dựng hai app riêng để bộ đếm Prometheus của bên này không
    đếm request của bên kia.
    """
    response = client.post(
        "/chat",
        json={"message": message, **body},
        headers={"Authorization": f"Bearer {key}"},
    )
    response.read()
    result: httpx.Response = response
    return result


def _names(trace: Trace) -> list[str]:
    return [span.name for span in trace.spans]


def _span(trace: Trace, name: str) -> Span:
    found = trace.find(name)
    assert found is not None, f"thiếu span {name!r}; có: {_names(trace)}"
    return found


# ---------------------------------------------------------------------------
# 1. DoD: một query hiện đủ span
# ---------------------------------------------------------------------------


class TestSpanCount:
    def test_one_query_produces_the_whole_chain(self, app: Any) -> None:
        """Đây là câu DoD, viết ra thành một phép đếm.

        Bộ span **đầy đủ**, không phải "ít nhất": một span mới xuất hiện mà
        không ai cập nhật bài test này là một span không ai quyết định thêm.
        """
        client, recorder = app
        assert _ask(client).status_code == 200
        assert len(recorder.traces) == 1
        assert sorted(_names(recorder.traces[0])) == [
            "citations",
            "completion",
            "prompt",
            "rerank",
            "retrieval",
            "retrieve.hybrid",
            "understand",
        ]

    def test_the_retrieval_layers_nest_the_way_the_code_does(self, app: Any) -> None:
        """`retrieval` ⊃ (`retrieve.hybrid`, `rerank`) — đúng hình dạng
        `RerankedRetriever.retrieve()`, không phải một danh sách phẳng."""
        client, recorder = app
        _ask(client)
        trace = recorder.traces[0]
        outer = _span(trace, "retrieval")
        assert {s.name for s in trace.children_of(outer)} == {"retrieve.hybrid", "rerank"}
        assert _span(trace, "understand").parent_id is None

    def test_the_retrieval_span_outlasts_both_of_its_children(self, app: Any) -> None:
        """Nếu con dài hơn cha thì span không đo cái nó tự nhận là đang đo."""
        client, recorder = app
        _ask(client)
        trace = recorder.traces[0]
        outer = _span(trace, "retrieval").duration_ms or 0.0
        for child in ("retrieve.hybrid", "rerank"):
            assert (_span(trace, child).duration_ms or 0.0) <= outer

    def test_every_span_is_closed(self, app: Any) -> None:
        client, recorder = app
        _ask(client)
        trace = recorder.traces[0]
        assert all(span.closed for span in trace.spans)
        assert "unclosed_spans" not in trace.metadata


# ---------------------------------------------------------------------------
# 2. Nội dung của span — điểm số, token, tiền
# ---------------------------------------------------------------------------


class TestContent:
    def test_the_retrieval_spans_carry_scores_not_just_ids(self, app: Any) -> None:
        client, recorder = app
        _ask(client)
        row = _span(recorder.traces[0], "retrieve.hybrid").output["hits"][0]
        assert {"chunk_id", "score", "dense_score", "sparse_score"} <= set(row)

    def test_rerank_reports_an_order_different_from_its_input(self, app: Any) -> None:
        """Đóng góp của reranker chỉ đọc được khi hai span mang hai thứ tự."""
        client, recorder = app
        _ask(client)
        trace = recorder.traces[0]
        before = [h["chunk_id"] for h in _span(trace, "retrieve.hybrid").output["hits"]]
        after = [h["chunk_id"] for h in _span(trace, "retrieval").output["hits"]]
        assert before != after

    def test_the_completion_span_carries_tokens_and_cost(self, app: Any) -> None:
        client, recorder = app
        _ask(client)
        usage = _span(recorder.traces[0], "completion").usage
        assert usage.prompt_tokens == 1613
        assert usage.completion_tokens == 48
        assert usage.cost_usd == pytest.approx(0.000412)

    def test_the_trace_totals_the_cost_of_its_steps(self, app: Any) -> None:
        client, recorder = app
        _ask(client)
        trace = recorder.traces[0]
        assert trace.total_cost_usd() == pytest.approx(0.000412)
        assert trace.unmeasured_cost_steps() == []

    def test_the_trace_is_keyed_to_the_conversation_and_the_tenant(self, app: Any) -> None:
        client, recorder = app
        response = _ask(client)
        trace = recorder.traces[0]
        assert trace.session_id == response.headers["X-Conversation-Id"]
        assert trace.user_id == "acme"
        assert trace.tags == ["tenant:acme"]

    def test_the_trace_carries_the_request_id_that_the_logs_carry(self, app: Any) -> None:
        """`X-Request-ID` của `W4-03` là chỗ duy nhất nối một dòng log JSON với
        một trace trong Langfuse."""
        client, recorder = app
        response = _ask(client)
        assert recorder.traces[0].metadata["request_id"] == response.headers["X-Request-ID"]

    def test_the_done_frame_now_declares_the_time_it_cannot_count(self, app: Any) -> None:
        """Nửa sau của `TD-55`. `total_ms` bắt đầu bấm giờ **sau** phần chậm
        nhất của chính nó; khung `done` phải nói ra khoảng bị bỏ, nếu không thì
        client đọc một SLA màu hồng và không có cách nào biết."""
        import json

        client, _ = app
        frames = [
            json.loads(line[6:])
            for line in _ask(client).text.splitlines()
            if line.startswith("data: ")
        ]
        done = frames[-1]
        assert done["prepare_ms"] is not None
        assert done["prepare_ms"] > 0

    def test_prepare_ms_is_recorded_because_the_done_frame_cannot_see_it(self, app: Any) -> None:
        """`TD-55`: khung `done` đếm từ `ChatTurn.started`, tức **sau** truy hồi.
        Trace bắt đầu ở handler nên nó đo được cả phần ấy."""
        client, recorder = app
        _ask(client)
        assert recorder.traces[0].metadata["prepare_ms"] > 0


# ---------------------------------------------------------------------------
# 3. Che dữ liệu trên đường thật
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_the_prompt_span_carries_the_prompt_but_not_the_nonce(self, app: Any) -> None:
        """`W4-12`: mã phiên là lớp duy nhất của hàng rào không phụ thuộc model,
        và DoD của hạng mục này là một **ảnh chụp** của trace."""
        client, recorder = app
        _ask(client)
        output = _span(recorder.traces[0], "prompt").output
        rendered = str(output)
        assert "NGỮ CẢNH" in rendered
        assert NONCE_MASK in rendered
        # `TD-74`: message ngữ cảnh phải đi vào dưới dạng `content_parts` —
        # từng khối một ngân sách cắt riêng. Ghim Ở ĐÂY (span thật, app thật)
        # vì unit test không thấy được việc callsite quay về chuỗi ghép.
        assert "content_parts" in output[-1], output[-1].keys()
        assert all(isinstance(p, str) for p in output[-1]["content_parts"])

    def test_no_span_anywhere_leaks_a_sixteen_hex_run(self, app: Any) -> None:
        import re

        client, recorder = app
        _ask(client)
        blob = str([(s.input, s.output, s.metadata) for s in recorder.traces[0].spans])
        assert not re.search(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])", blob)


# ---------------------------------------------------------------------------
# 4. Những lượt KHÔNG chạy trót lọt
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_a_404_still_produces_a_trace(self, app: Any) -> None:
        """Nếu trace ra đời cùng `ChatTurn` thì hệ quan sát phủ đúng những
        request đã chạy trót lọt — tức nó luôn báo rằng mọi thứ đều ổn."""
        client, recorder = app
        response = client.post(
            "/chat",
            json={"message": "xin chào", "conversation_id": "khongtontai"},
            headers={"Authorization": f"Bearer {KEY}"},
        )
        assert response.status_code == 404
        assert len(recorder.traces) == 1
        assert recorder.traces[0].level == "ERROR"
        assert recorder.traces[0].status_message is not None
        assert recorder.traces[0].status_message.startswith("404")

    def test_a_403_trace_does_not_name_the_tenant_that_was_refused(self, app: Any) -> None:
        """Cùng lý do lời của `HTTPException` không nhắc nó (`W4-04`)."""
        client, recorder = app
        response = client.post(
            "/chat",
            json={"message": "xin chào", "filters": {"tenant_id": "globex"}},
            headers={"Authorization": f"Bearer {KEY}"},
        )
        assert response.status_code == 403
        assert "globex" not in (recorder.traces[0].status_message or "")

    def test_a_refused_request_produces_exactly_one_trace_not_two(self, app: Any) -> None:
        """`finish()` được gọi ở `api/chat.py` và ở `finally` của `stream_turn`.
        Không idempotent thì mỗi lượt hỏng cho hai dòng, một trong đó rỗng."""
        client, recorder = app
        client.post(
            "/chat",
            json={"message": "xin chào", "conversation_id": "khongtontai"},
            headers={"Authorization": f"Bearer {KEY}"},
        )
        assert len(recorder.traces) == 1


# ---------------------------------------------------------------------------
# 5. Quan sát của quan sát
# ---------------------------------------------------------------------------


class TestTracingStatus:
    def test_admin_tracing_says_whether_traces_actually_leave(self, app: Any) -> None:
        """Hàng đợi đầy, khoá sai, host sai — cả ba trông y hệt nhau từ phía
        `/chat`. Không có endpoint này thì quan sát là thứ duy nhất trong hệ
        thống không quan sát được."""
        from serving.core.langfuse import LangfuseSink

        client, _ = app
        client.app.state.trace_sink = LangfuseSink(
            host="http://127.0.0.1:3000",
            public_key="pk",
            secret_key="sk",
            client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(207))),
        )
        body = client.get("/admin/tracing", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).json()
        assert body["enabled"] is True
        assert set(body) >= {"queued", "sent", "failed", "dropped"}

    def test_it_says_so_plainly_when_tracing_is_off(self, app: Any) -> None:
        client, _ = app
        client.app.state.trace_sink = None
        body = client.get("/admin/tracing", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).json()
        assert body == {
            "enabled": False,
            "reason": "chưa cấu hình LANGFUSE_* — xem build_sink()",
        }

    def test_it_is_behind_the_admin_scope_like_every_admin_route(self, app: Any) -> None:
        client, _ = app
        response = client.get("/admin/tracing", headers={"Authorization": f"Bearer {KEY}"})
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# 6. Đối chiếu với Langfuse THẬT
# ---------------------------------------------------------------------------


def _langfuse_up() -> bool:
    try:
        return httpx.get(f"{LANGFUSE_URL}/api/public/health", timeout=3.0).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _langfuse_up(), reason="cần `make up-langfuse`")
class TestAgainstRealLangfuse:
    def test_the_encoder_speaks_the_ingestion_protocol_the_server_accepts(self) -> None:
        """Bộ mã hoá viết tay đối chiếu với server thật.

        ⭐ Đây là bài test duy nhất ở đây cần hạ tầng, và nó cần vì phần còn lại
        của module này chỉ chứng minh **cây span đúng** — nó không chứng minh
        được rằng Langfuse *nhận* cái cây ấy. Một gói tin sai schema bị trả 207
        kèm một mảng `errors` mà không ai đọc, và bảng vẫn trống.
        """
        from serving.core.langfuse import LangfuseSink
        from serving.core.tracing import Usage

        sink = LangfuseSink(
            host=LANGFUSE_URL, public_key=LANGFUSE_AUTH[0], secret_key=LANGFUSE_AUTH[1]
        )
        trace = Trace(name="pytest-encoder", input="RRF là gì?", sink=sink)
        with trace.span("retrieval"), trace.span("rerank"):
            pass
        generation = trace.span("completion", kind="generation")
        generation.end(
            model="fake-model-served",
            usage=Usage(prompt_tokens=1613, completion_tokens=48, cost_usd=0.000412),
        )
        trace.finish(output="RRF hợp nhất thứ hạng.")
        sink.close()
        assert sink.status() == {
            "host": LANGFUSE_URL,
            "queued": 0,
            "sent": 1,
            # `W5-08` thêm `scored`: điểm feedback đi qua **cùng** hàng đợi, nên
            # nó phải có bộ đếm riêng — `sent` gộp cả hai thì "trace có tới nơi
            # không" và "điểm có tới nơi không" trở thành một câu hỏi.
            "scored": 0,
            "failed": 0,
            "dropped": 0,
        }
        assert _fetch(trace.id) is not None, "Langfuse nhận 207 nhưng không lưu gì"

    def test_the_stored_trace_carries_the_cost_we_reported(self) -> None:
        """`totalCost` đọc lại từ server, không đọc lại từ đối tượng của mình —
        một trường sai tên vẫn cho `sent=1` và một hoá đơn bằng 0 trên bảng."""
        stored = _fetch(_last_encoder_trace())
        assert stored is not None
        assert stored["totalCost"] == pytest.approx(0.000412)
        assert len(stored["observations"]) == 3


def _fetch(trace_id: str, *, attempts: int = 15) -> dict[str, Any] | None:
    """Langfuse v3 nạp qua worker nên ghi là **bất đồng bộ** — phải hỏi lại."""
    import time

    with httpx.Client(base_url=LANGFUSE_URL, auth=LANGFUSE_AUTH, timeout=20.0) as client:
        for _ in range(attempts):
            response = client.get(f"/api/public/traces/{trace_id}")
            if response.status_code == 200:
                body: dict[str, Any] = response.json()
                return body
            time.sleep(2.0)
    return None


def _last_encoder_trace() -> str:
    with httpx.Client(base_url=LANGFUSE_URL, auth=LANGFUSE_AUTH, timeout=20.0) as client:
        data = client.get("/api/public/traces", params={"name": "pytest-encoder"}).json()
    assert data["data"], "chưa có trace nào tên `pytest-encoder`"
    trace_id: str = data["data"][0]["id"]
    return trace_id
