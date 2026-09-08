"""Vòng phản hồi end-to-end — `W5-08`. Postgres thật, RLS thật, app thật.

Ba thứ ở đây không giả lập được, và cả ba là lý do module này tồn tại:

* **RLS.** "Tenant khác không chấm được câu trả lời của tôi" là hành vi của
  policy Postgres, không của mã Python — một mock sẽ xanh với một policy bị
  gỡ bỏ.
* **Upsert.** `ON CONFLICT (tenant_id, message_id)` cần chỉ mục duy nhất **thật**
  của `0004`; thiếu nó thì câu lệnh không lỗi, nó chỉ thôi là upsert.
* **Cầu nối `answer_message_id`.** Id phát ra trong khung `meta` phải là đúng id
  mà task nền ghi xuống — hai chỗ trong hai luồng khác nhau, nối bằng một
  dataclass.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_core.settings import Settings
from serving.api.app import create_app
from serving.core.auth import digest_of
from serving.core.feedback import SCORE_NAME
from serving.core.langfuse import Score, score_id
from tests.integration.chat_app import write_keys
from tests.integration.test_bundle_reload import write_bundle
from tests.integration.test_tracing import FakeLLM, _always_ready, _fake_runtime

pytestmark = pytest.mark.integration

KEY = "rag_acme_feedback_key"
ADMIN_KEY = "rag_acme_feedback_admin"
OTHER_KEY = "rag_globex_feedback_key"


class _CountingSink:
    """Đứng thay `LangfuseSink`: đếm điểm thay vì gửi chúng đi."""

    def __init__(self) -> None:
        self.traces: list[Any] = []
        self.scores: list[Score] = []

    def submit(self, trace: Any) -> None:
        self.traces.append(trace)

    def submit_score(self, score: Score) -> None:
        self.scores.append(score)

    def status(self) -> dict[str, Any]:
        return {"host": "fake", "queued": 0, "sent": len(self.traces), "scored": len(self.scores)}


@pytest.fixture(scope="module")
def feedback_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("feedback")
    write_bundle(root / "bundles", "0.2.0")
    write_keys(
        root / "api-keys.json",
        {
            digest_of(KEY): {
                "tenant_id": "acme",
                "key_id": "acme-fb",
                "scopes": [],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(ADMIN_KEY): {
                "tenant_id": "acme",
                "key_id": "acme-fb-admin",
                "scopes": ["admin"],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(OTHER_KEY): {
                "tenant_id": "globex",
                "key_id": "globex-fb",
                "scopes": ["admin"],
                "rate_limit_per_minute": 10_000,
            },
        },
    )
    return root


@pytest.fixture(autouse=True)
def _empty_tables(database: Any) -> None:
    """Mỗi bài bắt đầu từ ba bảng rỗng.

    ⚠️ `database` dọn ở **đầu và cuối module**, nên không có nó thì mọi phép
    đếm ở đây ("hàng đợi có đúng 1 mục") đo tổng của các bài chạy trước — và nó
    xanh khi chạy một mình, đỏ khi chạy cả module. Kiểu phụ thuộc thứ tự ấy đắt
    nhất lúc gỡ, vì bài đỏ không phải bài sai.
    """
    from sqlalchemy import text

    with database.begin() as conn:
        for table in ("feedback", "message", "conversation"):
            conn.execute(text(f"DELETE FROM {table}"))


@pytest.fixture
def app(feedback_workspace: Path, database: Any) -> Iterator[tuple[TestClient, _CountingSink]]:
    settings = Settings(
        bundle_root=feedback_workspace / "bundles",
        bundle_version="0.2.0",
        api_keys_file=feedback_workspace / "api-keys.json",
        chat_cache=False,
        chat_rewrite=False,
    )
    api = create_app(
        settings=settings,
        build_runtime=_fake_runtime,
        probe_factory=lambda registry: _always_ready(),
    )
    sink = _CountingSink()
    api.state.trace_sink = sink
    with TestClient(api) as client:
        api.state.chat.llm = FakeLLM()
        api.state.chat.sink = sink
        yield client, sink


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------


def _frames(response: httpx.Response) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    name = ""
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: "):
            out.append((name, json.loads(line[len("data: ") :])))
    return out


def _turn(client: TestClient, message: str = "RRF là gì?", *, key: str = KEY) -> dict[str, Any]:
    """Một lượt `/chat` trọn vẹn; trả về khung `meta`.

    ⚠️ `_drain_saves` phải chạy sau: hàng trợ lý được ghi trong một task nền,
    nên id trong khung `meta` trỏ vào một hàng chưa tồn tại cho tới lúc ấy. Đó
    không phải một chi tiết của test — nó là `TD-78` nhìn từ phía client.
    """
    response = client.post(
        "/chat", json={"message": message}, headers={"Authorization": f"Bearer {key}"}
    )
    response.read()
    assert response.status_code == 200, response.text
    frames = dict(_frames(response))
    _drain_saves()
    return frames["meta"]


#: Hạn chờ task ghi Postgres, tính bằng **giây thật**. Đo cục bộ: 9,0–13,4 ms.
#: Để rộng gấp ~750× vì con số này chỉ tốn thời gian khi hệ thật sự hỏng — một
#: `INSERT` + `COMMIT` mất hơn 10 s là một lỗi đáng đỏ, không phải một runner bận.
DRAIN_TIMEOUT_S = 10.0


def _drain_saves() -> None:
    """Đợi task ghi Postgres chạy xong, với hạn chờ tính bằng **giây**.

    Không có `await` nào ở đây bắt được nó: `_schedule_save` cố ý là đồng bộ
    (xem §"Ngắt kết nối" của `serving/core/chat.py`).

    ⭐⭐ Bản đầu lặp **50 lần một request `/health`**, kèm lời giải thích rằng
    đó là *"cách ép vòng lặp sự kiện quay thêm vài vòng"*. Hai bài trong module
    này đỏ trên CI vì nó, và cả hai vế của câu ấy đều sai:

    * **Không cần ép.** `TestClient` chạy vòng lặp trong một luồng portal riêng
      chứ không trong luồng test, nên task nền tiến triển dù ở đây chỉ `sleep`.
      Thay `/health` bằng `sleep(1 ms)` thuần: 24/24 vẫn xanh, và số vòng cần
      còn **giảm** (7–9 → 5–8). Những request ấy chưa bao giờ làm việc chúng
      được ghi là làm.
    * **⚠️ Ngân sách bị tính bằng SAI ĐƠN VỊ.** "50 vòng" quy ra
      `50 × độ trễ(/health)` ≈ 60 ms trên máy này — một đại lượng đo **độ nhanh
      của một endpoint không chạm Postgres**, trong khi thứ đang chờ là một
      vòng đi-về Postgres. Hai đại lượng ấy **không tương quan**: một lần khựng
      của Postgres (checkpoint, fsync, mở kết nối mới trong pool) kéo dài vế
      phải mà không đụng vế trái. Nên **không hằng số nào** làm nó an toàn —
      nâng 50 lên 500 chỉ làm bài test chập chờn hiếm hơn, đúng như việc nới
      `sleep` trong `test_ttl_expires_entry` sẽ chỉ thu hẹp cửa sổ chứ không
      xoá nó.

    ⚠️ Tham số `client` **biến mất cùng với `/health`** — và suýt thì không.
    Bản nháp của bản vá này giữ nó lại kèm lý do *"bỏ đi là đổi 22 chỗ gọi"*.
    Đếm ra thì có **đúng một** chỗ gọi: 22 là số **lượt chạy**, không phải số
    **chỗ gọi**. Đó đúng là lỗi mà `NEW-10` §4 vừa ghi lại cùng ngày — giữ một
    thứ thừa rồi bịa một lý do nghe hợp lý cho nó — tái diễn trong vòng một
    giờ, ở tay cùng một người. Nên: đếm, đừng ước lượng.
    """
    from serving.core.chat import _PENDING

    deadline = time.monotonic() + DRAIN_TIMEOUT_S
    while _PENDING:
        if time.monotonic() > deadline:
            # Báo cáo **cái gì** còn treo, không chỉ **rằng** có gì đó treo:
            # chậm và treo là hai chẩn đoán khác nhau, và lượt đỏ sau phải phân
            # biệt được chúng từ chính dòng annotation của CI.
            raise AssertionError(
                f"task ghi message không kết thúc sau {DRAIN_TIMEOUT_S:.0f}s; "
                f"còn treo: {[t.get_coro() for t in _PENDING]}"
            )
        time.sleep(0.001)


def _rate(
    client: TestClient, message_id: str, rating: int, *, key: str = KEY, **extra: Any
) -> httpx.Response:
    response: httpx.Response = client.post(
        "/feedback",
        json={"message_id": message_id, "rating": rating, **extra},
        headers={"Authorization": f"Bearer {key}"},
    )
    return response


# ---------------------------------------------------------------------------
# 1. DoD: một câu 👎 đi hết đường
# ---------------------------------------------------------------------------


class TestTheHappyPath:
    def test_a_thumbs_down_lands_in_postgres_and_in_langfuse(self, app: Any) -> None:
        """Câu DoD, viết ra thành một chuỗi bốn phép kiểm."""
        client, sink = app
        meta = _turn(client)

        response = _rate(
            client, meta["answer_message_id"], -1, reason="not_found", comment="báo cáo có nói mà"
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["rating"] == -1
        assert body["reason"] == "not_found"
        assert body["replaced"] is False

        # (1) trace_id lấy từ HÀNG, và nó khớp trace của đúng lượt ấy
        assert body["trace_id"] == meta["trace_id"]
        # (2) điểm đã xếp hàng sang Langfuse
        assert body["scored"] is True
        (score,) = sink.scores
        assert score.trace_id == meta["trace_id"]
        assert score.value == -1.0
        assert score.name == SCORE_NAME
        assert score.comment == "not_found · báo cáo có nói mà"
        # (3) hàng đợi review nhìn thấy nó
        queue = client.get(
            "/admin/feedback", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        ).json()
        assert queue["count"] == 1
        (item,) = queue["items"]
        assert item["message_id"] == meta["answer_message_id"]
        assert item["question"] == "RRF là gì?"
        assert item["answer"] == "RRF hợp nhất thứ hạng."
        assert item["model"] == "fake-model-served"
        assert item["bundle_version"] == "0.2.0"
        assert item["retrieved_chunk_ids"]

    def test_the_candidate_file_can_be_written_from_it(self, app: Any, tmp_path: Path) -> None:
        """Nửa sau của DoD: *"câu 👎 xuất ra được file candidate"*."""
        client, _ = app
        meta = _turn(client, "Câu hỏi sẽ bị chấm kém?")
        assert _rate(client, meta["answer_message_id"], -1, reason="wrong").status_code == 201

        response = client.get(
            "/admin/feedback/candidates", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        )
        assert response.status_code == 200
        assert response.headers["X-Candidate-Count"] == "1"
        (line,) = response.text.splitlines()
        candidate = json.loads(line)
        assert candidate["query"] == "Câu hỏi sẽ bị chấm kém?"
        assert candidate["rating"] == -1
        assert candidate["reviewed_by_human"] is False
        # ⭐⭐ Và nó KHÔNG mang nhãn — xem `GoldenCandidate`.
        assert "relevant_chunk_ids" not in candidate
        assert "reference_answer" not in candidate

        path = tmp_path / "candidates.jsonl"
        path.write_text(response.text, encoding="utf-8")
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    def test_a_thumbs_up_is_recorded_but_stays_out_of_the_review_queue(self, app: Any) -> None:
        """Hàng đợi review là danh sách **việc phải làm**; một lượt hài lòng
        không phải việc phải làm."""
        client, _ = app
        meta = _turn(client)
        assert _rate(client, meta["answer_message_id"], 1).status_code == 201

        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        assert client.get("/admin/feedback", headers=headers).json()["count"] == 0
        assert client.get("/admin/feedback?rating=0", headers=headers).json()["count"] == 1


# ---------------------------------------------------------------------------
# 2. ⭐⭐ Khoá nối không đến từ người gọi
# ---------------------------------------------------------------------------


class TestTheJoinKeyIsProven:
    def test_a_tenant_cannot_rate_another_tenants_answer(self, app: Any) -> None:
        """RLS lọc trước, nên hàng của `acme` **không tồn tại** với `globex` —
        404, không 403. Hướng ấy cũng đúng về mặt rò rỉ: một 403 xác nhận rằng
        id ấy có thật."""
        client, sink = app
        meta = _turn(client)
        before = len(sink.scores)

        response = _rate(client, meta["answer_message_id"], -1, key=OTHER_KEY)
        assert response.status_code == 404
        assert len(sink.scores) == before, "không được gắn điểm vào trace của tenant khác"

    def test_the_endpoint_refuses_a_body_that_carries_a_trace_id(self, app: Any) -> None:
        """`extra="forbid"`. Nếu một ngày trường này được nhận thì bài này đỏ,
        và người thêm nó phải đọc lý do trước khi xoá bài."""
        client, _ = app
        meta = _turn(client)
        response = client.post(
            "/feedback",
            json={
                "message_id": meta["answer_message_id"],
                "rating": -1,
                "trace_id": "0" * 32,
            },
            headers={"Authorization": f"Bearer {KEY}"},
        )
        assert response.status_code == 422

    def test_rating_the_question_instead_of_the_answer_is_refused(self, app: Any) -> None:
        """Khung `meta` mang **hai** id, và chấm nhầm cái kia là lỗi dễ nhất
        của người tích hợp — nên nó phải là 422 với lời giải thích, không phải
        một hàng hợp lệ vô nghĩa."""
        client, _ = app
        meta = _turn(client)
        response = _rate(client, meta["message_id"], -1)
        assert response.status_code == 422
        assert "role='user'" in response.json()["detail"]

    def test_an_unknown_message_is_404(self, app: Any) -> None:
        client, _ = app
        _turn(client)
        assert _rate(client, "f" * 32, -1).status_code == 404


# ---------------------------------------------------------------------------
# 3. Idempotent ở CẢ HAI kho
# ---------------------------------------------------------------------------


class TestOneRatingPerAnswer:
    def test_a_double_click_does_not_double_the_queue(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        first = _rate(client, meta["answer_message_id"], -1, reason="wrong")
        second = _rate(client, meta["answer_message_id"], -1, reason="wrong")
        assert first.status_code == second.status_code == 201
        assert first.json()["replaced"] is False
        assert second.json()["replaced"] is True
        assert first.json()["id"] == second.json()["id"]

        queue = client.get(
            "/admin/feedback", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        ).json()
        assert queue["count"] == 1

    def test_changing_your_mind_overwrites_in_both_stores(self, app: Any) -> None:
        """⭐⭐ Một luật idempotent cho hai kho: Postgres theo
        `(tenant, message)`, Langfuse theo `score_id()`."""
        client, sink = app
        meta = _turn(client)
        _rate(client, meta["answer_message_id"], -1, reason="wrong")
        _rate(client, meta["answer_message_id"], 1)

        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        assert client.get("/admin/feedback?rating=0", headers=headers).json()["count"] == 1
        (item,) = client.get("/admin/feedback?rating=0", headers=headers).json()["items"]
        assert item["rating"] == 1
        assert item["reason"] is None

        assert [s.value for s in sink.scores] == [-1.0, 1.0]
        ids = {score_id(s.trace_id, s.name) for s in sink.scores}
        assert len(ids) == 1, "hai điểm phải cùng id, nếu không Langfuse giữ cả hai"

    def test_a_re_rating_bubbles_back_to_the_top_of_the_queue(self, app: Any) -> None:
        """⭐ Đổi ý là một tín hiệu **mới**, và hàng đợi sắp theo thời gian.

        Giữ nguyên `created_at` cũ thì lần chấm lại chìm xuống dưới những lượt
        đã xảy ra sau nó, và người review không bao giờ thấy nó nổi lên — hàng
        đợi vẫn đúng số lượng, chỉ sai thứ tự. Phép tiêm `F7` sống sót vì không
        bài nào nhìn tới thứ tự.
        """
        client, _ = app
        first = _turn(client, "câu hỏi thứ nhất?")
        second = _turn(client, "câu hỏi thứ hai?")
        _rate(client, first["answer_message_id"], -1, reason="wrong")
        _rate(client, second["answer_message_id"], -1, reason="slow")

        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        order = [
            i["message_id"] for i in client.get("/admin/feedback", headers=headers).json()["items"]
        ]
        assert order[0] == second["answer_message_id"]

        _rate(client, first["answer_message_id"], -1, reason="citation")
        order = [
            i["message_id"] for i in client.get("/admin/feedback", headers=headers).json()["items"]
        ]
        assert order[0] == first["answer_message_id"], "lần chấm lại phải nổi lên đầu"


# ---------------------------------------------------------------------------
# 4. Cầu nối `answer_message_id` và hai cột citations của `0004`
# ---------------------------------------------------------------------------


class TestTheAnswerRow:
    def test_the_id_in_the_meta_frame_is_the_row_that_gets_written(self, app: Any) -> None:
        """Sinh ở `prepare()`, phát ở khung đầu, ghi ở một task nền sau khung
        cuối. Ba chỗ, một giá trị."""
        client, _ = app
        meta = _turn(client)
        history = client.get(
            f"/conversations/{meta['conversation_id']}",
            headers={"Authorization": f"Bearer {KEY}"},
        ).json()
        assistant = [m for m in history["messages"] if m["role"] == "assistant"]
        assert [m["id"] for m in assistant] == [meta["answer_message_id"]]

    def test_history_carries_the_trace_id_and_both_citation_columns(self, app: Any) -> None:
        """`TD-50` đóng ở đây: đọc lại một câu trả lời cũ mang theo **kết quả
        xác minh** của nó, không chỉ danh sách nguồn đã đưa vào."""
        client, _ = app
        meta = _turn(client)
        history = client.get(
            f"/conversations/{meta['conversation_id']}",
            headers={"Authorization": f"Bearer {KEY}"},
        ).json()
        (answer,) = [m for m in history["messages"] if m["role"] == "assistant"]
        assert answer["trace_id"] == meta["trace_id"]
        assert answer["sources"], "nguồn đã đưa cho model"
        assert answer["citations"] is not None, "khung xác minh của W4-09"
        assert "block" in answer["citations"]

    def test_the_two_columns_are_not_the_same_thing(self, app: Any) -> None:
        """⚠️ Bài này là chỗ lỗi của `0001` chết: một cột tên `citations` chứa
        `sources()` trông đúng cho tới khi ai đó hỏi *"citation nào đã được xác
        minh"*."""
        client, _ = app
        meta = _turn(client)
        history = client.get(
            f"/conversations/{meta['conversation_id']}",
            headers={"Authorization": f"Bearer {KEY}"},
        ).json()
        (answer,) = [m for m in history["messages"] if m["role"] == "assistant"]
        assert isinstance(answer["sources"], list)
        assert isinstance(answer["citations"], dict)


# ---------------------------------------------------------------------------
# 5. Hàng rào và số đo
# ---------------------------------------------------------------------------


class TestTheEdges:
    def test_the_review_queue_needs_the_admin_scope(self, app: Any) -> None:
        client, _ = app
        response = client.get("/admin/feedback", headers={"Authorization": f"Bearer {KEY}"})
        assert response.status_code == 403

    def test_posting_feedback_does_not(self, app: Any) -> None:
        """Người dùng thường phải chấm được — nếu không thì không có tín hiệu nào."""
        client, _ = app
        meta = _turn(client)
        assert _rate(client, meta["answer_message_id"], -1).status_code == 201

    def test_feedback_needs_a_key_at_all(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        response = client.post(
            "/feedback", json={"message_id": meta["answer_message_id"], "rating": -1}
        )
        assert response.status_code == 401

    def test_the_metric_counts_it_by_rating_and_reason(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        _rate(client, meta["answer_message_id"], -1, reason="citation")

        body = client.get("/metrics", headers={"Authorization": f"Bearer {KEY}"}).text
        line = next(
            ln
            for ln in body.splitlines()
            if ln.startswith('rag_feedback_total{rating="-1",reason="citation"}')
        )
        assert line.endswith(" 1.0")

    def test_an_unknown_reason_is_422_not_500(self, app: Any) -> None:
        """Ba bản sao của danh sách lý do phải trùng nhau; nếu chúng lệch thì
        Postgres từ chối `INSERT` và người dùng nhận 500."""
        client, _ = app
        meta = _turn(client)
        response = _rate(client, meta["answer_message_id"], -1, reason="hallucination")
        assert response.status_code == 422

    def test_a_comment_longer_than_the_cap_is_refused(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        assert _rate(client, meta["answer_message_id"], -1, comment="x" * 2001).status_code == 422

    @pytest.mark.asyncio
    async def test_the_core_layer_guards_the_vocabulary_too_not_only_the_api(
        self, app: Any
    ) -> None:
        """⭐ `Literal` của FastAPI chặn đường HTTP, nhưng `record_feedback` còn
        có **một** người gọi khác: CLI xuất ứng viên và bất kỳ script vận hành
        nào. Bỏ phép kiểm ở lõi thì một mã lạ đi thẳng tới `CheckConstraint`
        của Postgres và quay ra thành `IntegrityError` — tức 500 thay vì một
        lời từ chối đọc được.

        Phép tiêm `F19` sống sót vì mọi bài khác đi qua HTTP, nơi `Literal` đã
        chặn trước.
        """
        from serving.core.auth import Principal
        from serving.core.feedback import record_feedback

        client, _ = app
        meta = _turn(client)
        sessions = client.app.state.chat.sessions
        principal = Principal(tenant_id="acme", key_id="direct")

        with pytest.raises(ValueError, match="reason không hợp lệ"):
            await record_feedback(
                sessions,
                principal,
                message_id=meta["answer_message_id"],
                rating=-1,
                reason="hallucination",
            )
        with pytest.raises(ValueError, match="rating phải là"):
            await record_feedback(
                sessions, principal, message_id=meta["answer_message_id"], rating=0
            )


# ---------------------------------------------------------------------------
# `NEW-08`/`AU-07` — ghép câu hỏi bằng khoá thật, không bằng đồng hồ
# ---------------------------------------------------------------------------


def _insert_conversation(database: Any, conv_id: str) -> None:
    from sqlalchemy import text

    with database.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO conversation (id, tenant_id, bundle_version)"
                " VALUES (:id, 'acme', '0.1.0')"
            ),
            {"id": conv_id},
        )


def _insert_message(
    database: Any,
    *,
    id: str,
    conv_id: str,
    role: str,
    content: str,
    offset_s: int,
    user_message_id: str | None = None,
) -> None:
    """Ghi thẳng bằng engine chủ (bỏ qua app) với `created_at` TƯỜNG MINH —
    kịch bản chồng lấn cần kiểm soát đồng hồ, không được phó mặc cho `now()`."""
    from sqlalchemy import text

    with database.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO message (id, tenant_id, conversation_id, role, content,"
                " created_at, user_message_id) VALUES (:id, 'acme', :conv, :role, :content,"
                " now() + make_interval(secs => :offset), :umid)"
            ),
            {
                "id": id,
                "conv": conv_id,
                "role": role,
                "content": content,
                "offset": offset_s,
                "umid": user_message_id,
            },
        )


def _queue(client: TestClient) -> list[dict[str, Any]]:
    body = client.get("/admin/feedback", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).json()
    items: list[dict[str, Any]] = body["items"]
    return items


class TestQuestionAnswerPairing:
    def test_the_saved_answer_row_carries_the_user_message_key(
        self, app: Any, database: Any
    ) -> None:
        """Hàng trợ lý phải mang id của ĐÚNG câu hỏi nó trả lời — giá trị đã
        có sẵn trên `ChatTurn` từ `W4-06`, giờ được ghi xuống (`0005`)."""
        from sqlalchemy import text

        client, _ = app
        meta = _turn(client)

        with database.connect() as conn:
            stored = conn.execute(
                text("SELECT user_message_id FROM message WHERE id = :id"),
                {"id": meta["answer_message_id"]},
            ).scalar()
        assert stored == meta["message_id"]

    def test_overlapping_turns_do_not_swap_questions(self, app: Any, database: Any) -> None:
        """Kịch bản hai lượt chồng nhau, dựng đúng thứ tự ghi thật: user_A và
        user_B ghi ngay (hai request song song), assistant_A ghi SAU CÙNG
        (task nền của lượt chậm hơn). Suy luận "user muộn nhất trước answer"
        chọn B — sai; khoá `user_message_id` phải chọn A. Không có khoá, ứng
        viên golden mang câu hỏi B dán lên câu trả lời của A, sai không dấu
        vết."""
        client, _ = app
        _insert_conversation(database, "convrace")
        _insert_message(
            database, id="user_a", conv_id="convrace", role="user", content="Câu hỏi A?", offset_s=0
        )
        _insert_message(
            database, id="user_b", conv_id="convrace", role="user", content="Câu hỏi B?", offset_s=1
        )
        _insert_message(
            database,
            id="ans_a",
            conv_id="convrace",
            role="assistant",
            content="Trả lời cho A.",
            offset_s=2,
            user_message_id="user_a",
        )

        assert _rate(client, "ans_a", -1).status_code == 201
        mine = [i for i in _queue(client) if i["message_id"] == "ans_a"]
        assert mine, "câu 👎 phải nằm trong hàng đợi"
        assert mine[0]["question"] == "Câu hỏi A?", "khoá thật phải thắng suy luận thời gian"

    def test_a_legacy_row_still_pairs_by_the_old_heuristic(self, app: Any, database: Any) -> None:
        """Hàng ghi trước `0005` không có khoá — đường ghép cũ vẫn phải chạy.
        Bài này ghim rằng fallback TỒN TẠI, không ghim rằng nó đúng: rủi ro
        chọn nhầm là thuộc tính của dữ liệu cũ, không xoá được bằng code mới."""
        client, _ = app
        _insert_conversation(database, "convold")
        _insert_message(
            database,
            id="old_user",
            conv_id="convold",
            role="user",
            content="Câu hỏi cũ?",
            offset_s=0,
        )
        _insert_message(
            database,
            id="old_ans",
            conv_id="convold",
            role="assistant",
            content="Trả lời cũ.",
            offset_s=1,
        )

        assert _rate(client, "old_ans", -1).status_code == 201
        mine = [i for i in _queue(client) if i["message_id"] == "old_ans"]
        assert mine and mine[0]["question"] == "Câu hỏi cũ?"


class TestCommentRedaction:
    def test_pii_in_a_comment_is_redacted_before_it_is_stored(self, app: Any) -> None:
        """`NEW-08`/`AU-05`: cột comment chảy đi ba ngả (review queue, điểm
        Langfuse, file ứng viên golden sẽ sống trong git) — redact TẠI NGUỒN."""
        client, _ = app
        meta = _turn(client)

        response = _rate(
            client,
            meta["answer_message_id"],
            -1,
            reason="other",
            comment="gọi tôi 0912345678 hoặc toi@example.com nhé",
        )
        assert response.status_code == 201

        mine = [i for i in _queue(client) if i["message_id"] == meta["answer_message_id"]]
        assert mine, "lượt vừa chấm phải trong hàng đợi"
        comment = mine[0]["comment"]
        assert "0912345678" not in comment
        assert "toi@example.com" not in comment
