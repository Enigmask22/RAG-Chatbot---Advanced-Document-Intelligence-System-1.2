"""`W4-06` — `POST /chat` SSE, trên **uvicorn thật** và **Postgres thật**.

## Vì sao không dùng `TestClient`

`W4-03` đã ghim điều này bằng một test riêng: `TestClient` đệm phản hồi dạng
dòng, nên nó **không phân biệt được** một cài đặt streaming đúng với một cài đặt
gom hết token rồi gửi một cục. Câu DoD *"nhận ≥ 2 chunk SSE"* viết bằng nó là
một test xanh không chứng minh gì.

Nên ở đây có một tiến trình uvicorn thật, và phép kiểm là **thời điểm tới** của
từng khung, không phải số lượng của chúng.

Postgres cũng thật, vì nửa sau của DoD là *"chat history sống sót qua restart
container"* — và cách duy nhất kiểm câu đó là giết tiến trình rồi dựng lại
(`test_history_survives_a_restart`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, text

from rag_core.settings import get_settings
from serving.core.auth import digest_of
from serving.db.engine import make_engine
from serving.db.models import RLS_TABLES
from tests.integration.chat_app import (
    ENV_BUNDLES,
    ENV_DELTA_MS,
    ENV_KEYS,
    ENV_MODE,
    ENV_RETRIEVAL_MS,
    write_keys,
)
from tests.integration.test_bundle_reload import write_bundle

pytestmark = pytest.mark.integration

ACME_KEY = "rag_acme_chat_key"
GLOBEX_KEY = "rag_globex_chat_key"
ADMIN_KEY = "rag_acme_admin_key"

_PORTS = iter(range(8091, 8120))
"""⚠️ Mỗi tiến trình uvicorn phải có **cổng riêng**.

Lần viết đầu dùng một hằng số `PORT` cho tất cả. Server module-scope đã giữ cổng
ấy, nên mọi server dựng thêm (mode `fail_mid`, truy hồi chậm, restart) **không
bind được** — và vòng chờ khởi động lại hỏi `/health` của server **cũ**, thấy
200, rồi báo sẵn sàng. Bốn test chạy hết trên server sai, ba trong số đó vẫn
xanh vì hành vi mặc định giống nhau.

Cùng họ với phép tiêm hỏng ở `W4-05`: một test xanh có hai cách giải thích, và
"nó chưa từng chạy thứ tôi tưởng" là cách phải loại trừ trước.
"""


# ---------------------------------------------------------------------------
# Hạ tầng
# ---------------------------------------------------------------------------


def _alembic(*argv: str) -> None:
    from alembic.config import Config

    from alembic import command
    from serving.db.engine import _ALEMBIC_DIR

    config = Config(str(_ALEMBIC_DIR.parent / "alembic.ini"))
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    getattr(command, argv[0])(config, *argv[1:])


@pytest.fixture(scope="module")
def database() -> Iterator[Engine]:
    from sqlalchemy.exc import OperationalError

    owner = make_engine(get_settings().postgres_dsn)
    try:
        with owner.connect():
            pass
    except OperationalError as exc:  # pragma: no cover - phụ thuộc máy
        pytest.skip(f"không có Postgres: {exc}")
    _alembic("upgrade", "head")
    with owner.begin() as conn:
        for table in RLS_TABLES:
            conn.execute(text(f"DELETE FROM {table}"))
    yield owner
    with owner.begin() as conn:
        for table in RLS_TABLES:
            conn.execute(text(f"DELETE FROM {table}"))
    owner.dispose()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("chat")
    write_bundle(root / "bundles", "0.2.0")
    write_keys(
        root / "api-keys.json",
        {
            digest_of(ACME_KEY): {
                "tenant_id": "acme",
                "key_id": "acme-1",
                "scopes": [],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(GLOBEX_KEY): {
                "tenant_id": "globex",
                "key_id": "globex-1",
                "scopes": [],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(ADMIN_KEY): {
                "tenant_id": "acme",
                "key_id": "acme-admin",
                "scopes": ["admin"],
                "rate_limit_per_minute": 10_000,
            },
        },
    )
    return root


def _serve(workspace: Path, **extra: str) -> tuple[subprocess.Popen[bytes], str]:
    port = next(_PORTS)
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        ENV_BUNDLES: str(workspace / "bundles"),
        ENV_KEYS: str(workspace / "api-keys.json"),
        ENV_DELTA_MS: "120",
        ENV_RETRIEVAL_MS: "0",
        ENV_MODE: "ok",
        **extra,
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "serving",
            "--app",
            "tests.integration.chat_app:make",
            "--factory",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:  # pragma: no cover - máy hỏng
            raise RuntimeError("uvicorn chết lúc khởi động")
        try:
            if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                return proc, base
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.terminate()  # pragma: no cover
    raise RuntimeError("uvicorn không lên trong 60 s")  # pragma: no cover


@pytest.fixture(scope="module")
def server(database: Engine, workspace: Path) -> Iterator[str]:
    proc, base = _serve(workspace)
    yield base
    proc.terminate()
    proc.wait(timeout=20)


def _headers(key: str = ACME_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _frames(response: httpx.Response) -> Iterator[tuple[float, str, dict[str, Any]]]:
    """Từng khung kèm **thời điểm** nó tới — đó là thứ đang được kiểm."""
    name = ""
    for line in response.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            yield time.perf_counter(), name, json.loads(line[6:])


def _chat(
    client: httpx.Client, base: str, message: str, **body: Any
) -> tuple[httpx.Response, list[tuple[float, str, dict[str, Any]]]]:
    with client.stream(
        "POST", f"{base}/chat", json={"message": message, **body}, headers=_headers()
    ) as response:
        return response, list(_frames(response))


# ---------------------------------------------------------------------------
# 1. ⭐⭐ DoD — token về theo dòng, không về một cục
# ---------------------------------------------------------------------------


def test_deltas_arrive_spread_out_in_time_not_all_at_once(
    server: str,
) -> None:
    """⭐ Phép kiểm là **khoảng cách thời gian**, không phải số khung.

    Server chờ 120 ms giữa hai token. Một cài đặt gom hết rồi gửi một cục cũng
    trả về đúng 5 khung `delta` — cùng nội dung, cùng số lượng, và tất cả tới
    trong cùng một mili giây. Đếm khung không phân biệt được hai thứ đó; đo
    thời điểm thì có.
    """
    with httpx.Client(timeout=30.0) as client:
        _, frames = _chat(client, server, "RRF là gì?")

    deltas = [(t, data) for t, name, data in frames if name == "delta"]
    assert len(deltas) >= 2
    spread = deltas[-1][0] - deltas[0][0]
    assert spread > 0.3, (
        f"5 khung tới trong {spread * 1000:.0f} ms — đây là một cục, không phải dòng"
    )


def test_the_frame_order_and_the_terminal_frame(server: str) -> None:
    with httpx.Client(timeout=30.0) as client:
        response, frames = _chat(client, server, "RRF là gì?")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    names = [name for _, name, _ in frames]
    assert names[0] == "meta"
    assert names[1] == "sources"
    assert names[-1] == "done"
    assert names.count("delta") == 5


def test_the_stream_carries_a_request_id_like_every_other_response(
    server: str,
) -> None:
    """Middleware ASGI thuần bọc `send`, nên nó gắn được header **trước** khi
    khung đầu tiên rời đi — thứ `BaseHTTPMiddleware` không làm được mà không
    đệm cả stream."""
    with (
        httpx.Client(timeout=30.0) as client,
        client.stream(
            "POST", f"{server}/chat", json={"message": "x"}, headers=_headers()
        ) as response,
    ):
        assert response.headers["x-request-id"]
        assert response.headers["cache-control"] == "no-cache, no-transform"
        assert response.headers["x-accel-buffering"] == "no"
        response.read()


def test_sources_and_usage_are_reported(server: str) -> None:
    with httpx.Client(timeout=30.0) as client:
        _, frames = _chat(client, server, "RRF là gì?")

    by_name = {name: data for _, name, data in frames}
    assert [s["n"] for s in by_name["sources"]["sources"]] == [1, 2]
    assert by_name["sources"]["sources"][0]["title"] == "Tài liệu 1"
    assert by_name["done"]["model"] == "scripted-model-served"
    assert by_name["done"]["usage"]["completion_tokens"] == 11
    assert by_name["done"]["ttfb_ms"] < by_name["done"]["total_ms"]


# ---------------------------------------------------------------------------
# 2. ⭐ Vòng lặp sự kiện không bị truy hồi chặn
# ---------------------------------------------------------------------------


def test_health_answers_while_a_chat_is_retrieving(database: Engine, workspace: Path) -> None:
    """⭐ `retrieve()` là đồng bộ và tốn hàng trăm mili giây (embed + rerank).

    Gọi thẳng nó trong `async def` thì suốt khoảng đó `/health` không trả lời —
    và orchestrator đọc đúng điều đó là "tiến trình chết", rồi giết một pod đang
    hoạt động bình thường. Đây là cùng một bài học với ba handler `def` của
    `admin.py`, chỉ khác là ở đây nó xảy ra ở **mọi** request.
    """
    proc, base = _serve(workspace, CHAT_TEST_RETRIEVAL_MS="1500", CHAT_TEST_DELTA_MS="10")
    chatting = threading.Thread(
        target=lambda: httpx.post(
            f"{base}/chat", json={"message": "chậm"}, headers=_headers(), timeout=30.0
        )
    )
    try:
        chatting.start()
        # ⚠️ Phải đo từ một **luồng khác**, và bản đầu của test này thì không: nó
        # mở `client.stream(...)` rồi đo bên trong khối `with`. Nhưng `stream()`
        # chỉ trả về khi **header** đã tới, mà header của SSE chỉ ra sau khi
        # `prepare()` — tức cả phần truy hồi — đã xong. Phép đo vì thế rơi đúng
        # vào lúc vòng lặp đã rảnh trở lại: test xanh kể cả khi truy hồi chặn
        # event loop, và phép tiêm lỗi tương ứng **không đỏ**.
        #
        # Cùng khuôn với `test_a_hung_dependency_does_not_hang_ready` ở `W4-03`:
        # cái sai không nằm ở ngưỡng mà ở **chỗ đặt đồng hồ**.
        time.sleep(0.4)
        started = time.perf_counter()
        probe = httpx.get(f"{base}/health", timeout=10.0)
        elapsed = time.perf_counter() - started
    finally:
        chatting.join(timeout=30)
        proc.terminate()
        proc.wait(timeout=20)

    assert probe.status_code == 200
    assert elapsed < 0.5, f"/health mất {elapsed * 1000:.0f} ms — truy hồi đang chặn event loop"


# ---------------------------------------------------------------------------
# 3. ⭐⭐ Đường phân giới: còn status thật vs chỉ còn khung SSE
# ---------------------------------------------------------------------------


def test_a_missing_conversation_is_404_not_a_200_with_an_error_frame(
    server: str,
) -> None:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{server}/chat",
            json={"message": "x", "conversation_id": "khongcothat"},
            headers=_headers(),
        )

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_a_failure_after_the_first_byte_can_only_be_an_error_frame(
    database: Engine, workspace: Path
) -> None:
    """⭐ Nửa dưới của đường phân giới.

    Provider chết ở token thứ 3. Status **đã** là 200 và không đổi được nữa — nên
    thứ duy nhất còn nói được sự thật là một khung `error`. Không có nó, client
    nhận đúng hai token rồi im lặng, và điều đó trông y hệt một câu trả lời ngắn
    đã kết thúc bình thường.
    """
    proc, base = _serve(workspace, CHAT_TEST_MODE="fail_mid", CHAT_TEST_DELTA_MS="10")
    try:
        with httpx.Client(timeout=30.0) as client:
            response, frames = _chat(client, base, "sẽ hỏng")
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    assert response.status_code == 200
    names = [name for _, name, _ in frames]
    assert names[-1] == "error"
    assert "done" not in names
    assert frames[-1][2]["partial_chars"] == len("Theo tài liệu ")


def test_an_infrastructure_failure_is_a_503_that_keeps_internals_server_side(
    database: Engine, workspace: Path
) -> None:
    """`NEW-08`/`AU-03`: Qdrant chết thì client nhận một 503 nói "thử lại" kèm
    `request_id` — KHÔNG nhận tên service, port, tên collection. Chuỗi lỗi giả
    trong harness (`BOOM_MESSAGE`) mang đúng ba thứ ấy, và bài này canh rằng
    không mảnh nào của nó đi qua biên HTTP. Chi tiết đầy đủ vẫn ở log server
    (`logger.exception`) và trace — chỗ của người vận hành, không phải client.
    """
    proc, base = _serve(workspace, CHAT_TEST_RETRIEVAL_BOOM="1")
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{base}/chat", json={"message": "câu hỏi thường"}, headers=_headers()
            )
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    assert response.status_code == 503
    body = json.dumps(response.json())
    for fragment in ("qdrant", "6333", "rag_noi_bo", "ResponseHandlingException"):
        assert fragment not in body, f"chi tiết nội bộ {fragment!r} rò ra client"
    assert response.json()["detail"]["request_id"], "phải có request_id để đối chiếu log"


def test_another_tenants_conversation_is_404_not_403(
    server: str,
) -> None:
    """⭐ RLS làm cho hai ca không phân biệt được, và phản hồi giữ nguyên như thế.

    Trả `403` cho hội thoại của người khác và `404` cho hội thoại không tồn tại
    biến endpoint này thành máy dò `conversation_id`.
    """
    with httpx.Client(timeout=30.0) as client:
        _, frames = _chat(client, server, "của acme")
        conv = frames[0][2]["conversation_id"]

        mine = client.get(f"{server}/conversations/{conv}", headers=_headers(ACME_KEY))
        theirs = client.get(f"{server}/conversations/{conv}", headers=_headers(GLOBEX_KEY))
        ghost = client.get(f"{server}/conversations/khongcothat", headers=_headers(GLOBEX_KEY))

    assert mine.status_code == 200
    # Cùng status, cùng khuôn thông báo, và thông báo chỉ nhắc lại đúng cái id
    # người gọi vừa gửi — tức nó không mang thêm bit thông tin nào.
    assert theirs.status_code == ghost.status_code == 404
    assert theirs.json()["detail"] == f"không có hội thoại '{conv}'"
    assert ghost.json()["detail"] == "không có hội thoại 'khongcothat'"


def test_chat_needs_a_key_like_everything_else(server: str) -> None:
    with httpx.Client(timeout=30.0) as client:
        assert client.post(f"{server}/chat", json={"message": "x"}).status_code == 401


def test_a_client_supplied_tenant_filter_cannot_widen_the_search(
    server: str,
) -> None:
    """`tenant_filter()` của `W4-04` có người gọi thật lần đầu ở đây."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{server}/chat",
            json={"message": "x", "filters": {"tenant_id": "globex"}},
            headers=_headers(ACME_KEY),
        )

    assert response.status_code == 403
    assert "globex" not in response.text


# ---------------------------------------------------------------------------
# 4. ⭐⭐ Lịch sử — nửa sau của DoD
# ---------------------------------------------------------------------------


def _wait_for_assistant(
    client: httpx.Client, base: str, conv: str, key: str = ACME_KEY
) -> list[Any]:
    """Message trợ lý được ghi bằng một task **ngoài** request (xem `chat.py`),
    nên nó có thể tới sau khung `done` vài mili giây."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"{base}/conversations/{conv}", headers=_headers(key)).json()
        messages: list[Any] = body["messages"]
        if any(m["role"] == "assistant" for m in messages):
            return messages
        time.sleep(0.1)
    raise AssertionError("không có message trợ lý sau 10 s")


def test_a_turn_is_written_as_two_rows_with_provenance(
    server: str,
) -> None:
    with httpx.Client(timeout=30.0) as client:
        _, frames = _chat(client, server, "câu hỏi được lưu")
        conv = frames[0][2]["conversation_id"]
        messages = _wait_for_assistant(client, server, conv)

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "câu hỏi được lưu"
    assert messages[1]["content"] == "Theo tài liệu [1], câu trả lời là vậy."
    # `W4-06` thêm hai cột này (migration `0002`) — không có chúng thì một câu
    # trả lời cụt không phân biệt được với một câu trả lời ngắn.
    assert messages[1]["model"] == "scripted-model-served"
    assert messages[1]["finish_reason"] == "stop"
    # `W5-08` (migration `0004`) tách cột: `sources` = đã đưa gì cho model,
    # `citations` = model tuyên bố gì và quote nào khớp. Trước `0004` một cột
    # tên `citations` chứa cái đầu — xem `0004_feedback_loop`.
    assert len(messages[1]["sources"]) == 2
    assert messages[1]["trace_id"]


def test_the_second_turn_sees_the_first_one(server: str) -> None:
    with httpx.Client(timeout=30.0) as client:
        _, first = _chat(client, server, "lượt một")
        conv = first[0][2]["conversation_id"]
        _wait_for_assistant(client, server, conv)
        _, second = _chat(client, server, "lượt hai", conversation_id=conv)
        assert second[0][2]["conversation_id"] == conv
        messages = client.get(f"{server}/conversations/{conv}", headers=_headers()).json()[
            "messages"
        ]

    assert [m["content"] for m in messages][:3] == [
        "lượt một",
        "Theo tài liệu [1], câu trả lời là vậy.",
        "lượt hai",
    ]


def test_history_survives_a_restart(database: Engine, workspace: Path) -> None:
    """⭐ Câu DoD nguyên văn: *"chat history sống sót qua restart container"*.

    Cách duy nhất kiểm nó là giết tiến trình. Một test đọc lại từ cùng một tiến
    trình chỉ chứng minh rằng biến trong bộ nhớ còn nguyên.
    """
    first, first_base = _serve(workspace)
    try:
        with httpx.Client(timeout=30.0) as client:
            _, frames = _chat(client, first_base, "sống sót qua restart")
            conv = frames[0][2]["conversation_id"]
            _wait_for_assistant(client, first_base, conv)
    finally:
        first.terminate()
        first.wait(timeout=20)

    second, second_base = _serve(workspace)
    try:
        with httpx.Client(timeout=30.0) as client:
            body = client.get(f"{second_base}/conversations/{conv}", headers=_headers()).json()
    finally:
        second.terminate()
        second.wait(timeout=20)

    assert [m["content"] for m in body["messages"]] == [
        "sống sót qua restart",
        "Theo tài liệu [1], câu trả lời là vậy.",
    ]


def test_a_client_that_hangs_up_still_gets_its_partial_answer_saved(
    database: Engine, workspace: Path
) -> None:
    """⭐⭐ Token đã trả tiền rồi.

    Đọc 2 khung `delta` rồi đóng kết nối. Phần đã sinh phải nằm trong DB, đánh
    dấu `client_disconnect` — nếu không thì lịch sử có một câu hỏi không có câu
    trả lời, và hoá đơn có một khoản không có gì đối chiếu.
    """
    proc, base = _serve(workspace, CHAT_TEST_DELTA_MS="250")
    try:
        with httpx.Client(timeout=30.0) as client:
            conv = ""
            seen = 0
            with client.stream(
                "POST", f"{base}/chat", json={"message": "sẽ đóng tab"}, headers=_headers()
            ) as response:
                for _, name, data in _frames(response):
                    if name == "meta":
                        conv = data["conversation_id"]
                    if name == "delta":
                        seen += 1
                        if seen == 2:
                            break
            assert conv and seen == 2
            messages = _wait_for_assistant(client, base, conv)
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    assistant = messages[1]
    assert assistant["finish_reason"] == "client_disconnect"
    assert assistant["content"] == "Theo tài liệu "
    assert len(assistant["content"]) < len("Theo tài liệu [1], câu trả lời là vậy.")


def test_the_history_window_keeps_the_newest_turns_not_the_oldest(
    database: Engine, workspace: Path
) -> None:
    """⭐ `MAX_HISTORY_MESSAGES` cắt ở đầu nào?

    `ORDER BY created_at ASC LIMIT 10` là câu SQL người ta viết ra trước, và nó
    giữ 10 message **đầu tiên** của hội thoại. Hệ thống vẫn chạy, prompt vẫn có
    lịch sử, và cuộc trò chuyện càng dài thì trợ lý càng trả lời như thể nó vẫn
    đang ở lượt thứ hai. Không có gì đỏ ở đâu cả.

    Cách kiểm duy nhất từ bên ngoài: bắt LLM giả đọc ngược lại phần lịch sử nó
    nhận được.
    """
    proc, base = _serve(workspace, CHAT_TEST_MODE="echo_history", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=60.0) as client:
            _, frames = _chat(client, base, "lượt 1")
            conv = frames[0][2]["conversation_id"]
            _wait_for_assistant(client, base, conv)
            for n in range(2, 9):
                _chat(client, base, f"lượt {n}", conversation_id=conv)
                _wait_history_len(client, base, conv, 2 * n)
            _, last = _chat(client, base, "lượt cuối", conversation_id=conv)
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    echoed = "".join(d["text"] for _, name, d in last if name == "delta")
    size, first = echoed.split(";", 1)

    assert size == "n=10", f"cửa sổ lịch sử phải đúng 10 message, nhận {size}"
    # 8 lượt = 16 message; 10 cái mới nhất bắt đầu từ câu hỏi của lượt 4.
    assert first == "first=lượt 4", f"cắt nhầm đầu: {first!r} (ASC LIMIT sẽ cho 'lượt 1')"


def _wait_history_len(client: httpx.Client, base: str, conv: str, want: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"{base}/conversations/{conv}", headers=_headers()).json()
        if len(body["messages"]) >= want:
            return
        time.sleep(0.05)
    raise AssertionError(f"hội thoại không đạt {want} message sau 10 s")


# ---------------------------------------------------------------------------
# `W4-07` — hiểu câu hỏi, trên đường thật
# ---------------------------------------------------------------------------


def test_a_greeting_reaches_the_model_without_reaching_qdrant(server: str) -> None:
    """DoD: `"hello"` không gọi retrieval.

    ⭐ Đo bằng khung `sources` **rỗng**, không bằng một cờ nội bộ: đó là thứ duy
    nhất một client thấy được, và nó cũng là thứ chứng minh không có chunk nào
    bị nhét vào prompt. Model vẫn được gọi — một lời chào vẫn phải được chào lại.
    """
    with httpx.Client(timeout=30.0) as client:
        _, frames = _chat(client, server, "hello")
    kinds = {name: data for _, name, data in frames}
    assert kinds["meta"]["route"] == "no_retrieval"
    assert kinds["meta"]["language"] == "en"
    assert kinds["sources"]["sources"] == []
    assert kinds["done"]["finish_reason"] == "stop"
    assert "".join(d["text"] for _, n, d in frames if n == "delta")


def test_a_question_with_nothing_to_retrieve_never_reaches_the_model(server: str) -> None:
    """⭐⭐ Nhánh `CLARIFY` — nhánh duy nhất trả lời mà **không** gọi model nào.

    Bằng chứng là `model: null` trong khung `done`: LLM giả của bộ test luôn
    khai `scripted-model-served`, nên chuỗi ấy vắng mặt chỉ có một cách xảy ra.
    """
    with httpx.Client(timeout=30.0) as client:
        _, frames = _chat(client, server, "cái đó thì sao?")
        conv = next(d for _, n, d in frames if n == "meta")["conversation_id"]
        messages = _wait_for_assistant(client, server, conv)

    kinds = {name: data for _, name, data in frames}
    assert kinds["meta"]["route"] == "clarify"
    assert kinds["done"]["finish_reason"] == "clarify"
    assert kinds["done"]["model"] is None
    assert kinds["done"]["usage"] == {}
    assert kinds["sources"]["sources"] == []

    answer = "".join(d["text"] for _, n, d in frames if n == "delta")
    assert answer.startswith("Câu hỏi chưa đủ rõ")
    # Vẫn là một lượt thật: nó nằm trong lịch sử, với đủ nguồn gốc.
    assert messages[1]["content"] == answer
    assert messages[1]["finish_reason"] == "clarify"
    assert messages[0]["route"] == "clarify"


def test_the_same_question_after_a_first_turn_is_rewritten_before_retrieval(
    database: Engine, workspace: Path
) -> None:
    """⭐⭐ DoD: `"cái đó thì sao?"` thành một câu độc lập — và nó phải đi tới **Qdrant**.

    `meta.question` một mình không chứng minh được điều đó: nó chỉ nói kế hoạch
    ghi gì. Bằng chứng nằm ở mode `echo_prompt`, chỗ LLM giả đọc ngược lại lượt
    người dùng cuối cùng — trong đó khối NGỮ CẢNH do `SlowRetriever` dựng chứa
    **nguyên văn** chuỗi truy vấn nó nhận được.

    Cùng lúc đó, phép kiểm thứ hai chạy: câu hỏi đưa cho model vẫn là câu **gốc**.
    Truy hồi cần một chuỗi tự đủ nghĩa; model thì đã có lịch sử ở ngay trên và
    cần thấy đúng thứ người dùng vừa gõ.
    """
    rewritten = "Báo cáo WDR 2023 nói gì về di cư lao động?"
    proc, base = _serve(
        workspace,
        CHAT_TEST_MODE="echo_prompt",
        CHAT_TEST_DELTA_MS="1",
        CHAT_TEST_REWRITE=rewritten,
    )
    try:
        with httpx.Client(timeout=60.0) as client:
            _, first = _chat(client, base, "WDR 2023 nói gì về di cư?")
            conv = first[0][2]["conversation_id"]
            _wait_for_assistant(client, base, conv)
            _, second = _chat(client, base, "cái đó thì sao?", conversation_id=conv)
            _wait_history_len(client, base, conv, 4)
            messages = client.get(f"{base}/conversations/{conv}", headers=_headers()).json()[
                "messages"
            ]
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    meta = next(d for _, n, d in second if n == "meta")
    assert meta["rewritten"] is True
    assert meta["question"] == rewritten

    echoed = "".join(d["text"] for _, n, d in second if n == "delta")
    assert f"nói về {rewritten}" in echoed, "truy hồi chạy bằng câu gốc, không phải câu viết lại"
    assert "CÂU HỎI: cái đó thì sao?" in echoed, "model phải thấy câu người dùng đã gõ"
    assert f'(Hiểu đầy đủ theo hội thoại: "{rewritten}")' in echoed, (
        "model cũng phải thấy bản đã giải nghĩa — nếu không nó từ chối trả lời chuỗi mơ hồ"
    )

    # Và cả hai chuỗi được lưu lại, tách bạch — xem migration `0003`.
    assert messages[2]["content"] == "cái đó thì sao?"
    assert messages[2]["rewritten_query"] == rewritten
    assert messages[2]["route"] == "retrieve"


def test_a_self_contained_question_is_not_rewritten_even_with_history(
    database: Engine, workspace: Path
) -> None:
    """Ngược lại thì mỗi lượt từ thứ hai trở đi tốn thêm một lượt gọi LLM — và
    bộ viết lại ở đây trả một chuỗi cố định, nên nếu nó chạy thì nó sẽ **thay**
    câu hỏi thật bằng chuỗi ấy."""
    proc, base = _serve(
        workspace,
        CHAT_TEST_MODE="echo_prompt",
        CHAT_TEST_DELTA_MS="1",
        CHAT_TEST_REWRITE="CÂU BỊ THAY",
    )
    try:
        with httpx.Client(timeout=60.0) as client:
            _, first = _chat(client, base, "lượt một nói về nghèo đói")
            conv = first[0][2]["conversation_id"]
            _wait_for_assistant(client, base, conv)
            _, second = _chat(
                client,
                base,
                "Chi tiêu công cho giáo dục của Indonesia là bao nhiêu?",
                conversation_id=conv,
            )
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    meta = next(d for _, n, d in second if n == "meta")
    assert meta["rewritten"] is False
    echoed = "".join(d["text"] for _, n, d in second if n == "delta")
    assert "CÂU BỊ THAY" not in echoed


def test_the_answer_language_is_measured_not_only_requested(
    database: Engine, workspace: Path
) -> None:
    """⭐⭐ Chỉ dẫn ngôn ngữ vẫn chỉ là chỉ dẫn — nhưng từ đây nó **đếm được**.

    LLM giả luôn trả lời tiếng Việt. Hỏi bằng tiếng Anh thì chỉ thị `"Answer in
    English."` có mặt trong prompt (kiểm bằng `echo_prompt`), model vẫn phớt lờ
    nó, và `done.language_mismatch` phải là `true`.

    Đó chính là ca đã xảy ra ở lần chạy thật của `W4-06`, chỉ khác một điều: khi
    ấy không có gì ghi lại rằng nó đã xảy ra.
    """
    proc, base = _serve(workspace, CHAT_TEST_MODE="ok", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=60.0) as client:
            _, english = _chat(client, base, "What is the extreme poverty line?")
            _, vietnamese = _chat(client, base, "Ngưỡng nghèo cùng cực là bao nhiêu?")
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    en_done = next(d for _, n, d in english if n == "done")
    vi_done = next(d for _, n, d in vietnamese if n == "done")
    assert next(d for _, n, d in english if n == "meta")["language"] == "en"
    assert en_done["language_mismatch"] is True, "hỏi tiếng Anh, đáp tiếng Việt — phải bị ghi lại"
    assert vi_done["language_mismatch"] is False


def test_the_language_directive_reaches_the_prompt(database: Engine, workspace: Path) -> None:
    proc, base = _serve(workspace, CHAT_TEST_MODE="echo_prompt", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=60.0) as client:
            _, english = _chat(client, base, "What is the extreme poverty line?")
            _, unknown = _chat(client, base, "GDP per capita?")
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    assert "Answer in English." in "".join(d["text"] for _, n, d in english if n == "delta")
    echoed = "".join(d["text"] for _, n, d in unknown if n == "delta")
    # ⭐ Không phát hiện được ngôn ngữ ⇒ **không** chỉ thị nào cả. Đoán ở đây là
    # cách ép một người đang gõ tiếng Việt không dấu phải đọc câu trả lời tiếng Anh.
    assert "Answer in English." not in echoed
    assert "Trả lời bằng tiếng Việt." not in echoed


def test_a_cross_tenant_filter_is_still_403_on_a_turn_that_never_retrieves(server: str) -> None:
    """⚠️ `tenant_filter()` **từ chối**, nó không chỉ thu hẹp.

    Bỏ nó ở nhánh `no_retrieval` thì cùng một request nhận `403` hay `200` tuỳ
    theo người dùng có chào hỏi hay không — tức hành vi bảo mật phụ thuộc vào bộ
    phân loại câu hỏi, và nó thành một chỗ dò xem tenant nào tồn tại.
    """
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{server}/chat",
            json={"message": "hello", "filters": {"tenant_id": "globex"}},
            headers=_headers(),
        )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# `W4-08` — bộ định tuyến LLM, nhìn từ ngoài qua SSE
# ---------------------------------------------------------------------------


def test_a_dead_primary_is_invisible_to_the_client(database: Engine, workspace: Path) -> None:
    """Nhánh chính chết **trước** token đầu ⇒ người dùng không thấy gì khác.

    ⭐ Nhưng hoá đơn và model thì khác, nên bằng chứng phải là `done.model`:
    `200 OK` cộng một câu trả lời trôi chảy là chính xác thứ mà một fallback
    thành công trông giống — và cũng là thứ mà một fallback **im lặng hỏng**
    trông giống, nếu không ai nhìn tên model.
    """
    proc, base = _serve(workspace, CHAT_TEST_ROUTER="fallback", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=60.0) as client:
            _, frames = _chat(client, base, "câu hỏi thật về nghèo đói")
            status = client.get(f"{base}/admin/llm", headers=_headers(ADMIN_KEY)).json()
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    done = next(d for _, n, d in frames if n == "done")
    assert done["finish_reason"] == "stop"
    assert done["model"] == "scripted-model-served", "câu trả lời phải đến từ nhánh dự phòng"
    assert "".join(d["text"] for _, n, d in frames if n == "delta").startswith("Theo ")
    assert "CHÍNH-" not in "".join(d["text"] for _, n, d in frames if n == "delta")

    primary = status["routes"][0]
    assert primary["label"] == "primary"
    assert primary["consecutive_failures"] == 1, "cầu dao phải đếm được lần hỏng này"


def test_a_primary_that_dies_mid_stream_becomes_an_error_frame_not_a_spliced_answer(
    database: Engine, workspace: Path
) -> None:
    """⭐⭐ Ranh giới mẩu đầu tiên, đo trên đường thật.

    Nhánh chính gửi hai mẩu rồi chết. Nếu router chuyển nhánh ở đây, client nhận
    `CHÍNH-0 CHÍNH-1 Theo tài liệu [1]…` — một đoạn văn trôi chảy mà **không
    model nào nói ra**, kèm `finish_reason: stop`. Không có gì trong phản hồi
    nói rằng nó đã xảy ra.
    """
    proc, base = _serve(workspace, CHAT_TEST_ROUTER="midstream", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=60.0) as client:
            _, frames = _chat(client, base, "câu hỏi thật về nghèo đói")
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    text = "".join(d["text"] for _, n, d in frames if n == "delta")
    kinds = [n for _, n, _ in frames]

    assert text == "CHÍNH-0 CHÍNH-1 ", "phần đã sinh giữ nguyên, không nối thêm của nhánh khác"
    assert "Theo " not in text, "nhánh dự phòng KHÔNG được nối vào giữa stream"
    assert "error" in kinds and "done" not in kinds
    frame = next(d for _, n, d in frames if n == "error")
    # ⚠️ Trước `NEW-08`/`AU-03` bài này đọc LỜI của router trong `detail`
    # ("không ai nói") để biết đúng đường lỗi đã bắn — tức nó dựa vào chính
    # chỗ rò mà AU-03 bịt. Đường lỗi giờ chứng minh bằng HÀNH VI (ba assertion
    # trên: text không bị nối, không done), còn detail phải KÍN: lời của
    # router và tên lớp lỗi ở lại log server.
    assert "không ai nói" not in frame["detail"]
    assert "LLMError" not in frame["detail"]
    assert frame["trace_id"], "client cần trace_id để báo lỗi có địa chỉ"


def test_an_exhausted_daily_budget_is_429_before_a_single_byte_is_sent(
    database: Engine, workspace: Path
) -> None:
    """⭐⭐ "Từ chối có thông báo rõ" là một **HTTP status**, không phải một dòng
    SSE dừng lại.

    Trần chi phí kiểm được **trước** truy hồi, nên nó còn thành `429` kèm
    `Retry-After` đọc được bằng máy — cùng đường phân giới mà `W4-06` đã dựng.
    Để nó rơi vào trong stream thì client nhận `200 OK` và một dòng token trống.
    """
    proc, base = _serve(workspace, CHAT_TEST_ROUTER="broke", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base}/chat", json={"message": "câu hỏi bất kỳ"}, headers=_headers()
            )
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
    assert "trần" in response.json()["detail"]


def test_the_llm_status_endpoint_needs_the_admin_scope(database: Engine, workspace: Path) -> None:
    """`W4-04` ép scope theo **tiền tố đường dẫn** ở middleware, nên một route
    admin mới không thể quên hàng rào. Test này là phép kiểm rằng điều đó vẫn
    đúng cho một route vừa thêm ở file khác (`api/chat.py`, không `api/admin.py`)."""
    keys = workspace / "no-admin-keys.json"
    write_keys(
        keys,
        {
            digest_of("rag_reader_key"): {
                "tenant_id": "acme",
                "key_id": "reader",
                "scopes": [],
                "rate_limit_per_minute": 1000,
            }
        },
    )
    proc, base = _serve(workspace, CHAT_TEST_KEYS=str(keys))
    try:
        with httpx.Client(timeout=30.0) as client:
            denied = client.get(
                f"{base}/admin/llm", headers={"Authorization": "Bearer rag_reader_key"}
            )
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    assert denied.status_code == 403


# ---------------------------------------------------------------------------
# 8. `W4-09` — khung citations qua HTTP thật
# ---------------------------------------------------------------------------


def test_citations_are_verified_and_the_block_never_reaches_the_client(
    database: Engine, workspace: Path
) -> None:
    """Một quote thật + một quote bịa, marker cắt đôi giữa hai delta.

    Ba điều phải đúng cùng lúc qua một stream HTTP thật: khung `citations` đứng
    trước `done` và phân biệt thật/bịa; không ký tự nào của block lọt vào bất kỳ
    khung `delta` nào; và bản ghi Postgres đúng bằng những gì client đã thấy.
    """
    proc, base = _serve(workspace, CHAT_TEST_MODE="citations", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=30.0) as client:
            response, frames = _chat(client, base, "nguồn nào nói vậy?")
            conv = frames[0][2]["conversation_id"]
            messages = _wait_for_assistant(client, base, conv)
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    assert response.status_code == 200
    names = [name for _, name, _ in frames]
    assert names.index("citations") == len(names) - 2
    assert names[-1] == "done"

    citations = next(data for _, name, data in frames if name == "citations")
    assert citations["block"] == "ok"
    assert [c["verified"] for c in citations["citations"]] == [True, False]
    assert citations["citations"][0]["chunk_id"] == "chunk-1"
    assert citations["verified"] == 1

    deltas = "".join(data["text"] for _, name, data in frames if name == "delta")
    assert "CITAT" not in deltas
    assert deltas == "Trả lời [1] và [2]."
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert assistant["content"] == deltas


def test_an_answer_without_a_block_reports_absent_instead_of_pretending(
    database: Engine, workspace: Path
) -> None:
    """Mode "ok" là model đời `W4-06` — chưa từng nghe về block. Khung
    `citations` phải nói `absent` thay vì im lặng biến mất: model bỏ qua chỉ
    dẫn là một tín hiệu vận hành, không phải một ngày bình thường."""
    proc, base = _serve(workspace, CHAT_TEST_MODE="ok", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=30.0) as client:
            _, frames = _chat(client, base, "không có block")
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    citations = next(data for _, name, data in frames if name == "citations")
    assert citations["block"] == "absent"
    assert citations["citations"] == []


@pytest.mark.integration
def test_history_paginates_by_cursor_not_by_offset(database: Engine, workspace: Path) -> None:
    """`AU-08` vá ở `W6-06`: `GET /conversations/{id}` từng trả **toàn bộ**.

    ## ⭐⭐ Vì sao con trỏ theo id chứ không `OFFSET`

    `OFFSET n` phải đếm qua n hàng mỗi lần, và tệ hơn: một hàng chèn vào giữa
    hai lần gọi làm mọi trang sau **trượt đi một** — người đọc mất đúng một
    message, im lặng. Ở đây hàng mới *luôn* được chèn (task nền ghi câu trả lời
    sau khi stream xong), nên đó không phải một ca hiếm.

    Test này ghép lại toàn bộ hội thoại bằng cách đi trang, rồi so với lượt đọc
    một-phát: hai bản phải **giống hệt** về thứ tự lẫn nội dung.
    """
    proc, base = _serve(workspace, CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=60.0) as client:
            _, frames = _chat(client, base, "lượt 1")
            conv = frames[0][2]["conversation_id"]
            _wait_for_assistant(client, base, conv)
            for n in range(2, 6):
                _chat(client, base, f"lượt {n}", conversation_id=conv)
                _wait_history_len(client, base, conv, 2 * n)

            whole = client.get(f"{base}/conversations/{conv}", headers=_headers()).json()
            assert len(whole["messages"]) == 10
            assert whole["next_after"] is None, "trang chưa đầy thì không có trang sau"

            pages, cursor, walked = 0, None, []
            while True:
                params = {"limit": 3, **({"after": cursor} if cursor else {})}
                body = client.get(
                    f"{base}/conversations/{conv}", headers=_headers(), params=params
                ).json()
                walked.extend(body["messages"])
                pages += 1
                cursor = body["next_after"]
                if cursor is None:
                    break
                assert pages < 10, "vòng lặp phân trang không dừng"

            assert [m["id"] for m in walked] == [m["id"] for m in whole["messages"]]
            assert pages == 4, f"10 message chia trang 3 phải đi 4 lượt, đi {pages}"

            # ⚠️ Con trỏ lạ là 422, KHÔNG phải một trang rỗng: trang rỗng đọc y
            # hệt "hết lịch sử", và client vòng lặp sẽ dừng sớm mà tưởng xong.
            bad = client.get(
                f"{base}/conversations/{conv}", headers=_headers(), params={"after": "khongcothat"}
            )
            assert bad.status_code == 422

            # Trần trên `limit` — một client xin 10.000 vẫn là một payload vài MB.
            assert (
                client.get(
                    f"{base}/conversations/{conv}", headers=_headers(), params={"limit": 10_000}
                ).status_code
                == 422
            )
    finally:
        proc.terminate()
        proc.wait(timeout=20)
