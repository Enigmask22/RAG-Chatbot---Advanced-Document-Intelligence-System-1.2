"""`W4-13` — smoke e2e qua compose thật, và ba câu của gate `G4`.

## Vì sao test này khác mọi test integration đã có

`tests/integration/` dựng app **trong tiến trình pytest** rồi trỏ nó vào các
service của compose. Nó kiểm được logic, và nó **không** kiểm được thứ duy nhất
`W4-13` thêm vào: rằng cái *image* chạy được, rằng cấu hình mạng của compose
đúng, rằng bundle nạp được từ đường dẫn bên trong container, và rằng GPU đi qua
được ranh giới container.

Mỗi thứ trong danh sách ấy đã hỏng ít nhất một lần ở một dự án nào đó theo kiểu
"chạy trên máy tôi": app đúng, image sai. Nên test ở đây **không** import
`serving`, không dựng app, không chạm SQLAlchemy. Nó chỉ nói HTTP với
`127.0.0.1:8000` — đúng những gì một người dùng thật có.

⚠️ Cần `make up-api` trước. Không tự khởi động compose: một test tự `docker
compose up` là một test có thể xoá dữ liệu của người đang chạy nó, và thời gian
nạp model (hàng chục giây) biến mọi lần chạy pytest thành một lần chờ.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.e2e

BASE = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000")
COMPOSE = ["docker", "compose", "-f", "infra/docker-compose.yml", "--profile", "api"]


def _admin_key() -> str:
    """Khoá có scope `admin`. **Khác** khoá chat, và đó là điều đáng ghim.

    `W4-04` tách scope nên một khoá demo phát cho người dùng không đọc được
    `/admin/bundle` — hai lần đỏ đầu tiên của bộ e2e này chính là điều đó, và
    chúng đúng: một khoá chat mà xem được cấu hình bundle là một lỗ phân quyền.
    """
    key = os.environ.get("E2E_ADMIN_KEY")
    if not key:
        pytest.skip("cần E2E_ADMIN_KEY (scope admin) — xem `tasks/w4-13-docker.md` §2")
    return key


def _key() -> str:
    """Khoá API đọc từ store thật — cùng file mà container mount đọc-only.

    Không sinh khoá mới ở đây: container nạp store **lúc khởi động**
    (`W4-09` đã trả giá cho bài học này bằng hai lần 401), nên một khoá mint
    sau đó là một khoá vô hình.
    """
    key = os.environ.get("E2E_API_KEY")
    if not key:
        pytest.skip("cần E2E_API_KEY — xem `plans/reports/tasks/w4-13-docker.md` §2")
    return key


def _require_api_container() -> None:
    """Chốt skip cho bài test **không** đi qua fixture `client`.

    ⚠️ Mọi bài e2e khác nhận `client`, và fixture ấy `skip` khi không có API. Bài
    kiểm phiên bản thư viện lại nói chuyện thẳng với `docker compose exec`, nên
    nó là bài duy nhất **đỏ** thay vì **skip** khi chưa `make up-api` — một
    `pytest` trần trên máy sạch báo hỏng vì một lý do không phải lỗi. Và cái giá
    thật không nằm ở màu: đây là phép kiểm duy nhất nối "cái đã đo" với "cái
    đang chạy", nên nó là bài tệ nhất để người ta học cách bỏ qua.

    ⭐ Phân biệt hai cảnh, chứ không nuốt cả hai: **không có container** là điều
    kiện môi trường ⇒ skip; **có container mà probe hỏng** là hỏng thật ⇒ để nó
    nổ. Một `except Exception: skip` sẽ giấu đúng chế độ hỏng mà `W5-01` đã gặp.
    """
    try:
        listed = subprocess.run(
            [*COMPOSE, "ps", "--status", "running", "-q", "api"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # docker không có / compose lỗi
        pytest.skip(f"không hỏi được compose ({exc}) — chạy `make up-api` trước")
    if not listed.stdout.strip():
        pytest.skip("container `api` không chạy — chạy `make up-api` trước")


@pytest.fixture(scope="module")
def client() -> Any:
    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        try:
            c.get("/health")
        except httpx.ConnectError:
            pytest.skip(f"không có API ở {BASE} — chạy `make up-api` trước")
        yield c


def _sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    frames: list[tuple[str, dict[str, Any]]] = []
    event = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: ") and event:
            frames.append((event, json.loads(line[6:])))
            event = None
    return frames


# ---------------------------------------------------------------------------
# 1. Image lên được và phục vụ được
# ---------------------------------------------------------------------------


def test_health_is_up(client: httpx.Client) -> None:
    assert client.get("/health").status_code == 200


def test_ready_means_the_bundle_actually_loaded(client: httpx.Client) -> None:
    """`/ready` 200 là câu "bundle đã nạp, Qdrant đếm được, migration đã chạy".

    Nếu image thiếu một thư viện runtime (`libgomp1` chẳng hạn) thì `/health`
    vẫn xanh còn câu này đỏ — đó chính là lý do healthcheck của compose trỏ vào
    `/ready` chứ không `/health`.
    """
    response = client.get("/ready")
    assert response.status_code == 200, response.text


def test_the_container_serves_the_bundle_it_was_measured_on(client: httpx.Client) -> None:
    """⭐⭐ Câu quan trọng nhất của cả file.

    Bundle `v0.2.0` được eval trên `…@cuda:L512:float16:n50`. Một container CPU
    dựng ra `…@cpu:…:float32:…` và phép kiểm danh tính `TD-38` từ chối nạp — trừ
    khi ai đó bật `BUNDLE_ALLOW_RUNTIME_DRIFT`, và khi ấy demo phục vụ một hệ
    thống KHÁC hệ thống mọi con số trong `plans/` nói về.

    Test này ghim rằng cửa thoát ấy **đang đóng**: không có drift nào được khai.
    """
    payload = client.get(
        "/admin/bundle", headers={"Authorization": f"Bearer {_admin_key()}"}
    ).json()
    detail = payload["active_detail"]
    assert detail is not None, "chưa nạp được bundle nào"
    assert detail.get("runtime_drift") in (None, {}), (
        f"container đang phục vụ một hệ thống KHÁC bundle đã eval: {detail.get('runtime_drift')}"
    )


# ---------------------------------------------------------------------------
# 2. `G4` câu 1 — /chat trả citation ĐÃ VERIFY
# ---------------------------------------------------------------------------


def test_chat_streams_an_answer_with_a_verified_citation(client: httpx.Client) -> None:
    """DoD của `W4-13` và câu đầu của `G4`, đo qua HTTP thật.

    "Có citation" chưa đủ — `W4-09` tồn tại vì một citation bịa trông y hệt một
    citation thật. Câu phải kiểm là **`verified: true`**: quote đã được đối
    chiếu với đúng chunk mà nó trỏ vào.
    """
    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {_key()}"},
        json={"message": "Tăng trưởng GDP của Việt Nam gần đây ra sao?"},
    )
    assert response.status_code == 200, response.text
    frames = _sse(response.text)
    names = [name for name, _ in frames]

    assert names[0] == "meta" and names[1] == "sources"
    assert "delta" in names and names[-1] == "done"
    assert "citations" in names, "không có khung citations — `W4-09` không chạy trong image"

    citations = next(data for name, data in frames if name == "citations")
    verified = [c for c in citations.get("citations", []) if c.get("verified")]
    assert verified, f"không citation nào verify được: {citations}"

    done = frames[-1][1]
    assert done["finish_reason"] in ("stop", "cache")


def test_an_unauthenticated_chat_is_refused(client: httpx.Client) -> None:
    """Cổng 8000 giờ mở ra khỏi tiến trình. `W4-03` ghi rằng "chưa có xác thực"
    chấp nhận được khi bind `127.0.0.1` và **không** chấp nhận được khi đóng
    Docker — đây là chỗ câu ấy được kiểm."""
    assert client.post("/chat", json={"message": "xin chào"}).status_code == 401


# ---------------------------------------------------------------------------
# 3. `G4` câu 2 và 3
# ---------------------------------------------------------------------------


def test_reloading_a_bundle_needs_no_rebuild(client: httpx.Client) -> None:
    """`G4` câu 2: đổi cấu hình chỉ bằng `POST /admin/bundle/reload`.

    Nạp lại **đúng version đang chạy** thay vì một version khác: kiểm được rằng
    đường nạp lại sống mà không cần dựng thêm một bundle thứ hai trong image, và
    không để lại trạng thái khác lúc bắt đầu.
    """
    admin = {"Authorization": f"Bearer {_admin_key()}"}
    current = client.get("/admin/bundle", headers=admin).json()["active"]
    response = client.post("/admin/bundle/reload", headers=admin, json={"version": current})
    assert response.status_code == 200, response.text
    assert client.get("/admin/bundle", headers=admin).json()["active"] == current
    assert client.get("/ready").status_code == 200


@pytest.mark.slow
def test_chat_history_survives_a_compose_restart(client: httpx.Client) -> None:
    """`G4` câu 3, và nó chỉ thật khi container **thật sự** bị khởi động lại.

    Postgres giữ dữ liệu qua volume; thứ dễ hỏng là phía API — một
    `conversation_id` chỉ sống trong bộ nhớ tiến trình sẽ đi qua mọi test khác
    và chết đúng ở đây.
    """
    key = _key()
    first = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {key}"},
        json={"message": "Lạm phát năm 2024 ở mức nào?"},
    )
    assert first.status_code == 200, first.text
    conversation_id = _sse(first.text)[0][1]["conversation_id"]

    subprocess.run([*COMPOSE, "restart", "api"], check=True, capture_output=True)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            if client.get("/ready").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)
    else:
        pytest.fail("API không sẵn sàng lại trong 300 s sau restart")

    history = client.get(
        f"/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert history.status_code == 200, history.text
    roles = [m["role"] for m in history.json()["messages"]]
    assert roles[:2] == ["user", "assistant"], roles


def test_the_container_runs_the_versions_the_lockfile_pins() -> None:
    """⭐⭐ `runtime_drift: null` **không** đủ để nói "đúng hệ thống đã đo".

    Phép kiểm danh tính bundle (`TD-38`) soi tên model, thiết bị, dtype,
    max_length — nhưng không soi phiên bản thư viện. `W5-01` đo được image trôi
    lên `transformers 5.16.1 / sentence-transformers 6.0.1 / torch 2.14.0` trong
    khi lock ghim `5.15.0 / 5.7.0 / 2.13.0`, và suốt lúc đó `/admin/bundle` vẫn
    báo `runtime_drift: null`. Hậu quả: cross-encoder nạp ra dtype trộn, **mọi**
    request có truy hồi trả 503.

    Test này đọc lockfile và hỏi thẳng container. Nó là phép kiểm duy nhất trong
    dự án nối được "cái đã đo" với "cái đang chạy" ở mức thư viện.
    """
    import tomllib

    _require_api_container()

    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    locked: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        locked.setdefault(package["name"], set()).add(package["version"])

    probe = (
        "import json, torch, transformers, sentence_transformers as st; "
        "print(json.dumps({'torch': torch.__version__, "
        "'transformers': transformers.__version__, "
        "'sentence-transformers': st.__version__}))"
    )
    result = subprocess.run(
        [*COMPOSE, "exec", "-T", "api", "python", "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    running = json.loads(result.stdout.strip().splitlines()[-1])

    for name, version in running.items():
        # `torch` báo `2.13.0+cu126`; lock ghi cả `2.13.0` lẫn `2.13.0+cu126`.
        # So phần trước dấu `+` là đủ và không phụ thuộc index nào phục vụ wheel.
        base = version.split("+")[0]
        assert base in {v.split("+")[0] for v in locked[name]}, (
            f"container chạy {name} {version} nhưng uv.lock ghim "
            f"{sorted(locked[name])} — image đã trôi khỏi môi trường đã đo"
        )
