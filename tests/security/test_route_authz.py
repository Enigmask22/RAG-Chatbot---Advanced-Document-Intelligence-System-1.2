"""Mọi route của app thật đều được phân loại đúng — `W6-06`.

## ⭐⭐ Vì sao test này tồn tại, và nó thay thế cái gì

`admin.py` từng mang một docstring nói *"🔓 CHƯA CÓ XÁC THỰC — ai gọi cũng
được"*, kèm lời hứa rằng một test tên `test_admin_routes_are_still_open` sẽ đỏ
khi `W4-04` gắn auth vào. `W4-04` gắn auth **và** xoá test ấy; đoạn văn thì
không ai xoá. Nó sống thêm hai tuần như một tuyên bố **sai** về tình trạng bảo
mật của chính file mình, ở đúng chỗ người đọc tin nhất.

Bài học không phải "nhớ sửa docstring". Một dòng chữ mô tả *trạng thái* là một
bản sao thứ hai của sự thật (họ `AU-12`), và bản sao ấy không có gì bắt nó đồng
bộ. Nên chỗ này thay bằng một phép kiểm đọc **bảng route thật** của
`create_app()`: thêm một route mà không nghĩ về quyền là một test đỏ, không phải
một đoạn văn cũ.

⚠️ Kiểm bằng **request thật qua middleware**, không bằng cách đọc `PUBLIC_PATHS`
rồi so với chính nó. So một hằng số với chính nó là một test luôn xanh.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Route

from rag_core.settings import Settings
from serving.api.app import create_app
from serving.api.security import ADMIN_PREFIX, PUBLIC_PATHS
from serving.core.auth import digest_of
from serving.core.probes import ReadinessProbes

PLAIN_KEY = "test-plain"
ADMIN_KEY = "test-admin"

#: Method rẻ nhất để hỏi "middleware có chặn không". Thân request cố ý bỏ trống:
#: ta chỉ quan tâm mã 401/403, và 422 cũng chứng minh là **đã qua** được auth.
_PROBE_METHOD = {"GET": "get", "POST": "post"}


def _routes(app: Any) -> list[tuple[str, str]]:
    """(method, path) cho mọi route có thể gọi được, trừ HEAD/OPTIONS tự sinh.

    ⚠️⚠️ Phải đi **qua `original_router`**. `app.routes` của bản FastAPI này
    không trải phẳng router được `include_router`: nó giữ mỗi cái thành một
    `_IncludedRouter` — một object **không có** `.routes`, chỉ có
    `.original_router`. Một vòng `for route in app.routes` vì thế chỉ thấy 4
    route tài liệu tự sinh.

    Bản đầu của test này làm đúng thế và **xanh cả hai phép kiểm âm** — vì cả 4
    route ấy đều nằm ngoài `PUBLIC_PATHS` và đều trả 401. Một test đi qua vì
    *không tìm thấy gì để kiểm* trông y hệt một test đi qua vì *mọi thứ đều
    đúng*. Thứ tách được hai ca là một điều kiện **dương** — `assert
    admin_routes` và `PUBLIC_PATHS <= known` — và đó là thứ đã đỏ. Cùng họ với
    lượt tiêm giả của `W6-01`: màu xanh không phải bằng chứng, nó chỉ là màu.
    """
    out: list[tuple[str, str]] = []

    def walk(node: Any) -> None:
        for route in getattr(node, "routes", ()):
            if isinstance(route, Route) and route.path:
                for method in sorted(route.methods or ()):
                    if method in _PROBE_METHOD:
                        out.append((method, route.path))
            inner = getattr(route, "original_router", None) or route
            if inner is not route or not isinstance(route, Route):
                walk(inner)

    walk(app)
    return out


def _concrete(path: str) -> str:
    """Thay tham số đường dẫn bằng một giá trị bất kỳ — ta đo tầng auth, không
    đo handler, và auth chạy **trước** khi handler nhìn thấy giá trị nào."""
    out: list[str] = []
    for part in path.split("/"):
        out.append("x" if part.startswith("{") else part)
    return "/".join(out)


def _build(bundle: Any) -> tuple[Any, None]:
    """Runtime giả. `create_app` nhận nó qua tham số nên test không cần Qdrant
    lẫn GPU — cùng điểm nối mà `W4-02` dựng sẵn."""
    return object(), None


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> FastAPI:
    keys = tmp_path_factory.mktemp("w606") / "api-keys.json"
    keys.write_text(
        json.dumps(
            {
                digest_of(PLAIN_KEY): {"tenant_id": "t", "key_id": "plain", "scopes": []},
                digest_of(ADMIN_KEY): {"tenant_id": "t", "key_id": "adm", "scopes": ["admin"]},
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        bundle_root=Path(tmp_path_factory.mktemp("bundles")),
        api_keys_file=keys,
        log_level="WARNING",
        # Không bao giờ đi ra Internet, kể cả khi máy có key thật trong `.env`.
        chat_provider="none",
        chat_cache=False,
    )
    return create_app(
        settings=settings,
        build_runtime=_build,
        probe_factory=lambda _registry: ReadinessProbes(checks={}, ttl_s=0.0),
    )


@pytest.fixture(scope="module")
def anonymous(app: FastAPI) -> TestClient:
    """⚠️ `TestClient(app)` **không** chạy lifespan khi dùng ngoài `with` — cố ý:
    tầng auth chặn trước mọi handler, nên test này không cần bundle nào được nạp
    và không cần một dịch vụ nào chạy."""
    return TestClient(app)


@pytest.fixture(scope="module")
def chat_client(app: FastAPI) -> TestClient:
    return TestClient(app, headers={"Authorization": f"Bearer {PLAIN_KEY}"})


def test_every_route_is_either_public_or_behind_a_key(app: FastAPI, anonymous: TestClient) -> None:
    """Không có route nào ở giữa hai loại ấy.

    ⭐ Đây là phép kiểm mà mô hình "chặn theo mặc định" của `W4-04` hứa hẹn:
    quên nghĩ về một route mới nghĩa là nó **bị khoá**. Test này biến lời hứa
    thành thứ đo được trên bảng route thật.
    """
    routes = _routes(app)
    # ⭐ Điều kiện DƯƠNG trước điều kiện âm. Không có dòng này thì một bộ đi
    # bảng route hỏng làm cả phép kiểm thành "0 vi phạm trên 0 route".
    assert len(routes) >= 15, f"chỉ thấy {len(routes)} route — bộ đi bảng route hỏng"
    unclassified: list[tuple[str, str, int]] = []
    for method, path in routes:
        url = _concrete(path)
        status = getattr(anonymous, _PROBE_METHOD[method])(url).status_code
        public = path in PUBLIC_PATHS
        if public and status == 401:
            unclassified.append((method, path, status))
        if not public and status != 401:
            unclassified.append((method, path, status))
    assert not unclassified, (
        "route không khớp phân loại nào (công khai / cần khoá): "
        f"{unclassified}. Nếu đây là route mới, thêm nó vào PUBLIC_PATHS "
        "một cách có ý thức, hoặc để nó bị khoá."
    )


def test_admin_routes_need_the_admin_scope(app: FastAPI, chat_client: TestClient) -> None:
    """Khoá hợp lệ **không** admin phải nhận 403 ở mọi route dưới `/admin`.

    Quy tắc là **tiền tố đường dẫn**, không phải một dependency gắn tay từng
    route — nên test cũng phải suy ra danh sách từ tiền tố, không từ một danh
    sách chép tay sẽ lệch.
    """
    admin_routes = [(m, p) for m, p in _routes(app) if p.startswith(ADMIN_PREFIX)]
    assert admin_routes, "không tìm thấy route /admin nào — phép kiểm này đang đo hư không"
    for method, path in admin_routes:
        response = getattr(chat_client, _PROBE_METHOD[method])(_concrete(path))
        assert response.status_code == 403, (
            f"{method} {path} cho khoá thường: {response.status_code}"
        )


def test_docs_and_openapi_are_not_public(anonymous: TestClient) -> None:
    """`/docs` mô tả toàn bộ bề mặt tấn công — đắt hơn nhiều hai bit của `/ready`."""
    for path in ("/docs", "/openapi.json"):
        assert anonymous.get(path).status_code == 401, path


#: ⚠️⚠️ Bản sao **có chủ đích** của `PUBLIC_PATHS`, và là bản sao duy nhất trong
#: repo được phép tồn tại. Lý lẽ ngược với `AU-12` (một sự thật, một bản) vì đây
#: là một **allow-list bảo mật**: giá trị của nó nằm ở chỗ nới nó ra phải là hai
#: chữ ký ở hai file, chứ không phải một ký tự thêm vào một dòng.
#:
#: Phép tiêm thêm `/metrics` vào `PUBLIC_PATHS` **sống sót** trước khi có dòng
#: này: mọi phép kiểm còn lại đọc `PUBLIC_PATHS` làm chuẩn, nên nới nó ra là
#: nới luôn cả cái thước.
EXPECTED_PUBLIC = frozenset({"/", "/health", "/ready"})


def test_the_public_set_did_not_grow(app: FastAPI) -> None:
    """Thêm một đường dẫn công khai phải là một sửa đổi **có ý thức, hai chỗ**.

    `/` là trang tĩnh, `/health` và `/ready` lộ đúng hai bit (tiến trình sống,
    bundle đã nạp chưa). Bất cứ thứ gì khác đọc dữ liệu của tenant.
    """
    assert PUBLIC_PATHS == EXPECTED_PUBLIC, (
        "PUBLIC_PATHS đã đổi. Nếu đó là chủ ý, sửa EXPECTED_PUBLIC ở đây và nói "
        "rõ trong review vì sao endpoint ấy không lộ dữ liệu tenant."
    )


def test_no_public_route_needs_a_principal(app: FastAPI, anonymous: TestClient) -> None:
    """Điều kiện thực chất, không chỉ so hai hằng số: mỗi route công khai phải
    **trả lời được** mà không có khoá — chứ không phải 500 vì handler đi tìm
    `request.state.principal` mà middleware chưa đặt."""
    for path in sorted(PUBLIC_PATHS):
        status = anonymous.get(path).status_code
        # ⚠️ `!= 500`, không phải `< 500`: `/ready` trả **503** ở fixture này vì
        # chưa bundle nào được nạp, và đó là câu trả lời đúng của nó. Chế độ
        # hỏng đang canh là `principal_of()` ném `RuntimeError` — thứ middleware
        # dịch thành đúng 500.
        assert status != 500, f"{path} công khai nhưng nổ khi không có principal"


def test_the_public_list_is_exactly_what_it_claims(app: FastAPI) -> None:
    """Mọi đường dẫn trong `PUBLIC_PATHS` phải **tồn tại** trong bảng route.

    ⭐ Hướng ngược lại của phép kiểm đầu, và nó bắt một lỗi khác: một mục thừa
    trong danh sách công khai (route đã đổi tên, đã xoá) trông vô hại nhưng nó
    là một cái cửa mở sẵn cho một route tương lai vô tình trùng tên.
    """
    known = {path for _, path in _routes(app)}
    assert known >= PUBLIC_PATHS, f"PUBLIC_PATHS có mục không còn route: {PUBLIC_PATHS - known}"


# ---------------------------------------------------------------------------
# Quyền của workflow — bề mặt thứ hai, và nó không nằm trong mã Python
# ---------------------------------------------------------------------------

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _workflow(name: str) -> dict[str, Any]:
    import yaml

    return dict(yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8")))


def test_a_pull_request_workflow_declares_least_privilege() -> None:
    """⭐⭐ Không khai `permissions:` **không phải** là "quyền mặc định an toàn".

    Nó nghĩa là `GITHUB_TOKEN` nhận quyền mặc định của **repository** — một ô
    tick trong Settings, không phải một dòng trong repo: đổi được mà không để
    lại commit nào, và không có gì trong diff phản ánh. Với repo bật read/write
    thì mọi step của một workflow chạy trên `pull_request` — gồm cả ba action
    bên thứ ba — thấy một token ghi được vào repo.

    ⚠️ Phép kiểm là **có khai và chỉ đọc**, không phải "không có write": một
    khối `permissions:` vắng mặt và một khối `contents: read` là hai thứ khác
    hẳn nhau, mà nhìn từ xa thì cái nào cũng "không thấy write".
    """
    ci = _workflow("ci.yml")
    assert "permissions" in ci, "ci.yml không khai permissions — token nhận mặc định của repo"
    assert ci["permissions"] == {"contents": "read"}, (
        f"ci.yml xin quyền hơn mức cần: {ci['permissions']}"
    )


def test_no_fork_triggerable_workflow_can_see_a_secret() -> None:
    """`pull_request` chạy được từ fork. Một workflow như thế mà đọc `secrets.*`
    là đường rò credential kinh điển — và `pull_request_target` còn tệ hơn vì nó
    chạy **mã của base** với đủ quyền lẫn secret."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        raw = path.read_text(encoding="utf-8")
        # ⚠️ YAML 1.1 đọc `on:` thành boolean `True`, không phải chuỗi `"on"`.
        # Đó là lý do khoá tra ở đây là `True` — một trong những cái bẫy nổi
        # tiếng nhất của YAML, và nó im lặng: `.get("on")` trả `None` và vòng
        # lặp bên dưới không kiểm gì cả.
        parsed: dict[Any, Any] = _workflow(path.name)
        triggers = parsed[True]
        assert "pull_request_target" not in triggers, f"{path.name} dùng pull_request_target"
        if "pull_request" in triggers:
            assert "secrets." not in raw, f"{path.name} chạy trên PR mà vẫn đọc secrets"
