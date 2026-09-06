"""Trang giao diện — `W6-01`.

Đây là test **tĩnh**: nó đọc chính tệp sẽ được ship. Phần hành vi (stream, bấm
citation, feedback) nằm ở `tests/e2e/test_ui_smoke.py` với trình duyệt thật.

## ⭐⭐ Vì sao có một bài test đọc mã JavaScript bằng regex

Bất biến quan trọng nhất của trang này là *"không bao giờ `innerHTML`"*, và nó
là bất biến vì trang dựng DOM từ hai nguồn không tin được: văn bản chunk corpus
(khung `sources` gắn cờ `flags` cho nó — `W4-12`) và câu trả lời của model, thứ
được sinh **ra từ** nội dung ấy.

Một bất biến chỉ tồn tại trong đầu người viết là một bất biến sẽ mất trong lần
sửa thứ ba. Một bài test grep thì thô, nhưng nó chạy trong 3 ms, không cần trình
duyệt, và nó đỏ đúng vào lúc ai đó thêm dòng đầu tiên vi phạm — kể cả khi dòng
ấy trông vô hại.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from serving.api import ui
from serving.api.ui import CSP, INDEX_PATH

PAGE = INDEX_PATH.read_text(encoding="utf-8")


def _script_without_comments(page: str) -> str:
    """Phần JavaScript, đã bỏ chú thích.

    ⚠️ Cần thiết chứ không phải cầu kỳ: docstring của chính trang **nhắc tên**
    những API bị cấm để giải thích vì sao chúng bị cấm, và một bài test grep thô
    sẽ đỏ vì đúng đoạn văn nói rằng chúng không được dùng. Bỏ chú thích trước
    khi quét là cách duy nhất để bài test đo *mã* thay vì đo *lời bàn về mã*.

    Cách bóc chú thích bằng regex nói chung là sai (nó không biết chuỗi và regex
    literal). Ở đây nó đúng vì đã kiểm: tệp không có `//` bên trong chuỗi nào,
    và không có regex literal nào chứa `/*`. Một tệp lớn hơn sẽ cần parser thật.
    """
    body = "".join(re.findall(r"<script[^>]*>(.*?)</script>", page, re.DOTALL))
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", " ", body)


CODE = _script_without_comments(PAGE)


@pytest.fixture
def ui_client() -> TestClient:
    """App tối thiểu chỉ có router giao diện.

    Cố ý **không** dựng app thật: trang là tĩnh và không phụ thuộc bundle,
    Postgres hay Qdrant. Kéo cả app vào đây sẽ đổi một bài test 3 ms thành một
    bài cần hạ tầng, và làm nó đỏ vì những lý do không liên quan tới giao diện.
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(ui.router)
    return TestClient(app)


class TestTheDomIsNeverBuiltFromStrings:
    """Xem docstring module."""

    @pytest.mark.parametrize(
        "forbidden",
        [
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "document.write",
            # `eval` và `new Function` không phải chỗ chèn HTML, nhưng chúng là
            # con đường thứ hai từ chuỗi tới mã, và cùng một nguồn dữ liệu.
            "eval(",
            "new Function(",
        ],
    )
    def test_the_page_does_not_contain(self, forbidden: str) -> None:
        assert forbidden not in CODE, (
            f"`{forbidden}` xuất hiện trong trang. Nội dung chunk corpus và câu "
            "trả lời của model đều đi qua đây; dựng DOM bằng chuỗi là mở một "
            "đường từ tài liệu tới mã chạy trong trình duyệt."
        )

    def test_it_does_build_nodes_the_safe_way(self) -> None:
        """Bài trên là một điều kiện **âm** và nó xanh cả khi trang rỗng. Cần
        một điều kiện dương đi kèm, nếu không thì "xoá hết JavaScript" cũng là
        một cách làm nó xanh."""
        assert "createTextNode" in CODE
        assert "createElement" in CODE
        assert CODE.count("textContent") > 10


class TestTheServedPage:
    def test_it_is_public_because_it_is_where_you_type_the_key(self, ui_client: TestClient) -> None:
        """Bài toán con gà–quả trứng: không có giao diện để nhập khoá thì không
        có cách nhập khoá. Trang là tĩnh; mọi lời gọi nó phát ra vẫn cần khoá."""
        response = ui_client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_it_ships_a_content_security_policy(self, ui_client: TestClient) -> None:
        """ "Không bao giờ `innerHTML`" là lời hứa của mã; CSP là hàng rào của
        trình duyệt. Hai thứ hỏng theo hai cách khác nhau, nên cần cả hai."""
        policy = ui_client.get("/").headers["content-security-policy"]
        assert policy == CSP
        assert "default-src 'none'" in policy
        assert "connect-src 'self'" in policy
        assert "frame-ancestors 'none'" in policy

    def test_nothing_is_loaded_from_another_host(self) -> None:
        """CSP đã chặn, nhưng một tham chiếu ngoài trong trang nghĩa là trang
        **cần** nó — tức trang sẽ hỏng lặng lẽ sau CSP thay vì chạy đúng."""
        externals = re.findall(r'(?:src|href)\s*=\s*"(https?:)?//[^"]+"', PAGE)
        assert externals == []

    def test_it_does_not_cache_across_deploys(self, ui_client: TestClient) -> None:
        """Một bản vá giao diện bị trình duyệt giữ lại chỉ hỏng với người **đã**
        mở trang trước đó — chế độ hỏng tốn hàng giờ để chẩn đoán."""
        assert ui_client.get("/").headers["cache-control"] == "no-cache"


class TestThePageKnowsWhatTheApiActuallySends:
    """Trang và máy chủ là hai nửa của một hợp đồng. Không có bài nào ở đây
    chứng minh trang *đúng* — nhưng chúng đỏ khi một bên đổi tên khung hoặc
    trường mà bên kia không biết."""

    @pytest.mark.parametrize("frame", ["meta", "sources", "delta", "citations", "done", "error"])
    def test_it_handles_every_sse_frame(self, frame: str) -> None:
        assert f'"{frame}"' in CODE

    def test_it_reads_the_answer_message_id_not_the_question_one(self) -> None:
        """`POST /feedback` chấm **câu trả lời**. Gửi nhầm `message_id` của câu
        hỏi thì hàng feedback gắn vào lượt sai và không ai thấy gì đỏ."""
        assert "answer_message_id" in CODE

    def test_it_disables_feedback_on_an_empty_answer(self) -> None:
        """`TD-78`: `_save()` bỏ qua câu trả lời rỗng, nên `answer_message_id`
        trỏ vào một hàng không tồn tại và `POST /feedback` trả 404 — đúng lượt
        đáng nhận 👎 nhất."""
        assert 'finishReason === "empty"' in CODE
        assert "TD-78" in PAGE

    def test_it_treats_end_of_stream_as_an_error_not_as_success(self) -> None:
        """Docstring của `POST /chat`: một dòng `delta` dừng lại giống hệt nhau
        khi model nói xong, khi kết nối đứt, và khi provider hết hạn mức."""
        assert "sawDone" in CODE

    def test_it_shows_the_injection_flags_instead_of_hiding_them(self) -> None:
        """`W4-12` cho cờ đi ra tới client chứ không chỉ vào log: người đọc câu
        trả lời là người duy nhất biết nó có bất thường hay không."""
        assert "s.flags" in CODE
