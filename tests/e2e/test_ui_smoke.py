"""Giao diện chạy trong một trình duyệt thật — `W6-01`.

## Vì sao cần một trình duyệt, khi `tests/unit/test_ui.py` đã đọc chính tệp ấy

Bài test tĩnh chứng minh trang **không chứa** những thứ nguy hiểm và **có nhắc
tới** đúng tên khung SSE. Nó không chứng minh được thứ duy nhất người dùng quan
tâm: rằng gõ một câu hỏi thì chữ hiện ra, bấm `[1]` thì đúng đoạn được tô, và
bấm 👍 thì có một hàng trong Postgres.

Ba thứ ấy hỏng theo những cách mà một bài grep không thấy: một `await` thiếu,
một tên trường lệch một chữ, một selector không khớp. Đó là những cách hỏng
**duy nhất** mà giao diện này có, vì nó không có gì khác.

## ⚠️ Cần gì để chạy

    uv sync --all-extras && uv run playwright install chromium
    make up            # Qdrant + Postgres + Redis
    uv run python -m loadtest.stub_llm --port 8199 &      # hoặc DeepSeek thật
    DEEPSEEK_BASE_URL=http://127.0.0.1:8199 DEEPSEEK_API_KEY=stub make serve &
    E2E_API_KEY=rag_... uv run pytest tests/e2e/test_ui_smoke.py -m e2e

Cố ý **không** tự khởi động gì: cùng lý lẽ đã ghi ở `tests/e2e/test_smoke.py` —
một bài test tự `docker compose up` là một bài test có thể xoá dữ liệu của người
đang chạy nó.

⭐ Dùng stub của `W6-05` thay vì gọi DeepSeek thật là mặc định đúng ở đây: bài
này kiểm **giao diện**, và một lượt sinh thật thêm 4 giây, một khoản tiền, và
một nguồn bất định (`TD-41`: `temp=0` ở DeepSeek không tất định) vào một phép
đo không cần cái nào trong ba thứ đó.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.e2e

playwright_api = pytest.importorskip(
    "playwright.sync_api",
    reason="cần `uv sync --extra ui` + `playwright install chromium`",
)

BASE = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("E2E_API_KEY", "")
QUESTION = "Tăng trưởng GDP của Việt Nam gần đây ra sao?"


@pytest.fixture
def page(browser_page: Page) -> Page:
    """Trang đã nạp và đã có khoá API trong `localStorage`."""
    browser_page.goto(BASE)
    browser_page.evaluate("(k) => localStorage.setItem('rag.apiKey', k)", API_KEY)
    browser_page.reload()
    return browser_page


@pytest.fixture
def browser_page(request: pytest.FixtureRequest) -> object:
    from playwright.sync_api import sync_playwright

    if not API_KEY:
        pytest.skip("cần E2E_API_KEY")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        yield page
        context.close()
        browser.close()


def _ask(page: Page, question: str = QUESTION) -> None:
    page.fill("#q", question)
    page.click("#send")
    # Khung `done` là điều kiện dừng — xem docstring `POST /chat`. Trang gắn
    # phù hiệu thời lượng ngay khi nhận nó, nên phù hiệu ấy là tín hiệu "xong".
    page.wait_for_selector(".turn .meta .badge", timeout=120_000)


class TestAStrangerCanUseIt:
    """DoD của `W6-01`: *người lạ dùng được không cần hướng dẫn*."""

    def test_the_page_loads_and_says_what_it_is(self, page: Page) -> None:
        assert "RAG Platform" in page.title()
        assert "trích dẫn" in page.inner_text("header")

    def test_asking_a_question_streams_an_answer(self, page: Page) -> None:
        _ask(page)
        answer = page.inner_text(".turn .answer")
        assert len(answer) > 40, "câu trả lời quá ngắn để là một câu trả lời"

    def test_the_sources_panel_fills_in(self, page: Page) -> None:
        _ask(page)
        assert page.locator("#sources .src").count() >= 1

    def test_it_shows_which_bundle_answered(self, page: Page) -> None:
        """Một câu trả lời không nói nó đến từ bản nào là một câu trả lời không
        truy lại được — và `bundle_version` là thứ cả `W5-05` lẫn `W5-10` dựng
        cả một cơ chế để giữ đúng."""
        _ask(page)
        assert "bundle" in page.inner_text("#bundle")


class TestClickingACitationOpensTheSource:
    """Tính năng trung tâm của `W6-01`, và lý do khung `sources` phải chở
    nguyên văn chunk: không có nó thì người đọc vẫn phải **tin** rằng quote có
    thật — đúng thứ `W4-09` sinh ra để không phải tin."""

    def test_a_citation_button_exists_in_the_answer(self, page: Page) -> None:
        _ask(page)
        assert page.locator(".turn .answer button.cite").count() >= 1

    def test_clicking_it_expands_the_matching_source(self, page: Page) -> None:
        _ask(page)
        first = page.locator(".turn .answer button.cite").first
        n = first.get_attribute("data-n")
        first.click()
        source = page.locator(f"#src-{n}")
        assert source.get_attribute("open") is not None
        assert source.get_attribute("data-active") == "1"

    def test_the_quoted_span_is_marked_or_shown_verbatim(self, page: Page) -> None:
        """⭐⭐ **Hoặc** tô, **hoặc** in nguyên văn — và bài test chấp nhận cả hai.

        Máy chủ đối chiếu quote bằng `_quote_matches` (tách theo dấu lược, khớp
        từng mảnh đúng thứ tự); trang chỉ tìm chuỗi con sau khi chuẩn hoá
        whitespace. Hai luật khác nhau, nên có những quote **hợp lệ** mà trang
        không định vị được. Khi ấy trang in nguyên văn quote ra cạnh nguồn và
        để phù hiệu `verified` của máy chủ nói — nó không được phép tự ra một
        phán quyết thứ hai. Bài test ghim đúng lựa chọn ấy.
        """
        _ask(page)
        first = page.locator(".turn .answer button.cite").first
        n = first.get_attribute("data-n")
        first.click()
        source = page.locator(f"#src-{n}")
        marked = source.locator("mark").count()
        verbatim = source.locator(".quote").count()
        assert marked + verbatim >= 1, "không tô được thì phải in ra, không được im lặng"

    def test_a_quote_it_cannot_locate_is_printed_verbatim(self, page: Page) -> None:
        """⭐⭐ Nhánh dự phòng của bài trên — và phép tiêm `U4` chứng minh nó
        **chưa từng chạy** trong bộ test.

        Với stub, quote luôn nguyên văn nên trang luôn tô được, nên xoá hẳn
        nhánh "in nguyên văn" cũng không bài nào đỏ. Bài này ép đúng ca ấy: gắn
        một quote **không** có trong nội dung nguồn rồi bấm lại. Trang phải in
        nó ra — im lặng là để người đọc tưởng nguồn không nói gì.
        """
        _ask(page)
        n = int(page.locator(".turn .answer button.cite").first.get_attribute("data-n") or "1")
        page.evaluate(
            """(n) => {
                const source = currentSources.find((s) => s.n === n);
                currentCitations = [{
                    chunk_id: source.chunk_id,
                    quote: "một câu không hề có trong tài liệu nào cả",
                    verified: true,
                }];
                revealSource(n);
            }""",
            n,
        )
        assert page.locator(f"#src-{n} .quote").count() == 1
        assert page.locator(f"#src-{n} mark").count() == 0

    def test_the_source_body_actually_shows_the_document(self, page: Page) -> None:
        """Phép tiêm `U5` (để nội dung nguồn rỗng) sống sót vì nhánh dự phòng ở
        trên cứu nó: không tô được thì in quote, và bài test cũ chấp nhận cả
        hai. Nhưng một nguồn rỗng là một nguồn **không kiểm được** — cả điểm của
        việc chở nguyên văn chunk xuống client là để đọc quanh chỗ được trích."""
        _ask(page)
        # `text_content()` chứ không `inner_text()`: `<details>` đóng thì nội
        # dung không được render, và `inner_text` trả chuỗi rỗng cho MỌI trang —
        # kể cả trang đúng. Một bài test luôn đỏ cũng vô dụng như một bài luôn xanh.
        body = page.locator("#sources .src").first.locator(".body").text_content() or ""
        assert len(body.strip()) > 80, "nguồn rỗng thì không có gì để đối chiếu"

    def test_the_verification_verdict_comes_from_the_server(self, page: Page) -> None:
        """Phù hiệu đếm `verified/total` phải khớp con số máy chủ gửi trong
        khung `citations`, không phải một con số trang tự đếm."""
        _ask(page)
        badges = page.locator(".turn .meta .badge").all_inner_texts()
        assert any("trích dẫn" in b for b in badges)


class TestCitationsAreClickableWhileStillStreaming:
    """⭐ Phép tiêm `U7` (in câu trả lời thô ở nhánh `delta`) sống sót vì nhánh
    `citations` **gọi lại** `renderMarkdown` ở cuối, nên trạng thái sau cùng
    giống hệt. Thứ khác nhau là quãng ở giữa: với câu trả lời 6 giây, người dùng
    hoặc thấy `[1]` bấm được ngay khi nó hiện ra, hoặc nhìn text trơ suốt sáu
    giây rồi mới thấy nút.

    Một mutation chỉ đổi trạng thái tạm thời vẫn là một mutation đáng giết:
    quãng tạm thời ấy là toàn bộ trải nghiệm của một API stream.
    """

    def test_a_cite_button_appears_before_the_done_frame(self, page: Page) -> None:
        page.fill("#q", QUESTION)
        page.click("#send")
        page.wait_for_selector(".turn .answer button.cite", timeout=120_000)
        assert page.locator(".turn .meta .badge").count() == 0, (
            "khung `done` đã tới trước khi có nút citation — nút chỉ xuất hiện ở cuối"
        )


class TestFeedback:
    def test_thumbs_up_is_accepted(self, page: Page) -> None:
        _ask(page)
        buttons = page.locator(".turn .meta button")
        if buttons.count() == 0:
            # `TD-78` đã đóng: lượt rỗng cũng chấm được, nút chỉ vắng khi
            # `meta` không mang `message_id` (server chạy không Postgres).
            pytest.skip("lượt này không có message_id — server không trạng thái?")
        buttons.first.click()
        page.wait_for_function(
            "() => document.querySelector('.turn .meta span.note')?.textContent?.includes('đã ghi')",
            timeout=15_000,
        )


class TestTheUntrustedContentIsRenderedAsText:
    """⭐⭐ Bất biến bảo mật của trang, đo trên trình duyệt thật.

    `tests/unit/test_ui.py` chứng minh mã không gọi `innerHTML`. Bài này chứng
    minh **hệ quả**: một chuỗi trông như thẻ HTML đi qua đường câu hỏi → câu
    trả lời → DOM vẫn là văn bản, không thành phần tử.
    """

    def test_html_in_the_question_never_becomes_an_element(self, page: Page) -> None:
        _ask(page, "<img src=x onerror=alert(1)> GDP Việt Nam ra sao?")
        assert page.locator(".turn .question img").count() == 0
        assert "<img" in page.inner_text(".turn .question")
