"""`GET /metrics` — `W5-07`.

Câu hỏi của module này là *"bản phơi bày có mang đủ khoá mà bảng Grafana hỏi
không"*, và nó chỉ trả lời được khi **cả hai** đầu được kiểm cùng lúc: mọi biểu
thức PromQL trong `infra/grafana/dashboards/rag-health.json` được bóc ra và đối
chiếu với những gì `/metrics` thật sự in ra.

## ⭐⭐ Vì sao phải đọc dashboard JSON chứ không viết lại danh sách metric

Một bài test liệt kê tay `["rag_http_requests_total", …]` chỉ chứng minh rằng
mã khớp với chính nó. Chế độ hỏng thật của một bảng điều khiển là **đổi tên một
metric rồi quên sửa bảng** — bảng vẫn mở được, panel vẫn vẽ, và nó vẽ một
đường thẳng ở 0 mà không ai phân biệt được với "hôm nay không có traffic".

Bài `test_every_metric_the_dashboard_asks_for_exists` là chỗ chế độ ấy chết.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_core.settings import Settings
from serving.api.app import create_app
from serving.core.auth import digest_of
from tests.integration.chat_app import write_keys
from tests.integration.test_bundle_reload import write_bundle
from tests.integration.test_tracing import (
    FakeLLM,
    _always_ready,
    _ask,
    _fake_runtime,
)

pytestmark = pytest.mark.integration

KEY = "rag_acme_metrics_key"
ADMIN_KEY = "rag_acme_metrics_admin"
DASHBOARD = Path("infra/grafana/dashboards/rag-health.json")

_METRIC_NAME = re.compile(r"\brag_[a-z_]+\b")
"""Tên metric trong một biểu thức PromQL. Chỉ họ `rag_` — `up`, `clamp_min`,
`histogram_quantile` là của Prometheus, không phải của chúng ta."""


@pytest.fixture(scope="module")
def metrics_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("metrics")
    write_bundle(root / "bundles", "0.2.0")
    write_keys(
        root / "api-keys.json",
        {
            digest_of(KEY): {
                "tenant_id": "acme",
                "key_id": "acme-metrics",
                "scopes": [],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(ADMIN_KEY): {
                "tenant_id": "acme",
                "key_id": "acme-metrics-admin",
                "scopes": ["admin"],
                "rate_limit_per_minute": 10_000,
            },
        },
    )
    return root


@pytest.fixture
def app(metrics_workspace: Path, database: Any) -> Iterator[TestClient]:
    settings = Settings(
        bundle_root=metrics_workspace / "bundles",
        bundle_version="0.2.0",
        api_keys_file=metrics_workspace / "api-keys.json",
        chat_cache=False,
        chat_rewrite=False,
        # ⚠️⚠️ Tường minh, không dựa mặc định: `quota_shared` mặc định `True`,
        # nên thiếu dòng này thì `create_app` nối Redis **khi Docker đang bật**
        # và `state.limiter` thành `RedisRateLimiter` (có `degraded`) — làm
        # `test_bo_dem_cuc_bo_thuan_thi_khong_khai_bua` xanh khi Docker tắt và
        # đỏ khi Docker bật. Chính bài test viết để ghim bài học "kết quả do
        # một thứ ngoài bài test quyết định" lại tái phạm đúng bài học ấy,
        # cùng ngày nó được gọi tên (08/09/2026, lần thứ năm).
        quota_shared=False,
    )
    api = create_app(
        settings=settings,
        build_runtime=_fake_runtime,
        probe_factory=lambda registry: _always_ready(),
    )
    with TestClient(api) as client:
        api.state.chat.llm = FakeLLM()
        # ⭐ Sink giả thay cho Langfuse: gauge `rag_trace_sink` chỉ tồn tại khi
        # có một sink biết khai `status()`. Không đặt nó ở đây thì bài kiểm hợp
        # đồng với bảng bỏ sót đúng ô "sức khoẻ của chính lớp quan sát" — tức
        # đúng ô nói cho ta biết quan sát có đang chạy hay không.
        api.state.trace_sink = _FakeSink()
        yield client


class _FakeSink:
    """Đứng thay `LangfuseSink` cho phần `status()`. Không gửi đi đâu cả."""

    def submit(self, trace: Any) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"host": "http://fake", "queued": 0, "sent": 7, "failed": 0, "dropped": 2}


def _scrape(client: TestClient) -> str:
    response = client.get("/metrics", headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200, response.text
    body: str = response.text
    return body


def _exposed(text: str) -> set[str]:
    """Mọi tên chuỗi thời gian có trong bản phơi bày, gồm cả hậu tố."""
    names: set[str] = set()
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        names.add(line.split("{", 1)[0].split(" ", 1)[0])
    return names


def _dashboard_metrics() -> set[str]:
    body = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    wanted: set[str] = set()
    for panel in body["panels"]:
        for target in panel.get("targets", []):
            wanted.update(_METRIC_NAME.findall(target["expr"]))
    return wanted


# ---------------------------------------------------------------------------
# 1. Hợp đồng với bảng Grafana
# ---------------------------------------------------------------------------


class TestDashboardContract:
    def test_every_metric_the_dashboard_asks_for_exists(self, app: TestClient) -> None:
        """Chế độ hỏng thật của một bảng: đổi tên metric, quên sửa bảng, và
        panel vẽ một đường thẳng ở 0 — thứ không phân biệt được với một ngày
        yên tĩnh."""
        _ask(app, key=KEY)
        exposed = _exposed(_scrape(app))
        missing = sorted(name for name in _dashboard_metrics() if name not in exposed)
        assert not missing, f"bảng hỏi metric không tồn tại: {missing}"

    def test_the_dashboard_declares_the_datasource_the_provisioning_creates(self) -> None:
        """Grafana sinh `uid` ngẫu nhiên cho datasource tạo qua giao diện; một
        dashboard commit vào git với uid ấy chỉ mở được trên đúng cái Grafana đã
        sinh ra nó."""
        body = json.loads(DASHBOARD.read_text(encoding="utf-8"))
        uids = {
            target.get("datasource", panel.get("datasource", {})).get("uid")
            for panel in body["panels"]
            for target in panel.get("targets", [{}])
        } | {p.get("datasource", {}).get("uid") for p in body["panels"] if p["type"] != "row"}
        assert uids - {None} == {"rag-prom"}
        provisioning = Path("infra/grafana/provisioning/datasources/prometheus.yml")
        assert "uid: rag-prom" in provisioning.read_text(encoding="utf-8")

    def test_the_budget_bucket_the_dashboard_counts_on_is_really_there(
        self, app: TestClient
    ) -> None:
        """Panel "% lượt vượt ngân sách" **đếm** thay vì nội suy, và nó chỉ đếm
        được nếu histogram có đúng bucket `le="3.5"` — con số ngân sách của bảng
        mục tiêu."""
        _ask(app, key=KEY)
        assert 'rag_chat_turn_duration_seconds_bucket{le="3.5"}' in _scrape(app)


# ---------------------------------------------------------------------------
# 2. Hai tầng RED
# ---------------------------------------------------------------------------


class TestTwoLayers:
    def test_a_rejected_request_is_counted_even_though_it_never_reached_chat(
        self, app: TestClient
    ) -> None:
        """⭐⭐ Bộ đếm dựng từ cây span chỉ thấy lượt đã **qua auth**. Không có
        tầng middleware thì một sự cố khoá API hiện ra là traffic bằng 0 — thứ
        trông y hệt một đêm yên tĩnh."""
        app.post("/chat", json={"message": "xin chào"}, headers={"Authorization": "Bearer sai"})
        text = _scrape(app)
        assert 'rag_http_requests_total{method="POST",route="/chat",status="401"}' in text
        assert "rag_chat_turns_total" not in _series_with(text, 'outcome="401"')

    def test_a_chat_turn_lands_in_both_layers(self, app: TestClient) -> None:
        _ask(app, key=KEY)
        text = _scrape(app)
        assert (
            _value(text, 'rag_http_requests_total{method="POST",route="/chat",status="200"}') == 1
        )
        assert _value(text, 'rag_chat_turns_total{outcome="stop"}') == 1

    def test_the_conversation_id_never_becomes_a_label(self, app: TestClient) -> None:
        """⚠️ Cách phổ biến nhất để giết một Prometheus, và nó không hỏng ngay
        mà hỏng sau vài tuần."""
        response = _ask(app, key=KEY)
        conversation = response.headers["X-Conversation-Id"]
        app.get(f"/conversations/{conversation}", headers={"Authorization": f"Bearer {KEY}"})
        text = _scrape(app)
        assert conversation not in text
        assert 'route="/conversations/{id}"' in text

    def test_no_label_anywhere_carries_a_tenant(self, app: TestClient) -> None:
        """`/metrics` mở được bằng **bất kỳ** khoá hợp lệ, kể cả khoá của tenant
        khác. Nhãn mang tenant nghĩa là mọi khách hàng đọc được danh sách khách
        hàng — dữ liệu kinh doanh rò qua một cửa không ai coi là cửa dữ liệu."""
        _ask(app, key=KEY)
        assert "acme" not in _scrape(app)


# ---------------------------------------------------------------------------
# 3. Số đo đọc từ cây span
# ---------------------------------------------------------------------------


class TestFromTheSpanTree:
    def test_each_stage_of_the_turn_becomes_a_histogram_series(self, app: TestClient) -> None:
        """Bảng và trace đọc **cùng một** phép đo — một span mới trong `chat.py`
        là một dòng mới trên bảng, không cần ai nhớ sửa chỗ thứ hai."""
        _ask(app, key=KEY)
        text = _scrape(app)
        for stage in ("understand", "retrieval", "retrieve.hybrid", "rerank", "completion"):
            assert f'rag_stage_duration_seconds_count{{stage="{stage}"}}' in text, stage

    def test_the_first_token_of_a_real_turn_lands_in_the_ttft_histogram(
        self, app: TestClient
    ) -> None:
        """Chốt **đường dây**, không chỉ phép đếm: `chat.py` gán `ttfb_ms` lúc
        token đầu rời đi, span `completion` mang nó tới `MetricsSink`, và SLO
        p95 ≤ 2 s (chốt 08/09/2026) đọc được từ `/metrics` thật. Unit test
        không thấy được việc nối dây này — cùng bài học `M1` của `TD-74`."""
        _ask(app, key=KEY)
        text = _scrape(app)
        assert _value(text, "rag_ttft_seconds_count") == 1
        assert 'rag_ttft_seconds_bucket{le="2.0"}' in text

    def test_only_the_outermost_retrieval_span_feeds_the_hit_histogram(
        self, app: TestClient
    ) -> None:
        """`retrieve.hybrid` mang `n_hits=50` — độ sâu pool rerank, không phải
        kết quả. Đếm nó vào đây thì "truy hồi rỗng" không bao giờ xảy ra và
        histogram số chunk lệch hẳn một bậc."""
        _ask(app, key=KEY)
        text = _scrape(app)
        assert _value(text, "rag_retrieval_hits_count") == 1
        assert _value(text, "rag_retrieval_hits_sum") == 5

    def test_cost_and_tokens_are_split_by_model_and_step(self, app: TestClient) -> None:
        _ask(app, key=KEY)
        text = _scrape(app)
        label = 'model="fake-model-served",step="completion"'
        assert _value(text, f"rag_llm_cost_usd_total{{{label}}}") == pytest.approx(0.000412)
        assert _value(text, f'rag_llm_tokens_total{{direction="in",{label}}}') == 1613

    def test_a_priced_turn_leaves_the_unpriced_counter_alone(self, app: TestClient) -> None:
        """Mẫu số của mọi ô tiền. Bộ đếm này khác 0 ⇒ tổng chi phí là một **cận
        dưới** chứ không phải một phép đo (xem `tracing.Usage`).

        Khẳng định là **bằng 0**, không phải "vắng mặt": xem
        `RagMetrics._declare_zero` — một metric vắng mặt cho Grafana vẽ
        *"No data"*, thứ mang cả nghĩa "chưa xảy ra" lẫn nghĩa "đã đổi tên".
        """
        _ask(app, key=KEY)
        text = _scrape(app)
        assert _value(text, 'rag_llm_unpriced_steps_total{step="completion"}') == 0
        assert _value(text, 'rag_llm_unpriced_steps_total{step="rewrite"}') == 0

    def test_the_trace_counter_tracks_the_turn_counter(self, app: TestClient) -> None:
        """Lệch nhau nghĩa là có lượt kết thúc mà không đóng trace — một đường
        thoát không ai biết."""
        for _ in range(3):
            _ask(app, key=KEY)
        text = _scrape(app)
        assert _value(text, "rag_traces_finished_total") == 3
        assert _value(text, 'rag_chat_turns_total{outcome="stop"}') == 3

    def test_a_failed_turn_is_counted_by_its_status_code_not_as_unknown(
        self, app: TestClient
    ) -> None:
        """Lượt chết trong `prepare()` chưa có `finish_reason` nào. Gán
        `"unknown"` cho nó là gộp mọi 404/403/429/503 vào một ô không tách ra
        được."""
        app.post(
            "/chat",
            json={"message": "xin chào", "conversation_id": "khongtontai"},
            headers={"Authorization": f"Bearer {KEY}"},
        )
        assert _value(_scrape(app), 'rag_chat_turns_total{outcome="404"}') == 1


# ---------------------------------------------------------------------------
# 4. Chính endpoint
# ---------------------------------------------------------------------------


class TestTheEndpointItself:
    def test_it_is_not_public(self, app: TestClient) -> None:
        """Quy ước Prometheus là `/metrics` mở, vì nó thường ở một cổng nội bộ.
        Ở đây không có cổng nội bộ nào — cùng tiến trình, cùng cổng 8000."""
        assert app.get("/metrics").status_code == 401

    def test_it_does_not_demand_the_admin_scope(self, app: TestClient) -> None:
        """Scraper là một tiến trình hạ tầng, không phải một người vận hành."""
        assert app.get("/metrics", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200

    def test_it_speaks_the_prometheus_text_format(self, app: TestClient) -> None:
        response = app.get("/metrics", headers={"Authorization": f"Bearer {KEY}"})
        assert response.headers["content-type"].startswith("text/plain")
        assert "# HELP rag_http_requests_total" in response.text
        assert "# TYPE rag_http_requests_total counter" in response.text

    def test_the_refusal_metric_says_in_its_help_that_it_is_an_estimate(
        self, app: TestClient
    ) -> None:
        """⭐⭐ `W5-02` đo từ chối bằng một **nhãn của judge** và nói thẳng vì sao
        không dùng từ khoá. Bảng trực tuyến không gọi judge được, nên nó dùng
        ước lượng — và điều đó phải nằm trong `HELP`, chỗ người đọc bảng lúc 3
        giờ sáng nhìn thấy, không phải trong một docstring."""
        text = _scrape(app)
        line = next(
            ln for ln in text.splitlines() if ln.startswith("# HELP rag_refusals_suspected")
        )
        assert "ƯỚC LƯỢNG" in line
        assert "W5-02" in line

    def test_the_trace_sink_queue_is_readable_from_the_scrape(self, app: TestClient) -> None:
        """Gauge làm mới **tại thời điểm scrape**: độ dài hàng đợi đúng lúc được
        hỏi, không đúng lúc một request nào đó tình cờ chạm vào nó."""
        text = _scrape(app)
        assert _value(text, 'rag_trace_sink{state="dropped"}') == 2
        assert _value(text, 'rag_trace_sink{state="sent"}') == 7

    def test_a_condition_that_has_not_happened_reads_as_zero_not_as_no_data(
        self, app: TestClient
    ) -> None:
        """⭐⭐ Một metric có nhãn KHÔNG tồn tại cho tới lần quan sát đầu tiên, và
        Grafana vẽ *"No data"* — ba chữ mang hai nghĩa: "chưa xảy ra lần nào"
        (tin tốt) và "metric đã đổi tên, bảng chưa sửa" (hỏng lặng lẽ). Khai
        trước ở 0 để chỉ còn nghĩa thứ hai."""
        text = _scrape(app)
        assert _value(text, 'rag_llm_unpriced_steps_total{step="rewrite"}') == 0
        assert _value(text, 'rag_cache_lookups_total{result="replay"}') == 0

    def test_the_bundle_version_is_a_label_on_a_gauge(self, app: TestClient) -> None:
        assert _value(_scrape(app), 'rag_bundle_info{version="0.2.0"}') == 1

    def test_scraping_twice_does_not_double_count(self, app: TestClient) -> None:
        """Gauge được làm mới **tại thời điểm scrape**; counter thì không được
        chạm. Một `/metrics` tự tăng bộ đếm của chính nó là một bảng đo tần suất
        scrape thay vì tần suất request."""
        _ask(app, key=KEY)
        first = _value(_scrape(app), 'rag_chat_turns_total{outcome="stop"}')
        assert _value(_scrape(app), 'rag_chat_turns_total{outcome="stop"}') == first


class TestWorkerGauge:
    """`TD-75`, quyết ở `W6-05`. `prometheus_client` giữ số đo trong bộ nhớ của
    MỘT tiến trình; bật worker thứ hai thì mỗi worker có sổ riêng và scraper hỏi
    trúng một cái ngẫu nhiên. Bảng tụt đi một nửa mà không có gì báo."""

    def test_the_exposition_states_how_many_workers_it_speaks_for(self, app: TestClient) -> None:
        assert _value(_scrape(app), "rag_scrape_workers") == 1.0

    def test_it_reads_the_same_variable_uvicorn_does(
        self, app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """⭐ Một nguồn sự thật. Hai chỗ khai riêng thì ai đó đổi số worker mà
        quên gauge, và gauge sẽ khai một giả định không còn đúng — tệ hơn không
        có gauge nào."""
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        assert _value(_scrape(app), "rag_scrape_workers") == 4.0

    def test_zero_workers_is_impossible_so_it_reports_one(
        self, app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bản phơi bày này tồn tại nghĩa là có ít nhất một worker đang chạy.
        Khai 0 biến mọi phép chia trong bảng thành chia cho 0."""
        monkeypatch.setenv("WEB_CONCURRENCY", "0")
        assert _value(_scrape(app), "rag_scrape_workers") == 1.0

    def test_a_junk_value_reports_one_instead_of_crashing_the_scrape(
        self, app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/metrics` hỏng nghĩa là mất toàn bộ bảng. Một biến môi trường rác
        không được phép mua cái giá đó."""
        monkeypatch.setenv("WEB_CONCURRENCY", "auto")
        assert _value(_scrape(app), "rag_scrape_workers") == 1.0


# ---------------------------------------------------------------------------
# tiện ích
# ---------------------------------------------------------------------------


def _series_with(text: str, fragment: str) -> Sequence[str]:
    return [line for line in text.splitlines() if fragment in line and not line.startswith("#")]


def _value(text: str, series: str) -> float:
    for line in text.splitlines():
        if line.startswith(f"{series} "):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"không thấy chuỗi {series!r} trong bản phơi bày")


class TestQuotaDegradedGauge:
    """`TD-39` — con số duy nhất phân biệt *"hạn mức đang đúng"* với *"hạn mức
    đang là N× số replica và không ai biết"*.

    Cùng lý lẽ với `rag_scrape_workers` ngay trên: bản phơi bày phải **tự nói ra
    giả định của mình**. Khác ở chỗ giả định này đổi **lúc chạy** — nó đúng cho
    tới đúng giây Redis hỏng.
    """

    def test_chua_tut_ve_lan_nao_thi_doc_ra_so_0(self, app: TestClient) -> None:
        text = _scrape(app)
        assert _value(text, 'rag_quota_degraded_total{counter="ratelimit"}') == 0.0
        assert _value(text, 'rag_quota_degraded_total{counter="daily_spend"}') == 0.0

    def test_bo_dem_cuc_bo_thuan_thi_khong_khai_bua(self, app: TestClient) -> None:
        """`RateLimiter` trong tiến trình **không có** thuộc tính `degraded`.
        Đọc `getattr(..., 0)` sẽ khai 0 như thể nó đang chạy đúng chế độ dùng
        chung — mà nó thì không dùng chung gì cả. Ở cấu hình ấy con số phải giữ
        nguyên 0 của `_declare_zero`, không phải một 0 do ai đó bịa ra."""
        assert not hasattr(app.app.state.limiter, "degraded")  # type: ignore[attr-defined]
        assert _value(_scrape(app), 'rag_quota_degraded_total{counter="ratelimit"}') == 0.0

    def test_so_lan_tut_ve_di_thang_ra_bang(self, app: TestClient) -> None:
        """⭐ Ghim **đường ánh xạ**, không ghim gauge. Một gauge được khai báo mà
        không ai `set()` vẫn in ra 0 mãi mãi — và bảng sẽ báo "mọi thứ bình
        thường" trong suốt một sự cố Redis."""

        class _Gia:
            degraded = 7

        app.app.state.limiter = _Gia()  # type: ignore[attr-defined]
        assert _value(_scrape(app), 'rag_quota_degraded_total{counter="ratelimit"}') == 7.0

    def test_hai_bo_dem_la_hai_chuoi_thoi_gian(self, app: TestClient) -> None:
        """Gộp chúng lại nghĩa là bảng không phân biệt được *"hạn mức đang N×"*
        với *"trần chi phí đang N×"* — hai sự cố khác nhau, hai người phải gọi
        khác nhau."""

        class _Gia:
            def __init__(self, n: int) -> None:
                self.degraded = n

        app.app.state.limiter = _Gia(3)  # type: ignore[attr-defined]
        app.app.state.daily_spend = _Gia(11)  # type: ignore[attr-defined]
        text = _scrape(app)
        assert _value(text, 'rag_quota_degraded_total{counter="ratelimit"}') == 3.0
        assert _value(text, 'rag_quota_degraded_total{counter="daily_spend"}') == 11.0
