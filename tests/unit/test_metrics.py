"""Cây span → số đếm Prometheus — `W5-07`.

Phần thuần: không app, không Postgres, không container. `MetricsSink` nhận một
`Trace` dựng bằng tay và bài test đọc thẳng bản phơi bày — cùng lý lẽ đã dùng
cho `test_tracing.py`: phép đếm phải kiểm được ở chỗ nó sẽ thật sự được chạy
lại, tức trong `make test`.
"""

from __future__ import annotations

from typing import Any

import pytest

from serving.core.metrics import MetricsSink, RagMetrics, looks_like_refusal, route_label
from serving.core.tracing import FanoutSink, Trace, Usage


def _turn(
    *,
    outcome: str = "stop",
    hits: int = 5,
    cache: str | None = "miss",
    cost: float | None = 0.000412,
    verified: int = 2,
    claimed: int = 2,
    answer: str = "RRF hợp nhất thứ hạng [1].",
) -> Trace:
    """Một lượt hoàn chỉnh, cùng hình dạng span mà `chat.py` thật sự sinh ra."""
    trace = Trace(name="chat")
    with trace.span("understand"):
        pass
    if cache == "replay":
        trace.span("cache.replay").end()
    elif cache is not None:
        with trace.span("cache.lookup") as span, trace.span("embed.query"):
            span.end(hit=cache == "hit")
    with trace.span("retrieval") as outer:
        with trace.span("retrieve.hybrid") as inner:
            inner.end(n_hits=50)
        with trace.span("rerank"):
            pass
        outer.end(n_hits=hits)
    trace.span("prompt").end()
    generation = trace.span("completion", kind="generation")
    generation.end(
        output=answer,
        model="deepseek-v4-flash",
        usage=Usage(prompt_tokens=1613, completion_tokens=48, cost_usd=cost),
    )
    trace.span("citations").end(verified=verified, claimed=claimed)
    trace.metadata["finish_reason"] = outcome
    trace.finish(output=answer)
    return trace


def _text(metrics: RagMetrics) -> str:
    return metrics.render().decode("utf-8")


def _value(text: str, series: str) -> float:
    for line in text.splitlines():
        if line.startswith(f"{series} "):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"không thấy chuỗi {series!r}")


@pytest.fixture
def bag() -> RagMetrics:
    return RagMetrics()


@pytest.fixture
def sink(bag: RagMetrics) -> MetricsSink:
    return MetricsSink(bag)


# ---------------------------------------------------------------------------
# 1. Một lượt → những con số nào
# ---------------------------------------------------------------------------


class TestOneTurn:
    def test_every_span_becomes_a_histogram_series(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """Bảng và trace đọc **cùng một** phép đo. Một span mới trong `chat.py`
        là một dòng mới trên bảng, không cần ai nhớ sửa chỗ thứ hai."""
        sink.submit(_turn())
        text = _text(bag)
        for stage in ("understand", "retrieval", "retrieve.hybrid", "rerank", "completion"):
            assert _value(text, f'rag_stage_duration_seconds_count{{stage="{stage}"}}') == 1

    def test_the_outcome_label_comes_from_finish_reason(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        sink.submit(_turn(outcome="empty"))
        assert _value(_text(bag), 'rag_chat_turns_total{outcome="empty"}') == 1

    def test_a_turn_that_died_in_prepare_is_labelled_by_its_status_code(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """⭐ Lượt hỏng ở `prepare()` chưa có `finish_reason` nào — nó chết trước
        khi có một. Gán `"unknown"` là gộp mọi 404/403/429/503 vào một ô mà
        người vận hành không tách ra được."""
        trace = Trace(name="chat")
        trace.finish(level="ERROR", status="503 InterfaceError: Postgres chết")
        sink.submit(trace)
        assert _value(_text(bag), 'rag_chat_turns_total{outcome="503"}') == 1

    def test_cost_and_tokens_split_by_model_and_step(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        sink.submit(_turn())
        text = _text(bag)
        label = 'model="deepseek-v4-flash",step="completion"'
        assert _value(text, f"rag_llm_cost_usd_total{{{label}}}") == pytest.approx(0.000412)
        assert _value(text, f'rag_llm_tokens_total{{direction="in",{label}}}') == 1613
        assert _value(text, f'rag_llm_tokens_total{{direction="out",{label}}}') == 48

    def test_the_trace_counter_moves_with_the_turn_counter(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        for _ in range(3):
            sink.submit(_turn())
        assert _value(_text(bag), "rag_traces_finished_total") == 3


# ---------------------------------------------------------------------------
# 2. Chỗ dễ đếm nhầm
# ---------------------------------------------------------------------------


class TestCountingTraps:
    def test_only_the_outermost_retrieval_span_feeds_the_hit_histogram(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """`retrieve.hybrid` mang `n_hits=50` — độ sâu pool rerank, không phải
        kết quả. Đếm nó vào đây thì "truy hồi rỗng" **không bao giờ** xảy ra và
        histogram số chunk lệch hẳn một bậc."""
        sink.submit(_turn(hits=5))
        text = _text(bag)
        assert _value(text, "rag_retrieval_hits_count") == 1
        assert _value(text, "rag_retrieval_hits_sum") == 5

    def test_an_empty_retrieval_is_counted_as_empty(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        sink.submit(_turn(hits=0))
        assert _value(_text(bag), "rag_retrieval_empty_total") == 1

    def test_a_cache_replay_counts_toward_the_hit_rate(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """⭐ Một lượt phát lại từ cache **không** đi qua `cache.lookup` ở lượt
        NÀY — nó đi qua đó ở lượt trước. Không đếm riêng thì tỉ lệ trúng cache
        thiếu đúng phần mà cache đang phục vụ."""
        sink.submit(_turn(cache="replay", outcome="cache"))
        assert _value(_text(bag), 'rag_cache_lookups_total{result="replay"}') == 1

    def test_a_miss_and_a_hit_land_in_different_buckets(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        sink.submit(_turn(cache="miss"))
        sink.submit(_turn(cache="hit"))
        text = _text(bag)
        assert _value(text, 'rag_cache_lookups_total{result="hit"}') == 1
        assert _value(text, 'rag_cache_lookups_total{result="miss"}') == 1

    def test_an_answer_with_no_verified_citation_is_counted_apart(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        sink.submit(_turn(verified=0, claimed=0))
        text = _text(bag)
        assert _value(text, 'rag_answers_total{cited="no"}') == 1
        assert _value(text, 'rag_answers_total{cited="yes"}') == 0

    def test_a_claimed_citation_that_did_not_verify_is_counted(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """`W4-09`: một quote bịa không còn thành HTTP status được sau khi stream
        đã bắt đầu, nên nó phải thành con số này."""
        sink.submit(_turn(verified=1, claimed=3))
        assert _value(_text(bag), "rag_citations_unverified_total") == 2


# ---------------------------------------------------------------------------
# 3. "Chưa đo" ≠ "miễn phí"
# ---------------------------------------------------------------------------


class TestUnpriced:
    def test_a_generation_with_no_usage_increments_the_unpriced_counter(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """Mẫu số của mọi ô tiền. `rewrite` quá hạn vẫn bị nhà cung cấp tính
        tiền — chi phí thật là *chưa biết*, và cộng 0 cho nó biến một khoản chi
        không quan sát được thành một khoản chi bằng không."""
        trace = Trace(name="chat")
        trace.span("rewrite", kind="generation").end(level="WARNING", status="quá hạn")
        trace.finish()
        sink.submit(trace)
        assert _value(_text(bag), 'rag_llm_unpriced_steps_total{step="rewrite"}') == 1

    def test_a_priced_turn_leaves_it_at_zero(self, sink: MetricsSink, bag: RagMetrics) -> None:
        sink.submit(_turn())
        assert _value(_text(bag), 'rag_llm_unpriced_steps_total{step="completion"}') == 0

    def test_a_condition_that_never_happened_reads_as_zero_not_as_missing(
        self, bag: RagMetrics
    ) -> None:
        """⭐⭐ Một metric có nhãn KHÔNG tồn tại cho tới lần quan sát đầu tiên, và
        Grafana vẽ *"No data"*. Ba chữ ấy mang **hai** nghĩa — "chưa xảy ra lần
        nào" (tin tốt) và "metric đã đổi tên, bảng chưa sửa" (hỏng lặng lẽ) — mà
        bảng không phân biệt được. Khai trước ở 0 để chỉ còn nghĩa thứ hai."""
        text = _text(bag)
        for series in (
            'rag_cache_lookups_total{result="replay"}',
            'rag_answers_total{cited="no"}',
            'rag_llm_unpriced_steps_total{step="rewrite"}',
            # ⚠️ `single_flight` **thiếu ở đây từ `NEW-10` tới 08/09/2026**: nhãn
            # được thêm vào `MetricsSink` mà không thêm vào `_declare_zero`, nên
            # ô trên bảng đọc *"No data"* cho tới lượt gộp đầu tiên — đúng cái
            # mập mờ mà bài test này tồn tại để chống. Thêm một nhãn là **hai**
            # chỗ sửa, và chỗ thứ hai không có gì nhắc.
            'rag_cache_lookups_total{result="single_flight"}',
            # `TD-39`: 0 = "chưa phải tụt về bộ đếm cục bộ lần nào".
            'rag_quota_degraded_total{counter="ratelimit"}',
            'rag_quota_degraded_total{counter="daily_spend"}',
        ):
            assert _value(text, series) == 0, series


# ---------------------------------------------------------------------------
# 4. Nhãn — số chuỗi thời gian và cái gì rò ra
# ---------------------------------------------------------------------------


class TestLabels:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/chat", "/chat"),
            ("/health", "/health"),
            ("/metrics", "/metrics"),
            ("/conversations/2ec487b0fc804282a5f79a0afa038518", "/conversations/{id}"),
            ("/conversations/khac", "/conversations/{id}"),
            ("/admin/llm", "/admin/llm"),
            ("/admin/bundle/reload", "/admin/*"),
            ("/khong-ton-tai", "other"),
        ],
    )
    def test_paths_collapse_to_a_finite_set(self, path: str, expected: str) -> None:
        """⚠️ Để nguyên `/conversations/{id}` thì mỗi hội thoại là một chuỗi thời
        gian mới, vĩnh viễn — cách phổ biến nhất để giết một Prometheus, và nó
        không hỏng ngay mà hỏng sau vài tuần."""
        assert route_label(path) == expected

    def test_the_histogram_has_a_bucket_exactly_at_the_budget(self, bag: RagMetrics) -> None:
        """Panel "% lượt vượt ngân sách" **đếm** thay vì nội suy, và nó chỉ đếm
        được nếu bucket `le="3.5"` tồn tại — con số của bảng mục tiêu."""
        assert 'rag_chat_turn_duration_seconds_bucket{le="3.5"}' in _text(bag)

    def test_no_series_carries_a_tenant(self, sink: MetricsSink, bag: RagMetrics) -> None:
        """`/metrics` mở được bằng **bất kỳ** khoá hợp lệ. Nhãn mang tenant nghĩa
        là mọi khách hàng đọc được danh sách khách hàng."""
        trace = _turn()
        trace.user_id = "acme"
        trace.tags = ["tenant:acme"]
        sink.submit(trace)
        assert "acme" not in _text(bag)

    def test_two_registries_do_not_share_counters(self) -> None:
        """⚠️ `prometheus_client.REGISTRY` toàn cục là trạng thái tiến trình: bài
        test thứ hai dựng app sẽ nổ `Duplicated timeseries`, bài thứ ba đọc phải
        bộ đếm của bài thứ nhất."""
        first, second = RagMetrics(), RagMetrics()
        MetricsSink(first).submit(_turn())
        assert _value(_text(first), "rag_traces_finished_total") == 1
        assert _value(_text(second), "rag_traces_finished_total") == 0


# ---------------------------------------------------------------------------
# 5. Ước lượng từ chối — và việc nó tự khai là ước lượng
# ---------------------------------------------------------------------------


class TestRefusalEstimate:
    @pytest.mark.parametrize(
        "text",
        [
            "Dựa trên các tài liệu được cung cấp, tôi không tìm thấy thông tin cụ thể nào.",
            "Các nguồn không nêu rõ điều này.",
            "I could not find this in the provided context.",
            "The documents do not contain that information.",
            # Bốn dạng dưới đây là ca **bỏ sót thật**, tìm ra khi đối chiếu với
            # 242 nhãn judge của `W5-02` (xem `runs/w5-07-refusal-calibration.json`).
            "Based on the provided context, there is no information about ministries.",
            "Based on the provided context, there is no specific estimate for the cost.",
            "Dựa trên ngữ cảnh được cung cấp, không có đủ thông tin để xác định.",
            "The report gives no information about Malaysia's inflation rate.",
        ],
    )
    def test_common_refusal_wordings_are_caught(self, text: str) -> None:
        assert looks_like_refusal(text)

    def test_the_vietnamese_marker_needs_both_spellings(self) -> None:
        """⭐⭐ `"không đủ thông tin"` **không** là chuỗi con của `"không có đủ
        thông tin"` — chữ "có" chen vào giữa. Hai chuỗi đứng cạnh nhau trông như
        cái sau đã bao cái trước, và nó là nguyên nhân lớn nhất của 18 ca bỏ sót
        trong lần hiệu chỉnh đầu."""
        assert looks_like_refusal("không đủ thông tin để trả lời")
        assert looks_like_refusal("không có đủ thông tin để trả lời")

    def test_a_normal_answer_is_not_flagged(self) -> None:
        assert not looks_like_refusal("RRF hợp nhất thứ hạng của hai nhánh [1].")

    def test_the_help_line_says_it_is_an_estimate(self, bag: RagMetrics) -> None:
        """⭐⭐ `W5-02` đo từ chối bằng một **nhãn của judge**, và nói thẳng vì sao
        không dùng từ khoá. Bảng trực tuyến không gọi judge được, nên nó dùng
        ước lượng — và điều đó phải nằm trong `HELP`, chỗ người đọc bảng lúc 3
        giờ sáng nhìn thấy, chứ không phải trong một docstring."""
        line = next(
            ln for ln in _text(bag).splitlines() if ln.startswith("# HELP rag_refusals_suspected")
        )
        assert "ƯỚC LƯỢNG" in line
        assert "W5-02" in line

    def test_a_normal_turn_does_not_trip_it(self, sink: MetricsSink, bag: RagMetrics) -> None:
        sink.submit(_turn(answer="RRF hợp nhất thứ hạng [1]."))
        assert _value(_text(bag), "rag_refusals_suspected_total") == 0

    def test_a_refusal_replayed_from_cache_is_counted(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """⭐⭐ Lỗi thật, tìm ra bởi một phép tiêm **sống sót**.

        Bản đầu quét bên trong nhánh `kind == "generation"`, nên `cache.replay`
        — một span thường — không bao giờ được quét. Mẫu số là
        `rag_chat_turns_total` (có tính lượt cache) còn tử số thì không thể tăng
        cho chúng: ước lượng lệch **xuống** đúng bằng tỉ lệ trúng cache, và trên
        lượt đo thật tỉ lệ ấy là 22,2%.
        """
        trace = Trace(name="chat")
        trace.span("cache.replay").end(output="Các nguồn không nêu rõ điều này.")
        trace.metadata["finish_reason"] = "cache"
        trace.finish()
        sink.submit(trace)
        assert _value(_text(bag), "rag_refusals_suspected_total") == 1

    def test_a_marker_in_some_other_span_is_not_counted(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        """⚠️ `SYSTEM_PROMPT` chứa nguyên luật *"nếu ngữ cảnh không đủ thông tin
        thì nói rõ"*. Hôm nay span `prompt` lưu output dạng **danh sách** nên nó
        vô hại; đổi thành chuỗi ghép — rất dễ xảy ra — sẽ làm mọi lượt thành một
        lần từ chối và ô trên bảng nhảy lên 100%."""
        trace = Trace(name="chat")
        trace.span("prompt").end(output="Nếu ngữ cảnh không đủ thông tin, hãy nói rõ.")
        trace.finish()
        sink.submit(trace)
        assert _value(_text(bag), "rag_refusals_suspected_total") == 0


# ---------------------------------------------------------------------------
# 6. Không được làm hỏng một lượt chat
# ---------------------------------------------------------------------------


class TestItNeverBreaksTheTurn:
    def test_a_malformed_span_does_not_raise(self, sink: MetricsSink) -> None:
        trace = Trace(name="chat")
        span = trace.span("retrieval")
        span.end()
        span.metadata["n_hits"] = "năm"  # kiểu sai, đúng như một lần đổi metadata
        trace.finish()
        sink.submit(trace)  # không ném là toàn bộ nội dung bài này

    def test_a_trace_with_no_spans_at_all_still_counts_as_a_turn(
        self, sink: MetricsSink, bag: RagMetrics
    ) -> None:
        trace = Trace(name="chat")
        trace.finish(status="403 cross-tenant")
        sink.submit(trace)
        assert _value(_text(bag), 'rag_chat_turns_total{outcome="403"}') == 1


# ---------------------------------------------------------------------------
# 7. Thứ tự sink
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, log: list[str], name: str, *, explode: bool = False) -> None:
        self.log, self.name, self.explode = log, name, explode

    def submit(self, trace: Any) -> None:
        self.log.append(self.name)
        if self.explode:
            raise RuntimeError("sink chết")


class TestFanout:
    def test_it_calls_every_sink_in_order(self) -> None:
        log: list[str] = []
        FanoutSink((_Recorder(log, "metrics"), _Recorder(log, "langfuse"))).submit(Trace())
        assert log == ["metrics", "langfuse"]

    def test_a_sink_that_raises_does_not_block_the_next_one(self) -> None:
        """Cả hai đã hứa không ném; đây là chỗ lời hứa được cưỡng chế."""
        log: list[str] = []
        FanoutSink((_Recorder(log, "metrics", explode=True), _Recorder(log, "langfuse"))).submit(
            Trace()
        )
        assert log == ["metrics", "langfuse"]

    def test_the_app_puts_metrics_before_langfuse(self) -> None:
        """⭐⭐ Thứ tự là một quyết định, không phải một danh sách. `LangfuseSink`
        **vứt** trace khi hàng đợi đầy — và hàng đợi đầy đúng lúc hệ thống bận
        nhất. Đảo thứ tự thì bộ đếm RED cũng mất đúng phần tải cao, và một bảng
        thiếu đúng lúc có sự cố là một bảng nói ngược."""
        import inspect

        from serving.api import app as app_module

        source = inspect.getsource(app_module.create_app)
        assert source.index("MetricsSink(metrics)") < source.index("sinks.append(langfuse_sink)")
