"""Số đo Prometheus cho Serving Plane — `W5-07`.

## ⭐⭐ Bảng và trace đọc **cùng một** phép đo, không đo hai lần

`W5-06` đã đặt đồng hồ quanh từng lớp của một lượt và gom chúng thành một
`Trace`. Cách hiển nhiên để có metric là đặt thêm một bộ đồng hồ thứ hai ngay
cạnh — và đó là cách chắc chắn để tới một ngày bảng Grafana và trace Langfuse
nói hai điều khác nhau về cùng một request, rồi không ai biết cái nào đúng.

Nên `MetricsSink` là một `TraceSink`: cùng cây span ấy, đi qua hai người tiêu
thụ. Một span mới xuất hiện trong `chat.py` là một dòng mới trên bảng, không
cần ai nhớ sửa chỗ thứ hai.

⚠️ **Điều kiện để lối này không thành một cái bẫy**: `MetricsSink.submit` chạy
**đồng bộ, trước** hàng đợi của Langfuse, và **không bao giờ** được lấy mẫu.
`LangfuseSink` vứt trace khi hàng đợi đầy (`W5-06` §7) — nếu metric đi sau nó
thì bộ đếm cũng mất đúng những lượt xảy ra lúc hệ thống bận nhất, và một bảng
RED thiếu đúng phần tải cao là một bảng nói ngược.

## ⭐⭐ Hai tầng RED, và tầng dưới không thay được tầng trên

| họ metric | nguồn | thấy được |
|---|---|---|
| `rag_http_*` | `RequestContextMiddleware` | **mọi** request, kể cả 401/429/404 |
| `rag_chat_*`, `rag_stage_*` | cây span | chỉ những lượt `/chat` đã qua auth |

Chỉ có tầng dưới thì một sự cố xác thực — mọi khoá bị từ chối, `/chat` không
còn request nào — hiện ra trên bảng là **traffic bằng 0**, thứ trông y hệt một
đêm yên tĩnh. Tầng `rag_http_*` là chỗ duy nhất phân biệt được "không ai hỏi"
với "ai cũng bị chặn".

## ⚠️ Nhãn: không có `tenant`, và đó là quyết định

Cám dỗ rất mạnh vì mọi metric ở đây đều có một tenant rõ ràng. Hai lý do không:

1. **Nổ số chuỗi thời gian.** Mỗi tenant nhân với mỗi `stage` nhân với mỗi
   bucket histogram. Prometheus giữ tất cả trong RAM.
2. **`/metrics` là một danh sách khách hàng.** Ai đọc được endpoint ấy đọc được
   tên mọi tenant và lưu lượng của từng người. Đó là dữ liệu kinh doanh, và nó
   rò qua một cửa mà không ai coi là cửa dữ liệu.

Phân tách theo tenant thuộc về Langfuse (`session_id`/`user_id`, có xác thực
theo project) chứ không thuộc về một endpoint scrape.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from rag_core.generation.refusal import REFUSAL_MARKERS, looks_like_refusal
from serving.core.tracing import Trace
from serving.db.models import FEEDBACK_REASONS

__all__ = [
    "CONTENT_TYPE_LATEST",
    "MetricsSink",
    "RagMetrics",
    # Xuất lại từ `rag_core.generation.refusal` (`W5-11`): người gọi cũ — kể cả
    # test của `W5-07` — vẫn thấy nó ở đây, còn bản duy nhất thì nằm một chỗ.
    "looks_like_refusal",
    "route_label",
]

logger = logging.getLogger(__name__)

_DURATION_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.5,
    5.0,
    7.5,
    10.0,
    20.0,
)
"""Bucket cho mọi histogram thời lượng, tính bằng **giây**.

⭐ `3.5` có mặt vì nó là ngân sách p95 end-to-end của bảng mục tiêu. Một
histogram không có bucket đúng tại ngưỡng thì `histogram_quantile` phải nội suy
qua chỗ ấy, và câu hỏi "bao nhiêu phần trăm request vượt ngân sách" — thứ duy
nhất người vận hành thật sự hỏi — trở thành một phép ước lượng thay vì một phép
đếm. Với bucket này nó là `rag_..._bucket{le="3.5"}` chia cho `_count`.

⭐ `20.0` là trần trên cùng trước `+Inf`, và nó rộng có chủ đích: `W5-06` đo
được lượt đầu sau deploy tốn 9,9 s (`TD-72`). Một trần 5 s sẽ nhét mọi lần khởi
động lạnh vào `+Inf` chung với mọi sự cố thật.
"""

_HIT_BUCKETS = (0, 1, 2, 3, 5, 10, 20, 50)
"""Số chunk truy hồi được. `0` là bucket riêng vì "không tìm thấy gì" là một
trạng thái khác hẳn "tìm được ít", và bảng cần đếm được nó."""


def route_label(path: str) -> str:
    """Gom đường dẫn về một tập **hữu hạn** trước khi nó thành nhãn.

    ⚠️ `/conversations/{id}` mà để nguyên thì mỗi hội thoại là một chuỗi thời
    gian mới, vĩnh viễn — đây là cách phổ biến nhất để giết một Prometheus, và
    nó không hỏng ngay mà hỏng sau vài tuần.

    Không dùng `scope["route"]` của Starlette vì middleware này là lớp **ngoài
    cùng**: khi một request bị auth từ chối, router chưa từng chạy và
    `scope["route"]` không tồn tại — tức đúng những request đáng đếm nhất lại
    rơi vào nhánh không có nhãn.
    """
    if path in {"/health", "/ready", "/metrics", "/chat", "/docs", "/openapi.json"}:
        return path
    if path.startswith("/admin/"):
        return path if path.count("/") == 2 else "/admin/*"
    if path.startswith("/conversations/"):
        return "/conversations/{id}"
    return "other"


class RagMetrics:
    """Sổ đăng ký của tiến trình. Một thực thể cho mỗi app, tiêm được vào test.

    ⚠️ **Không** dùng `prometheus_client.REGISTRY` toàn cục: nó là trạng thái
    tiến trình, nên bài test thứ hai dựng app sẽ nổ `Duplicated timeseries`, và
    bài test thứ ba sẽ đọc phải bộ đếm của bài thứ nhất. Một `CollectorRegistry`
    riêng làm cho "số đo của app này" là một câu hỏi có câu trả lời.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        reg = self.registry

        # ------------------------------------------------------ tầng HTTP
        self.http_requests = Counter(
            "rag_http_requests",
            "Số request HTTP đã phục vụ, theo route và status.",
            ("method", "route", "status"),
            registry=reg,
        )
        self.http_duration = Histogram(
            "rag_http_request_duration_seconds",
            "Thời gian tới byte CUỐI của phản hồi (với SSE thì không phải TTFB).",
            ("route",),
            buckets=_DURATION_BUCKETS,
            registry=reg,
        )

        # ------------------------------------------------------ tầng lượt chat
        self.chat_turns = Counter(
            "rag_chat_turns",
            "Số lượt /chat đã kết thúc, theo `finish_reason`.",
            ("outcome",),
            registry=reg,
        )
        self.chat_duration = Histogram(
            "rag_chat_turn_duration_seconds",
            "Toàn bộ lượt, đo từ handler HTTP — gồm cả `prepare()` (xem TD-55).",
            buckets=_DURATION_BUCKETS,
            registry=reg,
        )
        self.stage_duration = Histogram(
            "rag_stage_duration_seconds",
            "Thời lượng mỗi bước, đọc thẳng từ cây span của W5-06.",
            ("stage",),
            buckets=_DURATION_BUCKETS,
            registry=reg,
        )

        # ------------------------------------------------------ cache & truy hồi
        self.cache_lookups = Counter(
            "rag_cache_lookups",
            "Lượt tra semantic cache, theo kết quả.",
            ("result",),
            registry=reg,
        )
        self.retrieval_hits = Histogram(
            "rag_retrieval_hits",
            "Số chunk trả về sau rerank. Bucket `0` = không tìm thấy gì.",
            buckets=_HIT_BUCKETS,
            registry=reg,
        )
        self.retrieval_empty = Counter(
            "rag_retrieval_empty",
            "Lượt truy hồi trả về 0 chunk.",
            registry=reg,
        )

        # ------------------------------------------------------ chất lượng câu trả lời
        self.answers = Counter(
            "rag_answers",
            "Câu trả lời đã sinh, theo việc có trích được nguồn đã xác minh hay không.",
            ("cited",),
            registry=reg,
        )
        self.refusals_suspected = Counter(
            "rag_refusals_suspected",
            (
                "ƯỚC LƯỢNG số lần từ chối, dò bằng từ khoá. KHÔNG phải phép đo "
                "refusal của W5-02 (judge) — xem docstring `looks_like_refusal`."
            ),
            registry=reg,
        )
        self.citations_unverified = Counter(
            "rag_citations_unverified",
            "Số citation model tuyên bố mà W4-09 không xác minh được.",
            registry=reg,
        )

        # ------------------------------------------------------ tiền
        self.llm_cost = Counter(
            "rag_llm_cost_usd",
            "Tiền đã tiêu, theo model và bước. `rate(...[1h]) * 3600` = $/giờ.",
            ("model", "step"),
            registry=reg,
        )
        self.llm_tokens = Counter(
            "rag_llm_tokens",
            "Token đã dùng, theo model, bước và chiều.",
            ("model", "step", "direction"),
            registry=reg,
        )
        self.unpriced_steps = Counter(
            "rag_llm_unpriced_steps",
            (
                "Bước gọi model KHÔNG khai được chi phí. Mẫu số của `rag_llm_cost_usd`: "
                "tổng tiền chỉ là tổng khi bộ đếm này đứng yên (xem tracing.Usage)."
            ),
            ("step",),
            registry=reg,
        )

        # ------------------------------------------------------ phản hồi người dùng
        self.feedback = Counter(
            "rag_feedback",
            (
                "Lượt 👍/👎 đã ghi. Nhãn `rating` là '1'/'-1', `reason` là tập ĐÓNG "
                "FEEDBACK_REASONS cộng 'none'. ⚠️ Mẫu số KHÔNG phải rag_chat_turns: "
                "gần như không ai bấm nút, nên đây là một mẫu tự chọn."
            ),
            ("rating", "reason"),
            registry=reg,
        )

        # ------------------------------------------------------ sức khoẻ quan sát
        self.traces = Counter(
            "rag_traces_finished",
            "Số trace đã đóng và đi qua lớp metric.",
            registry=reg,
        )
        self.trace_sink = Gauge(
            "rag_trace_sink",
            "Trạng thái hàng đợi đẩy trace sang Langfuse.",
            ("state",),
            registry=reg,
        )
        self.bundle = Gauge(
            "rag_bundle_info",
            "Luôn bằng 1; nhãn `version` là bundle đang phục vụ.",
            ("version",),
            registry=reg,
        )

        self._declare_zero()

    # ------------------------------------------------------------------ đọc

    def render(self) -> bytes:
        return generate_latest(self.registry)

    # ------------------------------------------------------------------ nội bộ

    def _declare_zero(self) -> None:
        """Tạo sẵn mọi tổ hợp nhãn **hữu hạn** ở giá trị 0.

        ## ⭐⭐ Một metric có nhãn KHÔNG tồn tại cho tới lần quan sát đầu tiên

        `Counter(...).labels(result="hit")` chỉ sinh ra một chuỗi thời gian khi
        có ai gọi nó. Trước đó, `/metrics` không in dòng nào, và Grafana vẽ ô
        *"No data"*.

        Ba chữ ấy có **hai** nghĩa hoàn toàn khác nhau, và bảng không phân biệt
        được:

        * "điều kiện này chưa xảy ra lần nào" — bình thường, thậm chí là tin tốt
          (chưa có citation nào hỏng, chưa có bước nào không định giá được);
        * "metric đã bị đổi tên và không ai sửa bảng" — hỏng, và hỏng lặng lẽ.

        Khai trước ở 0 làm nghĩa thứ nhất hiện ra là **số 0**, nên *"No data"*
        còn lại đúng một nghĩa. Đây cũng là điều bài
        `test_every_metric_the_dashboard_asks_for_exists` tìm ra: ba metric
        trong bảng không có mặt trong bản phơi bày chỉ vì chưa ai chạm tới
        chúng.

        ⚠️ Chỉ áp cho tập nhãn **đóng**. `outcome` (mọi `finish_reason` cộng mọi
        mã lỗi) và `model`/`step` của bảng chi phí là mở — khai trước ở đó là
        đoán trước một danh sách sẽ sai, và mỗi lần đoán sai là một chuỗi thời
        gian ma nằm mãi trên bảng.
        """
        for result in ("hit", "miss", "replay"):
            self.cache_lookups.labels(result=result)
        for cited in ("yes", "no"):
            self.answers.labels(cited=cited)
        for rating in ("1", "-1"):
            for reason in (*FEEDBACK_REASONS, "none"):
                self.feedback.labels(rating=rating, reason=reason)
        # Hai bước gọi model mà đường `/chat` thật sự có. Danh sách này đóng vì
        # nó là danh sách span `kind="generation"` trong `chat.py`, không phải
        # một danh sách mở của nhà cung cấp.
        for step in ("rewrite", "completion"):
            self.unpriced_steps.labels(step=step)


_ANSWER_SPANS = frozenset({"completion", "cache.replay"})
"""Span nào mang **văn bản câu trả lời**. Chỉ hai, và danh sách này đóng.

⚠️ Không quét mọi span có `output` là chuỗi. `SYSTEM_PROMPT` chứa nguyên luật
*"Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ là không đủ thông tin"* — hôm nay
span `prompt` lưu output dạng **danh sách** message nên nó vô hại, nhưng đổi nó
thành một chuỗi ghép (rất dễ xảy ra) sẽ làm mọi lượt thành một lần từ chối, và
ô trên bảng nhảy lên 100% mà không ai hiểu vì sao."""


#: `W5-11`: bộ dò chuyển xuống `rag_core` để `pipeline` hiệu chỉnh lại nó theo
#: từng model sinh (`TD-77`) mà không phải import ngược sang Serving Plane. Tên
#: cũ giữ nguyên ở đây vì `MetricsSink` bên dưới và các test của `W5-07` gọi nó.
_REFUSAL_MARKERS = REFUSAL_MARKERS


class MetricsSink:
    """`TraceSink` biến một cây span đã đóng thành các số đếm.

    Hợp đồng của `TraceSink`: **không chặn, không ném**. Ở đây "không chặn" là
    miễn phí (mọi thao tác là cộng vào một số nguyên trong bộ nhớ), còn "không
    ném" phải viết ra — một `KeyError` vì đổi tên metadata của span sẽ giết một
    câu trả lời đã trả tiền rồi.
    """

    def __init__(self, metrics: RagMetrics) -> None:
        self.metrics = metrics

    def submit(self, trace: Trace) -> None:
        try:
            self._record(trace)
        except Exception:
            logger.warning("metrics: không đọc được trace %s", trace.id, exc_info=True)

    # ------------------------------------------------------------------ nội bộ

    def _record(self, trace: Trace) -> None:
        m = self.metrics
        m.traces.inc()
        outcome = str(trace.metadata.get("finish_reason") or _outcome_of(trace))
        m.chat_turns.labels(outcome=outcome).inc()
        if trace.end_time is not None:
            m.chat_duration.observe((trace.end_time - trace.start_time).total_seconds())

        for span in trace.spans:
            if span.duration_ms is not None:
                m.stage_duration.labels(stage=span.name).observe(span.duration_ms / 1000.0)
            self._span_extras(span)

        for name in trace.unmeasured_cost_steps():
            m.unpriced_steps.labels(step=name).inc()

    def _span_extras(self, span: Any) -> None:
        m = self.metrics
        meta = span.metadata
        name = span.name

        if name == "cache.lookup":
            m.cache_lookups.labels(result="hit" if meta.get("hit") else "miss").inc()
        elif name == "cache.replay":
            # Một lượt phát lại từ cache **không** đi qua `cache.lookup` với
            # `hit=True` ở lượt này — nó đi qua đó ở lượt trước. Đếm riêng, nếu
            # không thì tỉ lệ trúng cache đếm thiếu đúng phần nó phục vụ.
            m.cache_lookups.labels(result="replay").inc()
        elif name in {"retrieval", "retrieve"}:
            # Chỉ span **ngoài cùng** của chuỗi. `retrieve.hybrid` là con
            # và mang `n_hits=50` (độ sâu pool rerank, không phải kết quả),
            # nên đếm nó vào đây sẽ làm "truy hồi rỗng" không bao giờ xảy ra
            # và histogram số chunk lệch hẳn một bậc.
            hits = meta.get("n_hits")
            if isinstance(hits, int):
                m.retrieval_hits.observe(hits)
                if hits == 0:
                    m.retrieval_empty.inc()
        elif name == "citations":
            verified = meta.get("verified")
            claimed = meta.get("claimed")
            if isinstance(verified, int) and isinstance(claimed, int):
                m.answers.labels(cited="yes" if verified else "no").inc()
                if claimed > verified:
                    m.citations_unverified.inc(claimed - verified)

        # ⭐⭐ Quét ở đây, TRƯỚC lối thoát `kind != "generation"` bên dưới.
        #
        # Bản đầu để phép quét bên trong nhánh generation, nên `cache.replay` —
        # một span `kind="span"` — không bao giờ được quét. Hệ quả: mẫu số là
        # `rag_chat_turns_total` (có tính lượt cache) còn tử số thì không thể
        # tăng cho chúng, tức ước lượng **lệch xuống đúng bằng tỉ lệ trúng
        # cache**. Đo được ở đây: 22,2% lượt là replay.
        #
        # Một phép tiêm sống sót đã phơi ra chỗ này — nó "sống" vì đổi luật
        # thành *quét mọi span* gần như không đổi hành vi, và điều đó chỉ đúng
        # khi luật hiện tại đang bỏ sót gần hết những gì đáng quét.
        carries_answer = name in _ANSWER_SPANS and isinstance(span.output, str)
        if carries_answer and looks_like_refusal(span.output):
            m.refusals_suspected.inc()

        if span.kind != "generation":
            return
        model = span.model or "unknown"
        usage = span.usage
        if usage.cost_usd is not None:
            m.llm_cost.labels(model=model, step=name).inc(usage.cost_usd)
        if usage.prompt_tokens is not None:
            m.llm_tokens.labels(model=model, step=name, direction="in").inc(usage.prompt_tokens)
        if usage.completion_tokens is not None:
            m.llm_tokens.labels(model=model, step=name, direction="out").inc(
                usage.completion_tokens
            )


def _outcome_of(trace: Trace) -> str:
    """`finish_reason` khi lượt tới được nửa dưới; mã lỗi khi không.

    ⭐ Lượt hỏng ở `prepare()` không có `finish_reason` nào cả — nó chết trước
    khi có một. Gán cho nó `"unknown"` là gộp mọi 404/403/429/503 vào một ô mà
    người vận hành không tách ra được, nên ở đây lấy tiền tố mã lỗi mà
    `api/chat.py` đã ghi vào `status_message` (`"404 …"`, `"503 …"`).
    """
    status = trace.status_message or ""
    head = status.split(" ", 1)[0]
    return head if head.isdigit() else (status or "unknown")
