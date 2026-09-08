"""`W4-08` — bộ định tuyến LLM: cầu dao, chuyển nhánh, trần chi phí theo ngày.

Ba câu DoD (`primary lỗi 3 lần → fallback`, `vượt budget → từ chối có thông báo
rõ`, `log model thực tế đã phục vụ`) mỗi câu có một test viết thẳng ra. Nhưng
phần đáng kiểm nhất của module này không nằm trong DoD:

**Chuyển nhánh giữa stream nối hai câu trả lời khác nhau.** Nó không ném lỗi,
không để lại dấu trong log, và kết quả là một đoạn văn trôi chảy mà không model
nào nói ra. Test cho nó dễ trở thành test rỗng (cùng họ với test huỷ của
`W4-06` không chạm đường nào), nên ở đây có **hai** test kẹp lấy ranh giới: một
cái chứng minh hỏng *trước* mẩu đầu thì chuyển nhánh được, một cái chứng minh
hỏng *sau* mẩu đầu thì không. Chỉ một trong hai thì không đo được gì.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any, cast

import pytest

from rag_core.llm import (
    AllRoutesFailed,
    BudgetExceeded,
    ChatMessage,
    CircuitBreaker,
    DailyBudget,
    LLMChunk,
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMRouter,
    ModelPricing,
    PermanentLLMError,
    Route,
)
from rag_core.schemas import TokenUsage

MESSAGES = [ChatMessage(role="user", content="Ngưỡng nghèo cùng cực là bao nhiêu?")]
PRICING = ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=2.0)


class FakeProvider(LLMProvider):
    """Nhà cung cấp giả, hỏng được ở **chỗ chọn trước** trong dòng token."""

    def __init__(
        self,
        model: str = "fake-a",
        *,
        text: str = "câu trả lời",
        deltas: Sequence[str] = ("một ", "hai ", "ba"),
        error: Exception | None = None,
        fail_after: int | None = None,
        served: str | None = None,
        cost_usd: float = 0.001,
    ) -> None:
        self.model = model
        self.name = f"fake:{model}"
        self.pricing = PRICING
        self.text = text
        self.deltas = list(deltas)
        self.error = error
        self.fail_after = fail_after
        self.served = served or model
        self.cost_usd = cost_usd
        self.calls = 0
        self.seen_extra: list[Mapping[str, Any] | None] = []

    def _response(self) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            model=self.served,
            model_requested=self.model,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20, cost_usd=self.cost_usd),
            finish_reason="stop",
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.seen_extra.append(extra_body)
        if self.error is not None:
            raise self.error
        return self._response()

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        self.calls += 1
        self.seen_extra.append(extra_body)
        if self.error is not None and self.fail_after is None:
            raise self.error
        for i, piece in enumerate(self.deltas):
            if self.error is not None and i == self.fail_after:
                raise self.error
            yield LLMChunk(delta=piece)
        yield LLMChunk(final=self._response())


class NonStreamingProvider(FakeProvider):
    """Không có `astream` — một profile chỉ phục vụ đường offline."""

    astream = None  # type: ignore[assignment]


def _drain(router: LLMRouter, **kwargs: Any) -> list[LLMChunk]:
    async def run() -> list[LLMChunk]:
        return [chunk async for chunk in router.astream(MESSAGES, **kwargs)]

    return asyncio.run(run())


def _router(*providers: LLMProvider, budget: DailyBudget | None = None, **kw: Any) -> LLMRouter:
    labels = ["primary", "fallback", "third"]
    return LLMRouter(
        routes=[
            Route(provider=p, label=labels[i], breaker=CircuitBreaker(**kw))
            for i, p in enumerate(providers)
        ],
        budget=budget,
    )


# ---------------------------------------------------------------------------
# 1. DoD — primary hỏng 3 lần thì mở cầu dao
# ---------------------------------------------------------------------------


def test_a_failing_primary_is_used_until_the_third_failure_then_skipped() -> None:
    """⭐ "Lỗi 3 lần" nghĩa là **3 request hỏng liên tiếp**, không phải 3 lần thử lại.

    Phân biệt này không phải chuyện chữ nghĩa: `OpenAICompatProvider` đã tự thử
    lại tới 5 lần với backoff bên trong **một** lời gọi, nên nếu cầu dao đếm lần
    thử lại thì nó mở sau đúng một request — và một lần 429 thoáng qua đủ để tắt
    nhà cung cấp chính.
    """
    primary = FakeProvider("primary-model", error=LLMError("HTTP 503"))
    fallback = FakeProvider("fallback-model")
    router = _router(primary, fallback, failure_threshold=3, cooldown_s=60.0)

    for _ in range(3):
        assert router.complete(MESSAGES).model == "fallback-model"
    assert primary.calls == 3, "ba request đầu vẫn phải thử primary"

    router.complete(MESSAGES)
    assert primary.calls == 3, "sau lần hỏng thứ ba, cầu dao mở và primary bị bỏ qua"
    assert router.status()["routes"][0]["state"] == "open"


def test_one_success_resets_the_counter() -> None:
    """Hỏng **liên tiếp**, không phải hỏng tích luỹ: hai lần hỏng cách nhau một
    lần thành công không được cộng lại thành ba."""
    primary = FakeProvider("primary-model", error=LLMError("HTTP 503"))
    router = _router(primary, FakeProvider("fallback-model"), failure_threshold=3, cooldown_s=60.0)

    router.complete(MESSAGES)
    router.complete(MESSAGES)
    primary.error = None
    router.complete(MESSAGES)
    primary.error = LLMError("HTTP 503")
    router.complete(MESSAGES)

    assert router.status()["routes"][0]["state"] == "closed"


def test_the_circuit_reopens_after_a_failed_probe() -> None:
    primary = FakeProvider("primary-model", error=LLMError("HTTP 503"))
    router = _router(primary, FakeProvider("fallback-model"), failure_threshold=1, cooldown_s=0.0)

    router.complete(MESSAGES)
    assert primary.calls == 1
    router.complete(MESSAGES)  # cooldown 0 ⇒ half-open ⇒ được thử lại một lần
    assert primary.calls == 2
    assert router.status()["routes"][0]["state"] in {"open", "half_open"}


def test_every_route_down_is_an_error_that_names_every_route() -> None:
    router = _router(
        FakeProvider("a", error=LLMError("HTTP 503")),
        FakeProvider("b", error=TimeoutError("hết giờ")),
    )
    with pytest.raises(LLMError) as excinfo:
        router.complete(MESSAGES)
    message = str(excinfo.value)
    assert "primary" in message and "fallback" in message
    assert "503" in message and "hết giờ" in message


# ---------------------------------------------------------------------------
# 2. ⭐ Half-open cho ĐÚNG MỘT request qua
# ---------------------------------------------------------------------------


def test_half_open_admits_exactly_one_caller() -> None:
    """⚠️ Chỗ subtle nhất của module, và nó chỉ hỏng **dưới tải**.

    Nếu half-open cho mọi request qua thì lúc cooldown hết, cả 200 request đang
    chờ cùng lao vào một nhà cung cấp vẫn đang chết — cầu dao thành bộ tạo đợt
    lỗi định kỳ thay vì một cái van. Test tuần tự vẫn thấy "nó thử lại được" nên
    vẫn xanh; thứ phân biệt là gọi `allow()` **hai lần** trước khi `record()`.
    """
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=0.0)
    breaker.record("failure")
    assert breaker.status()["state"] == "half_open"

    assert breaker.allow() is True
    assert breaker.allow() is False, "request thứ hai phải bị chặn khi chưa có kết quả thăm dò"

    breaker.record("success")
    assert breaker.status()["state"] == "closed"
    assert breaker.allow() is True


def test_a_failed_probe_puts_the_circuit_back_and_restarts_the_cooldown() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=999.0)
    breaker.record("failure")
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_the_probe_slot_is_released_even_when_the_call_raises() -> None:
    """⚠️ `allow()` giành chỗ; chỉ `record()` trả lại. Một đường thoát quên gọi
    `record()` sẽ khoá cầu dao ở half-open **vĩnh viễn** — và triệu chứng là
    nhà cung cấp chính không bao giờ được dùng lại, im lặng."""
    budget = DailyBudget(0.0000001)
    router = _router(FakeProvider("a"), budget=budget)
    with pytest.raises(BudgetExceeded):
        router.complete(MESSAGES)
    assert router.status()["routes"][0]["probing"] is False


# ---------------------------------------------------------------------------
# 3. ⭐⭐ Ranh giới mẩu đầu tiên — hai test kẹp lấy nó
# ---------------------------------------------------------------------------


def test_a_stream_that_dies_before_the_first_token_falls_over() -> None:
    """Nửa **được phép** của ranh giới. Không có test này thì test kế bên chứng
    minh được "không bao giờ chuyển nhánh", một hành vi khác hẳn."""
    primary = FakeProvider("a", error=LLMError("HTTP 503"))
    fallback = FakeProvider("b", deltas=["từ ", "fallback"])
    router = _router(primary, fallback)

    chunks = _drain(router)
    assert "".join(c.delta for c in chunks) == "từ fallback"
    assert fallback.calls == 1


def test_a_stream_that_dies_after_the_first_token_must_not_fall_over() -> None:
    """⭐⭐ Chuyển nhánh ở đây nối hai câu trả lời khác nhau thành một đoạn văn
    trôi chảy mà không model nào nói ra — không mã lỗi, không dấu vết.

    Ba phép kiểm, và cả ba đều cần: fallback **không** được gọi, lỗi **có** nổi
    lên, và phần đã sinh **không** bị vứt.
    """
    primary = FakeProvider("a", deltas=["một ", "hai ", "ba"], error=LLMError("đứt"), fail_after=2)
    fallback = FakeProvider("b", deltas=["TỪ FALLBACK"])
    router = _router(primary, fallback)

    got: list[str] = []

    async def run() -> None:
        async for chunk in router.astream(MESSAGES):
            got.append(chunk.delta)

    with pytest.raises(LLMError, match="không ai nói"):
        asyncio.run(run())

    assert fallback.calls == 0, "chuyển nhánh giữa stream = nối hai câu trả lời"
    assert "".join(got) == "một hai ", "phần đã sinh vẫn phải tới tay người gọi"


def test_a_route_that_cannot_stream_is_skipped_with_a_reason() -> None:
    router = _router(NonStreamingProvider("offline-only"), FakeProvider("b", deltas=["ok"]))
    assert "".join(c.delta for c in _drain(router)) == "ok"


def test_when_no_route_can_stream_the_error_says_so() -> None:
    router = _router(NonStreamingProvider("offline-only"))
    with pytest.raises(LLMError, match="không hỗ trợ stream"):
        _drain(router)


# ---------------------------------------------------------------------------
# 4. DoD — trần chi phí theo ngày
# ---------------------------------------------------------------------------


def test_exceeding_the_daily_budget_is_refused_with_numbers_in_the_message() -> None:
    """ "Thông báo rõ" nghĩa là đọc xong biết **đã tiêu bao nhiêu / trần bao
    nhiêu**, không phải một câu "hết ngân sách"."""
    budget = DailyBudget(0.005, name="chat")
    provider = FakeProvider("a", cost_usd=0.002)
    router = _router(provider, budget=budget)

    router.complete(MESSAGES)
    router.complete(MESSAGES)
    with pytest.raises(BudgetExceeded) as excinfo:
        for _ in range(10):
            router.complete(MESSAGES)

    message = str(excinfo.value)
    assert "0.0" in message and "trần" in message
    assert budget.status()["spent_usd"] == pytest.approx(0.004)


def test_running_out_of_budget_does_not_fall_over_to_another_provider() -> None:
    """⭐⭐ Chuyển nhánh lúc hết tiền là **tiêu thêm tiền** đúng lúc muốn ngừng tiêu.

    `BudgetExceeded` không phải `LLMError`, nên nó bay thẳng ra ngoài vòng lặp
    route — nhưng chỉ khi vòng lặp không bắt `Exception` một cách cẩu thả.
    """
    budget = DailyBudget(0.0000001)
    primary = FakeProvider("a")
    fallback = FakeProvider("b")
    router = _router(primary, fallback, budget=budget)

    with pytest.raises(BudgetExceeded):
        router.complete(MESSAGES)
    assert (primary.calls, fallback.calls) == (0, 0)


def test_running_out_of_budget_does_not_open_the_circuit() -> None:
    """Nhà cung cấp không làm gì sai. Mở cầu dao ở đây nghĩa là sáng hôm sau,
    khi ngân sách đã reset, primary vẫn đang bị cắt."""
    budget = DailyBudget(0.0000001)
    router = _router(FakeProvider("a"), FakeProvider("b"), budget=budget, failure_threshold=1)
    with pytest.raises(BudgetExceeded):
        router.complete(MESSAGES)
    assert router.status()["routes"][0]["state"] == "closed"


def test_an_unlimited_budget_must_be_stated_not_defaulted() -> None:
    """`cap <= 0` = không trần, và `budget=None` = không đếm gì cả. Hai câu khác
    nhau, và cả hai phải viết ra."""
    assert _router(FakeProvider("a")).status()["budget"] is None
    assert DailyBudget(0.0).status()["remaining_usd"] is None


def test_the_day_rolls_over_at_utc_midnight(monkeypatch: pytest.MonkeyPatch) -> None:
    """⭐ UTC chứ không giờ máy: giờ máy làm mốc reset trôi theo giờ mùa hè, và
    một hệ thống chạy ở hai vùng có "một ngày" dài khác nhau."""
    import datetime as dt

    budget = DailyBudget(0.01)
    budget.charge(0.009)
    assert budget.status()["spent_usd"] == pytest.approx(0.009)

    tomorrow = dt.datetime.now(dt.UTC).date() + dt.timedelta(days=1)
    monkeypatch.setattr(DailyBudget, "_today", staticmethod(lambda: tomorrow))

    assert budget.status()["spent_usd"] == 0.0
    budget.reserve(0.009)  # không ném: ngân sách đã sang ngày mới


def test_a_stream_cut_after_the_first_token_still_charges_something() -> None:
    """⚠️ Mẩu mang `usage` là mẩu **cuối**. Stream đứt giữa chừng thì nó không
    bao giờ tới — và ghi 0 nghĩa là một client ngắt kết nối ở mọi request sẽ
    **không bao giờ** chạm trần, trong khi token đã tiêu thật."""
    budget = DailyBudget(1.0)
    primary = FakeProvider("a", deltas=["một ", "hai"], error=LLMError("đứt"), fail_after=1)
    router = _router(primary, budget=budget)

    async def run() -> None:
        async for _ in router.astream(MESSAGES):
            pass

    with pytest.raises(LLMError):
        asyncio.run(run())
    assert budget.status()["spent_usd"] > 0.0, "đã sinh chữ thì phải tính tiền"


# ---------------------------------------------------------------------------
# 5. ⭐ Lỗi của MÌNH không được mở cầu dao
# ---------------------------------------------------------------------------


def test_a_permanent_4xx_falls_over_but_leaves_the_circuit_closed() -> None:
    """Hai route mang **`extra_body` khác nhau**, nên một 400 ở route 1 hoàn
    toàn có thể thành 200 ở route 2 — chuyển nhánh là đúng. Nhưng nó không nói
    gì về sức khoẻ route 1, và mở cầu dao là tự cắt primary vì lỗi của mình.
    """
    primary = FakeProvider("a", error=PermanentLLMError("HTTP 400: tham số lạ"))
    router = _router(primary, FakeProvider("b"), failure_threshold=1, cooldown_s=60.0)

    for _ in range(3):
        router.complete(MESSAGES)

    assert primary.calls == 3, "cầu dao không được mở vì lỗi của chính mình"
    assert router.status()["routes"][0]["state"] == "closed"
    assert router.status()["routes"][0]["consecutive_failures"] == 0


def test_a_client_that_hangs_up_mid_stream_does_not_open_the_circuit() -> None:
    """`NEW-08`/`AU-01`: khách đóng tab KHÔNG phải bằng chứng provider hỏng.

    Đường huỷ (`aclose` → `GeneratorExit`, hay `CancelledError` khi task bị
    huỷ) không đi qua `except Exception` nào, nhưng `finally` vẫn gọi
    `breaker.record(outcome)` — và `outcome` khởi tạo là `"failure"`. Trước
    sửa này, ba người dùng liên tiếp bỏ ngang một câu trả lời (đúng lúc
    provider chậm — chính lúc người ta hay bỏ) là mạch mở 30 giây cho một
    route hoàn toàn khoẻ mạnh: một sự cố tự gây, kích hoạt bởi client.
    """
    primary = FakeProvider("a", deltas=["một ", "hai ", "ba "] * 10)
    router = _router(primary, failure_threshold=3, cooldown_s=60.0)

    async def hang_up_once() -> None:
        # Kiểu khai là `AsyncIterator` nhưng runtime là một async generator —
        # `aclose` chính là "đóng tab" nhìn từ phía server.
        agen = cast("AsyncGenerator[LLMChunk, None]", router.astream(MESSAGES))
        await anext(agen)  # đã nhận chữ — như một người dùng thật
        await agen.aclose()  # rồi đóng tab

    for _ in range(3):
        asyncio.run(hang_up_once())

    status = router.status()["routes"][0]
    assert status["state"] == "closed", "ba lần khách bỏ ngang không được mở mạch"
    assert status["consecutive_failures"] == 0


def test_a_cancelled_stream_still_charges_but_stays_neutral() -> None:
    """Cùng đường huỷ, kiểm cả hai nghĩa vụ của `finally`: tiền VẪN ghi
    (token đã tiêu thật — bất biến của `W4-08`), cầu dao thì KHÔNG."""
    budget = DailyBudget(1.0)
    primary = FakeProvider("a", deltas=["một ", "hai ", "ba "] * 10)
    router = _router(primary, budget=budget, failure_threshold=1, cooldown_s=60.0)

    async def cancel_mid_stream() -> None:
        async def consume() -> None:
            async for _ in router.astream(MESSAGES):
                await asyncio.sleep(0)

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0)  # cho task chạy tới mẩu đầu
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_mid_stream())

    assert budget.status()["spent_usd"] > 0.0, "đã sinh chữ thì phải tính tiền"
    assert router.status()["routes"][0]["state"] == "closed"


# ---------------------------------------------------------------------------
# 6. DoD — log model THỰC TẾ đã phục vụ
# ---------------------------------------------------------------------------


def test_the_log_names_the_model_that_actually_served(caplog: pytest.LogCaptureFixture) -> None:
    """Quy tắc cứng #1. `model` của router là model **dự định**; sự thật là
    trường `model` trong response, và chỉ nó mới trả lời được câu hỏi của tháng
    sau: *"câu trả lời tệ này do model nào sinh ra?"*"""
    provider = FakeProvider("deepseek-v4-flash", served="deepseek-v4-flash-0925")
    router = _router(provider)

    with caplog.at_level(logging.INFO, logger="rag_core.llm.router"):
        response = router.complete(MESSAGES)

    assert response.model == "deepseek-v4-flash-0925"
    assert "served=deepseek-v4-flash-0925" in caplog.text
    assert "requested=deepseek-v4-flash" in caplog.text
    assert "fallback=False" in caplog.text


def test_the_log_says_when_the_answer_came_from_a_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = _router(FakeProvider("a", error=LLMError("503")), FakeProvider("b"))
    with caplog.at_level(logging.INFO, logger="rag_core.llm.router"):
        router.complete(MESSAGES)
    assert "fallback=True" in caplog.text


def test_the_routers_own_model_is_the_intended_one_not_the_served_one() -> None:
    """`ChatService` phát `llm.model` trong khung `meta`, tức **trước** khi lời
    gọi xảy ra. Nó không thể biết ai sẽ phục vụ, và giả vờ biết là cách quy tắc
    cứng #1 bị vô hiệu mà vẫn trông như đã làm."""
    router = _router(FakeProvider("primary-model"), FakeProvider("fallback-model"))
    assert router.model == "primary-model"
    assert "primary:primary-model" in router.name
    assert "fallback:fallback-model" in router.name


# ---------------------------------------------------------------------------
# 7. `extra_body` theo từng route
# ---------------------------------------------------------------------------


def test_each_route_carries_its_own_provider_parameters() -> None:
    """⭐ `MIN_REASONING` khác nhau theo nhà: `thinking={"type":"disabled"}` cho
    HTTP 400 ở GLM, `chat_template_kwargs` bị DeepSeek nhận rồi **bỏ qua**. Một
    bảng dùng chung vì thế hỏng im lặng ở nhà này và ồn ào ở nhà kia."""
    primary = FakeProvider("a", error=LLMError("503"))
    fallback = FakeProvider("b")
    router = LLMRouter(
        routes=[
            Route(provider=primary, label="primary", extra_body={"thinking": {"type": "disabled"}}),
            Route(provider=fallback, label="fallback", extra_body={"reasoning_effort": "low"}),
        ]
    )
    router.complete(MESSAGES)

    assert primary.seen_extra[0] == {"thinking": {"type": "disabled"}}
    assert fallback.seen_extra[0] == {"reasoning_effort": "low"}


def test_a_call_level_parameter_wins_over_the_route_default() -> None:
    provider = FakeProvider("a")
    router = LLMRouter(
        routes=[Route(provider=provider, label="primary", extra_body={"reasoning_effort": "low"})]
    )
    router.complete(MESSAGES, extra_body={"reasoning_effort": "high"})
    assert provider.seen_extra[0] == {"reasoning_effort": "high"}


def test_a_router_needs_at_least_one_route() -> None:
    with pytest.raises(LLMError, match="ít nhất một route"):
        LLMRouter(routes=[])


# ---------------------------------------------------------------------------
# 8. Hai lỗ do tiêm lỗi tìm ra
# ---------------------------------------------------------------------------


def test_a_neutral_outcome_does_not_reset_the_failure_counter() -> None:
    """⭐⭐ Phép tiêm `R6` sống sót, và nó là một lỗ thật.

    Coi `neutral` như `success` nghe vô hại: cả hai đều "không phải lỗi của nhà
    cung cấp". Nhưng `success` **xoá** bộ đếm và **đóng** mạch, nên một lượt hết
    ngân sách chen vào giữa chuỗi hỏng sẽ đưa bộ đếm về 0 — và cầu dao không bao
    giờ mở, kể cả khi nhà cung cấp đã chết hẳn.

    Kịch bản đúng ba bước: hỏng, hỏng, một lượt trung tính, hỏng ⇒ **phải** mở.
    """
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=60.0)
    breaker.record("failure")
    breaker.record("failure")
    breaker.record("neutral")
    breaker.record("failure")

    assert breaker.status()["consecutive_failures"] == 3
    assert breaker.status()["state"] == "open"


def test_a_neutral_outcome_does_not_close_an_already_open_circuit() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=999.0)
    breaker.record("failure")
    breaker.record("neutral")
    assert breaker.status()["state"] == "open"


def test_the_budget_day_is_utc_even_when_the_machine_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⭐ Phép tiêm `R8` sống sót vì test cuộn ngày **monkeypatch chính `_today`**.

    Không thể chữa bằng cách so `_today()` với ngày hiện tại: máy này ở UTC+7,
    nên `date.today()` và ngày UTC chỉ khác nhau 7 giờ trong ngày — một test như
    vậy xanh phần lớn thời gian **kể cả khi có bug**, và đó là dạng test tệ hơn
    không có test.

    Nên đóng băng đồng hồ ở đúng chỗ hai lịch lệch nhau.
    """
    import datetime as dt

    import rag_core.llm.router as router_module

    frozen_utc = dt.datetime(2026, 9, 4, 23, 30, tzinfo=dt.UTC)

    class FrozenClock:
        """Không kế thừa `datetime` — chỉ cần đúng một method mà mã kia gọi."""

        @staticmethod
        def now(tz: dt.tzinfo | None = None) -> dt.datetime:
            if tz is not None:
                return frozen_utc.astimezone(tz)
            # Giờ máy giả lập UTC+7: 23:30 UTC ⇒ 06:30 **hôm sau**.
            return frozen_utc.astimezone(dt.timezone(dt.timedelta(hours=7))).replace(tzinfo=None)

    monkeypatch.setattr(router_module, "datetime", FrozenClock)

    assert FrozenClock.now().date() == dt.date(2026, 9, 5), "giờ máy đã sang ngày mới"
    assert DailyBudget._today() == dt.date(2026, 9, 4), "ngân sách phải theo UTC"


def test_a_cancelled_stream_does_not_reset_the_failure_counter() -> None:
    """Huỷ là `neutral`, không phải `success`: hai lỗi thật + một khách bỏ
    ngang + lỗi thứ ba vẫn phải mở mạch. Ghi `success` cho đường huỷ thì một
    client hay đóng tab giữ cho một provider hỏng thật mãi mãi ở `closed`."""
    primary = FakeProvider("a", deltas=["một ", "hai ", "ba "] * 10)
    router = _router(primary, failure_threshold=3, cooldown_s=60.0)

    async def one_failure() -> None:
        primary.error = LLMError("HTTP 503")
        primary.fail_after = None
        with pytest.raises(LLMError):
            async for _ in router.astream(MESSAGES):
                pass

    async def one_hang_up() -> None:
        primary.error = None
        agen = cast("AsyncGenerator[LLMChunk, None]", router.astream(MESSAGES))
        await anext(agen)
        await agen.aclose()

    async def scenario() -> None:
        await one_failure()
        await one_failure()
        await one_hang_up()  # neutral: không cộng, không XOÁ
        await one_failure()

    asyncio.run(scenario())
    assert router.status()["routes"][0]["state"] == "open", (
        "ba lỗi thật phải mở mạch, kể cả khi một lần khách bỏ ngang chen giữa"
    )


# ---------------------------------------------------------------------------
# `TD-49` — "mọi route đều hỏng" phải nói được VÌ SAO, theo kiểu
# ---------------------------------------------------------------------------


class TestAllRoutesFailed:
    def test_moi_nha_deu_4xx_thi_permanent_va_cause_la_route_cuoi(self) -> None:
        """⭐⭐ Chính `TD-49`: "request của mình sai ở cả hai nhà" phải đọc được
        bằng mã, không chỉ bằng mắt trong log. Người gọi thấy `permanent=True`
        thì thôi thử lại — thử lại một 4xx là trả tiền cho cùng một lỗi."""
        loi_cuoi = PermanentLLMError("prompt vượt trần context")
        router = _router(
            FakeProvider(error=PermanentLLMError("json_mode không hỗ trợ")),
            FakeProvider(error=loi_cuoi),
        )
        with pytest.raises(AllRoutesFailed) as exc_info:
            router.complete(MESSAGES)
        assert exc_info.value.permanent, "hai nhà cùng từ chối request = lỗi của mình"
        assert exc_info.value.__cause__ is loi_cuoi, "traceback phải chỉ vào lần thử sau chót"
        assert len(exc_info.value.failures) == 2

    def test_mot_nha_sap_thi_KHONG_permanent(self) -> None:
        router = _router(
            FakeProvider(error=PermanentLLMError("400 của mình")),
            FakeProvider(error=LLMError("HTTP 503")),
        )
        with pytest.raises(AllRoutesFailed) as exc_info:
            router.complete(MESSAGES)
        assert not exc_info.value.permanent, "một nhà sập nghĩa là thử lại vẫn có nghĩa"

    def test_route_bi_cau_dao_bo_qua_thi_KHONG_ket_luan_permanent(self) -> None:
        """⚠️ Route bị cầu dao bỏ qua là route CHƯA ĐƯỢC HỎI — không có bằng
        chứng nó cũng sẽ từ chối. Thiếu vế này, một cầu dao đang mở biến một
        4xx lẻ ở route còn lại thành "request sai ở mọi nhà", và người gọi
        thôi thử lại đúng lúc nên thử."""
        hong = FakeProvider(error=LLMError("HTTP 500"))
        tu_choi = FakeProvider(error=PermanentLLMError("400 của mình"))
        router = _router(hong, tu_choi, failure_threshold=1, cooldown_s=3600.0)
        with pytest.raises(AllRoutesFailed):
            router.complete(MESSAGES)  # mở mạch route đầu
        with pytest.raises(AllRoutesFailed) as exc_info:
            router.complete(MESSAGES)  # lượt này route đầu bị bỏ qua
        assert exc_info.value.skipped == 1
        assert not exc_info.value.permanent

    def test_astream_cung_mang_du_loi_theo_kieu(self) -> None:
        """Hai đường ném — `complete` và `astream` — phải cùng một hợp đồng;
        một người gọi streaming không được nhận ít thông tin hơn."""
        router = _router(
            FakeProvider(error=PermanentLLMError("400 a")),
            FakeProvider(error=PermanentLLMError("400 b")),
        )
        with pytest.raises(AllRoutesFailed) as exc_info:
            _drain(router)
        assert exc_info.value.permanent
        assert len(exc_info.value.failures) == 2

    def test_van_la_LLMError_nen_moi_except_cu_con_bat_duoc(self) -> None:
        """Tương thích lùi là một hợp đồng, không phải một sự tình cờ: mọi
        `except LLMError` đang tồn tại phải tiếp tục bắt được lỗi gộp này."""
        router = _router(FakeProvider(error=LLMError("HTTP 500")))
        with pytest.raises(LLMError):
            router.complete(MESSAGES)
