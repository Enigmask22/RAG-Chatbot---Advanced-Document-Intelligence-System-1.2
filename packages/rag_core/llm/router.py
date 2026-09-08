"""Định tuyến giữa nhiều nhà cung cấp LLM, có cầu dao và trần chi phí — `W4-08`.

## ⭐⭐ Chuyển nhà cung cấp GIỮA STREAM là nối hai câu trả lời khác nhau

Đây là ràng buộc quan trọng nhất của cả module, và cách viết hiển nhiên vi phạm
nó:

```python
for route in routes:                      # SAI
    try:
        async for chunk in route.astream(...):
            yield chunk
        return
    except LLMError:
        continue                          # ← đã gửi 50 token của route 1 rồi
```

Người dùng nhận nửa câu trả lời của DeepSeek nối vào nửa câu trả lời của
OpenRouter. Kết quả là một đoạn văn **trôi chảy, mạch lạc, và không model nào
nói ra** — không có mã lỗi, không có gì trong log nói rằng nó đã xảy ra.

Ranh giới là **mẩu đầu tiên**, y hệt ranh giới retry mà `OpenAICompatProvider`
đã dựng một tầng bên dưới:

* chưa `yield` gì ⇒ chuyển route được (và đó là ca thường gặp nhất: 429/5xx xảy
  ra lúc mở kết nối)
* đã `yield` ⇒ `LLMError`, phần đã sinh giữ nguyên cho người gọi đánh dấu dở dang

## ⭐ Ba loại "hỏng", và chúng phải được đối xử khác nhau

| loại | chuyển route? | tính cầu dao? | vì sao |
|---|---|---|---|
| `BudgetExceeded` | **không** | **không** | hết tiền mà chuyển route là tiêu **thêm** |
| `PermanentLLMError` (4xx của mình) | có | **không** | request của mình sai, không phải nhà chết |
| còn lại (429/5xx/timeout/mạng) | có | **có** | đây mới là bằng chứng nhà cung cấp đang hỏng |

Hàng giữa quan trọng hơn vẻ ngoài của nó: mở cầu dao vì một 400 của chính mình
là **tự cắt** nhà cung cấp chính trong 30 giây cho một lỗi mà thử lại bao nhiêu
lần cũng thế.

Hàng thứ hai không phải chi tiết: hai route có **`extra_body` khác nhau**
(`thinking={"type":"disabled"}` cho HTTP 400 ở GLM, `chat_template_kwargs` bị
DeepSeek nhận rồi bỏ qua), nên một 400 ở route 1 hoàn toàn có thể thành 200 ở
route 2. Nhưng nó không nói gì về sức khoẻ của route 1.

## ⚠️ Trần theo ngày này đếm TRONG TIẾN TRÌNH

Cùng một giới hạn với hạn mức nhịp của `W4-04` (`TD-39`), và cùng hướng hỏng: 4
replica ⇒ trần thật là **4×** con số đã cấu hình, và một lần restart đưa bộ đếm
về 0. Nó vẫn chặn được ca mà nó sinh ra để chặn — một vòng lặp hỏng đốt hết ngân
sách trong mười phút — nhưng nó **không phải** một trần đúng nghĩa cho nhiều
tiến trình. Ghi ra ở đây để không ai đọc `chat_daily_budget_usd` mà tưởng đã có
trần thật. Chỗ đúng là một bộ đếm dùng chung (Redis) ở `W4-10`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from .base import ChatMessage, LLMChunk, LLMError, LLMProvider, LLMResponse, StreamingLLM
from .budget import BudgetExceeded, CostBudget
from .openai_compat import PermanentLLMError


class AllRoutesFailed(LLMError):
    """Mọi route đều không phục vụ được — và mang theo TỪNG lỗi, theo kiểu. `TD-49`.

    Một `LLMError` phẳng làm người gọi bằng mã không phân biệt được *"cả hai
    nhà đều sập"* (thử lại có nghĩa) với *"request của mình sai ở cả hai nhà"*
    (thử lại vô ích). Dòng log từng route vẫn ghi đúng phân loại — nhưng log là
    cho người vận hành; **kiểu** là cho người gọi. `__cause__` là lỗi của route
    cuối cùng, nên traceback mặc định đã chỉ thẳng vào lần thử sau chót.
    """

    def __init__(
        self, message: str, *, failures: Sequence[BaseException] = (), skipped: int = 0
    ) -> None:
        super().__init__(message)
        self.failures = list(failures)
        """Lỗi của từng route ĐÃ ĐƯỢC HỎI, đúng thứ tự thử."""
        self.skipped = skipped
        """Số route bị cầu dao bỏ qua — chưa được hỏi, không phải đã từ chối."""

    @property
    def permanent(self) -> bool:
        """`True` = request của MÌNH bị mọi nhà từ chối — thử lại vô ích.

        ⚠️ Một route bị cầu dao bỏ qua là một route **chưa được hỏi**: không có
        bằng chứng nó cũng sẽ từ chối, nên `skipped > 0` ⇒ `False`. Thiếu vế
        này thì một cầu dao đang mở biến một 4xx lẻ ở route còn lại thành
        "request sai ở mọi nhà" — và người gọi sẽ thôi thử lại đúng lúc nên thử.
        """
        return (
            self.skipped == 0
            and bool(self.failures)
            and all(isinstance(f, PermanentLLMError) for f in self.failures)
        )


__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "DailyBudget",
    "LLMRouter",
    "Route",
]

logger = logging.getLogger(__name__)

CircuitState = Literal["closed", "open", "half_open"]

Outcome = Literal["success", "failure", "neutral"]
"""`neutral` = không nói gì về sức khoẻ nhà cung cấp (ngân sách, lỗi của mình)."""


class CircuitBreaker:
    """Đếm hỏng liên tiếp, mở mạch, thử lại **một** request sau thời gian nguội.

    An toàn luồng bằng `threading.Lock` chứ không `asyncio.Lock`: cùng một bộ
    định tuyến phục vụ `complete()` chạy trong threadpool (`W4-07` gọi nó qua
    `to_thread`) và `astream()` chạy trên vòng lặp sự kiện. Khoá không bao giờ
    được giữ qua một `await`, nên nó đúng cho cả hai.
    """

    def __init__(self, *, failure_threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._consecutive = 0
        self._opened_at: float | None = None
        self._probing = False
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> CircuitState:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """`True` nếu được phép gọi — và **giành chỗ** nếu đang là half-open.

        ⭐ Half-open cho **đúng một** request qua, không phải "mọi request sau
        khi hết nguội". Khác biệt chỉ lộ ra dưới tải: nhà cung cấp vẫn chết,
        cooldown hết, và cả 200 request đang chờ cùng lao vào nó — tức cầu dao
        biến thành một bộ tạo đợt lỗi định kỳ thay vì một cái van.

        Người gọi **bắt buộc** gọi `record()` sau đó, kể cả trên đường lỗi: chỗ
        giành ở đây chỉ được trả lại ở đó.
        """
        with self._lock:
            state = self._state_locked()
            if state == "closed":
                return True
            if state == "open":
                return False
            if self._probing:
                return False
            self._probing = True
            return True

    def record(self, outcome: Outcome) -> None:
        with self._lock:
            self._probing = False
            if outcome == "neutral":
                return
            if outcome == "success":
                self._consecutive = 0
                self._opened_at = None
                return
            self._consecutive += 1
            if self._consecutive >= self.failure_threshold:
                # Lần hỏng thứ N **hoặc** một lần hỏng ở half-open đều mở lại
                # mạch với đồng hồ nguội mới.
                self._opened_at = time.monotonic()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state_locked(),
                "consecutive_failures": self._consecutive,
                "probing": self._probing,
            }


class DailyBudget:
    """Trần chi phí **theo ngày UTC**, cuộn sang ngày mới lúc nửa đêm.

    Dùng lại `CostBudget` bên trong thay vì đếm tay: phần khoá, phần `reserve`
    trước / `charge` sau, và luật `cap <= 0` nghĩa là không trần — cả ba đã có
    ở đó và đã có test từ `W3-04`.

    ⭐ **UTC chứ không giờ máy.** Giờ máy nghĩa là cùng một hệ thống có "một
    ngày" dài khác nhau tuỳ chỗ chạy, và mốc reset trôi theo giờ mùa hè. Không
    ai muốn đọc một hoá đơn có 23 giờ trong đó.
    """

    def __init__(self, cap_usd: float, *, name: str = "chat") -> None:
        self.cap_usd = cap_usd
        self.name = name
        self._lock = threading.Lock()
        self._day = self._today()
        self._budget = CostBudget(cap_usd, name=f"{name}/{self._day.isoformat()}")

    @staticmethod
    def _today() -> date:
        return datetime.now(UTC).date()

    def _roll(self) -> CostBudget:
        with self._lock:
            today = self._today()
            if today != self._day:
                logger.info(
                    "ngân sách %s sang ngày mới: %s đã tiêu $%.4f/%d lời gọi",
                    self.name,
                    self._day.isoformat(),
                    self._budget.spent_usd,
                    self._budget.calls,
                )
                self._day = today
                self._budget = CostBudget(self.cap_usd, name=f"{self.name}/{today.isoformat()}")
            return self._budget

    def reserve(self, estimate_usd: float) -> None:
        self._roll().reserve(estimate_usd)

    def charge(self, actual_usd: float) -> float:
        return self._roll().charge(actual_usd)

    def status(self) -> dict[str, Any]:
        budget = self._roll()
        return {
            "day": self._day.isoformat(),
            "cap_usd": self.cap_usd,
            "spent_usd": round(budget.spent_usd, 6),
            "remaining_usd": (None if budget.unlimited else round(budget.remaining_usd, 6)),
            "calls": budget.calls,
        }


_CHARS_PER_TOKEN = 3.0
"""Ước lượng **bi quan** để `reserve()` chặn sớm hơn là muộn.

`TD-17` đo tỉ lệ thật trên corpus này là ~0,20 token/ký tự (5 ký tự/token) cho
văn bản bình thường. Dùng 3 ở đây là cố ý ước cao: `reserve()` phải trả lời câu
*"lời gọi kế tiếp có thể vượt trần không"*, nên ước thấp là để trần bị vượt
đúng một lời gọi — và `charge()` ghi số **thật** ngay sau đó nên sai số không
cộng dồn.
"""


@dataclass
class Route:
    """Một nhánh của bộ định tuyến: nhà cung cấp + tham số riêng của nhà đó."""

    provider: LLMProvider
    label: str
    extra_body: Mapping[str, Any] | None = None
    """⚠️ Theo **từng route**, không phải một bảng cho cả router.

    `W4-06` đặt `MIN_REASONING` ở `ChatService` vì lúc đó chỉ có một nhà cung
    cấp. Hai trong bốn dòng của bảng ấy là *tham số được nhận, không lỗi, và
    không có tác dụng* nếu đặt nhầm nhà — và một dòng (`thinking`) làm GLM trả
    HTTP 400. Một `extra_body` dùng chung cho mọi route vì vậy hỏng **im lặng**
    ở route này và **ồn ào** ở route kia.
    """

    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def body(self, call_extra: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Trộn tham số của route với tham số của lời gọi; **lời gọi thắng**."""
        merged = {**(self.extra_body or {}), **(call_extra or {})}
        return merged or None

    def estimate_usd(self, messages: Sequence[ChatMessage], max_tokens: int | None) -> float:
        pricing = getattr(self.provider, "pricing", None)
        if pricing is None:
            return 0.0
        chars = sum(len(m.content) for m in messages)
        prompt_tokens = int(chars / _CHARS_PER_TOKEN)
        cost: float = pricing.cost(prompt_tokens, max_tokens or 1024)
        return cost


@dataclass
class LLMRouter(LLMProvider):
    """Thử từng route theo thứ tự, dừng ở route đầu tiên trả lời được.

    Cài **cả** `LLMProvider.complete` lẫn `StreamingLLM.astream`, nên nó cắm
    thẳng vào `ChatService.llm` và `QueryUnderstanding.llm` mà không sửa dòng
    nào ở hai chỗ đó — cả hai đã khai kiểu bằng Protocol từ `W4-06`/`W4-07`.
    """

    routes: Sequence[Route]
    budget: DailyBudget | None = None

    def __post_init__(self) -> None:
        if not self.routes:
            raise LLMError("bộ định tuyến cần ít nhất một route")
        # ⭐ Model **dự định**, không phải model đã phục vụ. `ChatService` phát
        # nó trong khung `meta` — tức trước khi lời gọi xảy ra, nên nó không thể
        # biết. Sự thật nằm ở khung `done`, đọc từ `LLMResponse.model` của route
        # đã thật sự trả lời. Hai trường cho hai câu hỏi khác nhau, và gộp chúng
        # lại là cách quy tắc cứng #1 ("log model **thực tế** đã phục vụ") bị vô
        # hiệu mà vẫn trông như đã làm.
        self.model = self.routes[0].provider.model
        self.name = "router[" + "|".join(f"{r.label}:{r.provider.model}" for r in self.routes) + "]"

    # ------------------------------------------------------------- ngân sách

    def _reserve(
        self, route: Route, messages: Sequence[ChatMessage], max_tokens: int | None
    ) -> None:
        if self.budget is not None:
            self.budget.reserve(route.estimate_usd(messages, max_tokens))

    def assert_within_budget(self) -> None:
        """Ném `BudgetExceeded` nếu ngân sách hôm nay **đã** cạn.

        ⭐ Tồn tại để `ChatService.prepare()` hỏi được câu ấy **trước** khi byte
        đầu tiên rời đi. Sau `200 OK` thì hết tiền chỉ còn là một dòng SSE dừng
        lại — trông y hệt một câu trả lời ngắn — trong khi câu trả lời đúng cho
        người gọi là một `429` đọc được bằng máy. Cùng đường phân giới mà
        `W4-06` đã dựng.

        Nó trả lời "đã cạn chưa", **không** phải "lời gọi kế tiếp có vượt
        không": ở thời điểm `prepare()`, prompt chưa tồn tại. Phép kiểm chặt hơn
        vẫn chạy trong `complete`/`astream` ngay trước lời gọi thật.
        """
        if self.budget is not None:
            self.budget.reserve(0.0)

    def _charge(self, actual_usd: float) -> None:
        if self.budget is not None:
            self.budget.charge(actual_usd)

    # ------------------------------------------------------------- không stream

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
        problems: list[str] = []
        failures: list[BaseException] = []
        skipped = 0
        for route in self.routes:
            if not route.breaker.allow():
                problems.append(f"{route.label}: cầu dao {route.breaker.state}")
                skipped += 1
                continue
            outcome: Outcome = "failure"
            try:
                # `BudgetExceeded` bay thẳng ra ngoài — xem bảng ở docstring.
                self._reserve(route, messages, max_tokens)
                response = route.provider.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    seed=seed,
                    extra_body=route.body(extra_body),
                )
            except BudgetExceeded:
                outcome = "neutral"
                raise
            except PermanentLLMError as exc:
                outcome = "neutral"
                problems.append(f"{route.label}: {exc}")
                failures.append(exc)
                logger.warning("route %s từ chối request (lỗi của mình): %s", route.label, exc)
            except Exception as exc:
                problems.append(f"{route.label}: {exc}")
                failures.append(exc)
                logger.warning("route %s hỏng: %s", route.label, exc)
            else:
                outcome = "success"
                self._charge(response.usage.cost_usd)
                self._log_served(route, response.model, response.usage.cost_usd)
                return response
            finally:
                route.breaker.record(outcome)
        raise AllRoutesFailed(self._all_failed(problems), failures=failures, skipped=skipped) from (
            failures[-1] if failures else None
        )

    # ------------------------------------------------------------------ stream

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        failures: list[BaseException] = []
        skipped = 0
        """Xem §"Chuyển nhà cung cấp GIỮA STREAM" ở docstring module."""
        problems: list[str] = []
        for route in self.routes:
            provider = route.provider
            if not isinstance(provider, StreamingLLM):
                problems.append(f"{route.label}: không hỗ trợ stream")
                continue
            if not route.breaker.allow():
                problems.append(f"{route.label}: cầu dao {route.breaker.state}")
                skipped += 1
                continue

            outcome: Outcome = "failure"
            emitted = False
            charged = False
            reserved = 0.0
            try:
                reserved = route.estimate_usd(messages, max_tokens)
                self._reserve(route, messages, max_tokens)
                async for chunk in provider.astream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=route.body(extra_body),
                ):
                    if chunk.delta:
                        emitted = True
                    if chunk.final is not None:
                        self._charge(chunk.final.usage.cost_usd)
                        charged = True
                        self._log_served(route, chunk.final.model, chunk.final.usage.cost_usd)
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                # ⭐⭐ `NEW-08`/`AU-01`: đường huỷ phải TỰ đặt `outcome`, vì nó
                # không đi qua `except Exception` nào mà `finally` thì vẫn gọi
                # `breaker.record(outcome)` với giá trị khởi tạo `"failure"`.
                # Khách đóng tab không phải bằng chứng nhà cung cấp hỏng — ba
                # người bỏ ngang liên tiếp mà mở mạch 30 giây cho một route
                # khoẻ mạnh là một sự cố tự gây, kích hoạt đúng lúc tải cao
                # (provider chậm là lúc người ta hay bỏ nhất).
                outcome = "neutral"
                raise
            except BudgetExceeded:
                outcome = "neutral"
                raise
            except Exception as exc:
                # `CancelledError`/`GeneratorExit` là `BaseException`, nên chúng
                # **không** rơi vào đây — chúng có nhánh riêng ở trên, vừa để đi
                # tiếp nguyên vẹn, vừa để cầu dao ghi `neutral` thay vì giá trị
                # khởi tạo. Khối `finally` vẫn chạy để ghi tiền.
                outcome = "neutral" if isinstance(exc, PermanentLLMError) else "failure"
                if emitted:
                    # ⭐⭐ Đã gửi token đi rồi thì **không** còn đường lui.
                    raise LLMError(
                        f"route {route.label} đứt giữa stream: {exc}. Không chuyển route: "
                        "nối hai câu trả lời khác nhau tạo ra một đoạn văn không ai nói."
                    ) from exc
                problems.append(f"{route.label}: {exc}")
                failures.append(exc)
                logger.warning("route %s hỏng trước token đầu: %s", route.label, exc)
            else:
                outcome = "success"
                return
            finally:
                if emitted and not charged:
                    # Stream đứt hoặc bị huỷ sau khi đã sinh chữ: token đã tiêu
                    # thật nhưng mẩu mang `usage` không bao giờ tới. Ghi phần đã
                    # giữ chỗ thay vì ghi 0 — một client ngắt kết nối ở mọi
                    # request sẽ **không bao giờ** chạm trần nếu ghi 0.
                    self._charge(reserved)
                route.breaker.record(outcome)
        raise AllRoutesFailed(self._all_failed(problems), failures=failures, skipped=skipped) from (
            failures[-1] if failures else None
        )

    # ------------------------------------------------------------------- phụ

    def _log_served(self, route: Route, served_model: str, cost_usd: float) -> None:
        """Quy tắc cứng #1: **model thực tế đã phục vụ**, mỗi request."""
        logger.info(
            "route=%s requested=%s served=%s cost=$%.6f fallback=%s",
            route.label,
            route.provider.model,
            served_model,
            cost_usd,
            route is not self.routes[0],
        )

    def _all_failed(self, problems: Sequence[str]) -> str:
        return "mọi route đều không phục vụ được: " + "; ".join(problems)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "routes": [
                {"label": r.label, "model": r.provider.model, **r.breaker.status()}
                for r in self.routes
            ],
            "budget": self.budget.status() if self.budget is not None else None,
        }

    def close(self) -> None:
        for route in self.routes:
            closer = getattr(route.provider, "close", None)
            if callable(closer):
                closer()
