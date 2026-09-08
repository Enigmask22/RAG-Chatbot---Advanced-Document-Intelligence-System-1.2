"""Hai bộ đếm dùng chung giữa các replica: nhịp (`TD-39`) và chi phí (`TD-47`).

## Vấn đề, nói bằng con số

Cả hai bộ đếm hôm nay sống trong **bộ nhớ của một tiến trình**:

* `RateLimiter` (`W4-04`): 4 replica ⇒ mỗi tenant được **240** request/phút chứ
  không phải 60, và con số ấy đổi mỗi lần autoscale. `uvicorn --workers N` y hệt.
* `DailyBudget` (`W4-08`): N replica ⇒ trần thật là **N×**, restart ⇒ bộ đếm về
  **0**. Và nó **không phân theo tenant** — một tenant đốt hết ngân sách thì
  *mọi* tenant còn lại nhận `429`.

## ⭐⭐ Quyết định 1: "Redis chết thì mở cổng hay đóng cổng" là một câu hỏi sai

Dòng nợ `TD-39` đặt nó như một lựa chọn nhị phân — *"mở = mất hạn mức đúng lúc
hệ thống đang yếu, đóng = một phụ thuộc mới có thể làm sập toàn bộ API"* — và
bảo phải quyết trước khi viết. Quyết được, nhưng câu trả lời là **cả hai đều
sai**, vì có một lựa chọn thứ ba mà dòng nợ không xét:

> **Redis hỏng ⇒ tụt về đúng bộ đếm trong tiến trình đang chạy hôm nay.**

Nó **không** phải "mở cổng": trần vẫn còn, chỉ là N× quá rộng — tức **đúng bằng
bảo đảm hiện tại, không bao giờ tệ hơn**. Và nó không biến một sự cố Redis
thành một sự cố API.

⚠️ Điều đó cũng sửa lại lý lẽ của `TD-47` (*"quyết định fail-open vs fail-closed
phải giống nhau ở cả hai"*). Hai câu trả lời **giống nhau**, nhưng không phải vì
chúng buộc phải giống — mà vì **cả hai đều đã có sẵn một hàng rào cục bộ có
biên**. Nếu một trong hai không có, câu trả lời của nó đã khác.

## ⭐⭐ Quyết định 2: token bucket phải nằm trong **một** script Lua

Đọc–sửa–ghi qua ba lệnh Redis tái lập **đúng** cuộc đua mà hạng mục này sinh ra
để đóng: hai replica cùng đọc `tokens=1`, cùng thấy còn, cùng cho qua. `EVAL`
chạy nguyên tử trên server, nên phép trừ token là một bước không chia được.

Cùng hình dạng với `join()` của `NEW-10` (*"đồng bộ — không `await` giữa tra và
ghi"*) và với `check()`+`commit()` nguyên tử của `W6-02`. Ba lần trong dự án
này, cùng một luật: **một hàng rào có khoảng hở giữa "tra" và "ghi" thì không
phải hàng rào.**

## ⭐⭐ Quyết định 4: "mở cổng" phải **nhanh**, không thì nó là "đóng cổng chậm"

Quyết định 1 chưa đủ. Redis **từ chối kết nối** thì đường lui rẻ (vài trăm µs
trên localhost), nhưng Redis **treo** — phân vùng mạng, node bị đóng băng, đĩa
đầy — thì mỗi request trả giá bằng cả socket timeout **trước khi** được tụt về
bộ đếm cục bộ. Với timeout mặc định của `redis-py`, đó là hàng giây cho **mọi**
request: API không trả 5xx nhưng chậm tới mức không dùng được.

Đó **đúng là** ca *"một phụ thuộc mới có thể làm sập toàn bộ API"* mà dòng nợ
`TD-39` cảnh báo — nó chỉ không sập theo cách người ta hình dung. Một bản vá
dừng ở quyết định 1 sẽ mang đúng tên "fail-open" mà vẫn hỏng y như fail-closed.

Nên: một `CircuitBreaker` (`W4-08`, đã có và đã có test) trước mọi lời gọi
Redis. Hỏng liên tiếp ⇒ mở mạch ⇒ những request sau **không gọi Redis nữa**, đi
thẳng đường lui trong micro giây, và cứ mỗi `cooldown_s` thả **một** request đi
thử. Không viết bộ ngắt mạch thứ hai: cái đang có đã đúng hình dạng này.

## ⭐⭐ Quyết định 3: trần theo tenant nằm ở tầng **serving**, không ở `rag_core`

`TD-47` muốn ngân sách phân theo tenant. Cách hiển nhiên — cho `DailyBudget`
biết `tenant_id` — **phá ranh giới hai plane**: `DailyBudget` sống trong
`rag_core/llm/router.py`, nơi phục vụ **cả đường eval**, và ở đường ấy không có
tenant nào cả. `NEW-01` có test AST chặn đúng chiều phụ thuộc này.

Nên: trần theo tenant áp ở **serving**, trước khi gọi router; trần toàn cục của
router **ở lại** làm hàng rào thô thứ hai. Cùng lý lẽ với `tenant_filter()` —
tenancy là khái niệm của biên HTTP, không của lõi truy hồi.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from rag_core.llm.router import CircuitBreaker

if TYPE_CHECKING:
    from serving.core.ratelimit import Decision, RateLimiter

__all__ = [
    "RedisDailySpend",
    "RedisRateLimiter",
    "SpendDecision",
]

logger = logging.getLogger(__name__)

#: Token bucket nguyên tử. `KEYS[1]` = khoá bucket; `ARGV` = trần/phút, thời
#: điểm hiện tại (giây, float), TTL.
#:
#: ⚠️ Đồng hồ đến **từ Redis** (`TIME`), không từ tiến trình gọi: lệch đồng hồ
#: giữa các replica sẽ làm bucket nhảy tới nhảy lui, và một replica chạy nhanh
#: 5 giây sẽ tự nạp đầy bucket của mình mỗi lần gọi. Một bộ đếm dùng chung phải
#: có **một** đồng hồ.
_LUA_TOKEN_BUCKET = """
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local refill = limit / 60.0

local now_arr = redis.call('TIME')
local now = tonumber(now_arr[1]) + tonumber(now_arr[2]) / 1000000.0

local data = redis.call('HMGET', KEYS[1], 'tokens', 'updated')
local tokens = tonumber(data[1])
local updated = tonumber(data[2])

if tokens == nil or updated == nil then
  tokens = limit
  updated = now
else
  local delta = now - updated
  if delta < 0 then delta = 0 end
  tokens = math.min(limit, tokens + delta * refill)
  updated = now
end

local allowed = 0
if tokens >= 1.0 then
  tokens = tokens - 1.0
  allowed = 1
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated', updated)
redis.call('EXPIRE', KEYS[1], ttl)
return {allowed, tostring(tokens)}
"""


class _RedisLike(Protocol):
    async def eval(self, script: str, numkeys: int, *args: Any) -> Any: ...
    async def incrbyfloat(self, key: str, amount: float) -> Any: ...
    async def expire(self, key: str, seconds: int) -> Any: ...
    async def get(self, key: str) -> Any: ...


def _decision(allowed: bool, remaining: float, limit: int) -> Decision:
    from serving.core.ratelimit import Decision as _D

    if allowed:
        return _D(True, int(remaining), 0, limit)
    # `ceil` với sàn 1 — cùng lý lẽ với `RateLimiter.check`: `Retry-After: 0`
    # biến header giảm tải thành một vòng lặp nóng.
    wait = (1.0 - remaining) / (limit / 60.0)
    return _D(False, 0, max(1, math.ceil(wait)), limit)


class RedisRateLimiter:
    """Token bucket dùng chung, tụt về `fallback` khi Redis hỏng.

    ⚠️ `fallback` là **bắt buộc**, không phải tuỳ chọn. Một hạn mức phân tán
    không có đường lui là một cách biến sự cố Redis thành sự cố API — xem quyết
    định 1 ở docstring module. Kiểu của nó là `RateLimiter` hiện có, nên đường
    lui **đúng bằng** hành vi hôm nay.
    """

    def __init__(
        self,
        redis: _RedisLike,
        *,
        fallback: RateLimiter,
        prefix: str = "rl",
        ttl_s: int = 120,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.redis = redis
        self.fallback = fallback
        self.prefix = prefix
        self.ttl_s = ttl_s
        self.breaker = breaker or CircuitBreaker(failure_threshold=3, cooldown_s=10.0)
        self.degraded = 0
        """Số lần phải tụt về bộ đếm cục bộ. Không phải chỉ số trang trí: nó là
        cách duy nhất phân biệt *"hạn mức đang đúng"* với *"hạn mức đang là N×
        và không ai biết"*."""

    async def check(self, key: str, limit_per_minute: int) -> Decision:
        if limit_per_minute < 1:
            raise ValueError(f"limit_per_minute phải ≥ 1, nhận {limit_per_minute}")
        if not self.breaker.allow():
            # Mạch đang mở: **không** gọi Redis. Xem quyết định 4 — đây là chỗ
            # phân biệt "mở cổng" với "đóng cổng chậm".
            self.degraded += 1
            return self.fallback.check(key, limit_per_minute)
        try:
            raw = await self.redis.eval(
                _LUA_TOKEN_BUCKET, 1, f"{self.prefix}:{key}", limit_per_minute, self.ttl_s
            )
            allowed, remaining = int(raw[0]), float(raw[1])
        except Exception:
            self.breaker.record("failure")
            self.degraded += 1
            logger.warning(
                "hạn mức nhịp: Redis hỏng, tụt về bộ đếm TRONG TIẾN TRÌNH "
                "(trần thật = N× số replica) — lần thứ %d",
                self.degraded,
                exc_info=self.degraded == 1,
            )
            return self.fallback.check(key, limit_per_minute)
        self.breaker.record("success")
        return _decision(bool(allowed), remaining, limit_per_minute)


@dataclass(frozen=True)
class SpendDecision:
    """Kết quả một phép hỏi ngân sách. `spent_usd` là **sau** khi ghi nhận."""

    allowed: bool
    spent_usd: float
    cap_usd: float
    degraded: bool

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)


class RedisDailySpend:
    """Chi tiêu ngày **theo tenant**, dùng chung giữa các replica.

    ⭐ Khác `DailyBudget` ở hai chỗ, và cả hai là nội dung của `TD-47`:

    1. Khoá gồm `tenant_id`, nên một tenant đốt hết phần của mình **không** làm
       tenant khác nhận `429`.
    2. Bộ đếm ở Redis, nên restart **không** đưa nó về 0 và N replica vẫn cộng
       vào một chỗ.

    ⚠️ Nó **cộng trước, hỏi sau** (`INCRBYFLOAT` rồi so), chứ không hỏi-rồi-cộng.
    Đó là lựa chọn có chủ đích: hỏi trước để lại đúng khoảng hở giữa "tra" và
    "ghi" mà quyết định 2 nói tới, và với **tiền** thì lỗi vượt trần một chút
    rẻ hơn nhiều so với lỗi tính hai lần hoặc không tính.
    """

    def __init__(
        self,
        redis: _RedisLike,
        *,
        cap_usd: float,
        prefix: str = "spend",
        ttl_s: int = 172_800,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.redis = redis
        self.cap_usd = cap_usd
        self.prefix = prefix
        self.ttl_s = ttl_s
        # Bộ ngắt mạch **riêng** cho bộ đếm này, không dùng chung với limiter:
        # hai lệnh khác nhau (`EVAL` vs `INCRBYFLOAT`) hỏng độc lập, và một
        # mạch dùng chung sẽ tắt cả hai vì lỗi của một.
        self.breaker = breaker or CircuitBreaker(failure_threshold=3, cooldown_s=10.0)
        self.degraded = 0
        self._local: dict[str, float] = {}

    @property
    def unlimited(self) -> bool:
        return self.cap_usd <= 0

    def _key(self, tenant: str) -> str:
        # Ngày **UTC**, cùng lý lẽ với `DailyBudget`: giờ máy nghĩa là mốc reset
        # trôi theo giờ mùa hè, và một hoá đơn có 23 giờ trong đó.
        return f"{self.prefix}:{tenant}:{datetime.now(UTC).date().isoformat()}"

    async def charge(self, tenant: str, amount_usd: float) -> SpendDecision:
        """Ghi nhận `amount_usd` rồi cho biết tenant ấy còn được đi tiếp không."""
        if self.unlimited:
            return SpendDecision(True, 0.0, self.cap_usd, degraded=False)
        key = self._key(tenant)
        if not self.breaker.allow():
            return self._cuc_bo(key, amount_usd)
        try:
            spent = float(await self.redis.incrbyfloat(key, amount_usd))
            await self.redis.expire(key, self.ttl_s)
            self.breaker.record("success")
            degraded = False
        except Exception:
            self.breaker.record("failure")
            self.degraded += 1
            logger.warning(
                "trần chi phí ngày: Redis hỏng, tụt về bộ đếm TRONG TIẾN TRÌNH "
                "(trần thật = N× số replica, và restart đưa về 0) — lần thứ %d",
                self.degraded,
                exc_info=self.degraded == 1,
            )
            return self._cuc_bo(key, amount_usd)
        return SpendDecision(spent <= self.cap_usd, spent, self.cap_usd, degraded)

    def _cuc_bo(self, key: str, amount_usd: float) -> SpendDecision:
        """Đường lui: bộ đếm trong tiến trình — đúng hành vi của `DailyBudget`.

        ⚠️ Tách thành hàm riêng **vì một lỗi thật**. Bản đầu gọi `allow()` bên
        trong `try` và ném khi mạch mở, nên nhánh `except` chạy `record("failure")`
        **mỗi request trong lúc mạch đã mở** — mà `record("failure")` đặt lại
        `_opened_at`, tức đồng hồ nguội không bao giờ chạy hết và Redis **không
        bao giờ được thử lại**. Một bộ ngắt mạch thêm vào để một sự cố tạm thời
        đừng lan ra, tự biến sự cố tạm thời ấy thành **vĩnh viễn**.
        """
        spent = self._local.get(key, 0.0) + amount_usd
        self._local[key] = spent
        return SpendDecision(spent <= self.cap_usd, spent, self.cap_usd, degraded=True)

    async def peek(self, tenant: str) -> SpendDecision:
        """Hỏi mà **không** ghi nhận — cho `/admin` và cho phép chặn trước khi
        gọi nhà cung cấp, thay vì chỉ phát hiện sau khi đã tiêu."""
        if self.unlimited:
            return SpendDecision(True, 0.0, self.cap_usd, degraded=False)
        key = self._key(tenant)
        if not self.breaker.allow():
            spent = self._local.get(key, 0.0)
            return SpendDecision(spent <= self.cap_usd, spent, self.cap_usd, degraded=True)
        try:
            raw = await self.redis.get(key)
            spent = float(raw) if raw is not None else 0.0
            self.breaker.record("success")
            degraded = False
        except Exception:
            self.breaker.record("failure")
            self.degraded += 1
            spent = self._local.get(key, 0.0)
            degraded = True
        return SpendDecision(spent <= self.cap_usd, spent, self.cap_usd, degraded)
