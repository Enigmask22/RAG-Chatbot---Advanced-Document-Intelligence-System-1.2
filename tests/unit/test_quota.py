"""`TD-39` + `TD-47` — bộ đếm dùng chung, và ba cách nó có thể **làm tệ hơn**.

Đưa một bộ đếm lên Redis là **thêm một phụ thuộc vào đường nóng**. Nó chỉ đáng
tin khi chứng minh được ba điều **âm**:

1. **Redis hỏng ⇒ hệ không hỏng theo.** Đường lui phải là bộ đếm trong tiến
   trình đang chạy hôm nay — tức bảo đảm không bao giờ **tệ hơn** hiện tại.
2. **Redis TREO ⇒ hệ không chậm theo.** Đây là điều mà (1) không phủ, và là ca
   mà một bản vá mang đúng tên "fail-open" vẫn hỏng y như fail-closed.
3. **Bộ ngắt mạch không được tự khoá vĩnh viễn.** Một cơ chế thêm vào để một sự
   cố tạm thời đừng lan ra mà lại biến nó thành vĩnh viễn là tệ hơn không có.

⚠️ Tính **nguyên tử** của script Lua **không** kiểm được ở tầng này — một Redis
giả chỉ kiểm chính nó. Nó ở `tests/integration/test_quota_redis.py`, chạy trên
Redis thật.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_core.llm.router import CircuitBreaker
from serving.core.quota import RedisDailySpend, RedisRateLimiter
from serving.core.ratelimit import RateLimiter

pytestmark = pytest.mark.asyncio


class _RedisHong:
    """Mọi lệnh đều ném — Redis chết hẳn."""

    def __init__(self) -> None:
        self.calls = 0

    async def eval(self, *args: Any) -> Any:
        self.calls += 1
        raise ConnectionError("redis chết")

    async def incrbyfloat(self, *args: Any) -> Any:
        self.calls += 1
        raise ConnectionError("redis chết")

    async def expire(self, *args: Any) -> Any:
        self.calls += 1
        raise ConnectionError("redis chết")

    async def get(self, *args: Any) -> Any:
        self.calls += 1
        raise ConnectionError("redis chết")


class _RedisTot:
    """Đủ để đi hết đường vui; **không** mô phỏng Lua."""

    def __init__(self, allowed: int = 1, remaining: str = "9") -> None:
        self.allowed = allowed
        self.remaining = remaining
        self.store: dict[str, float] = {}
        self.expires: list[tuple[str, int]] = []

    async def eval(self, *args: Any) -> Any:
        return [self.allowed, self.remaining]

    async def incrbyfloat(self, key: str, amount: float) -> Any:
        self.store[key] = self.store.get(key, 0.0) + amount
        return self.store[key]

    async def expire(self, key: str, seconds: int) -> Any:
        self.expires.append((key, seconds))
        return 1

    async def get(self, key: str) -> Any:
        return self.store.get(key)


class TestRedisHongThiHeKhongHongTheo:
    async def test_han_muc_tut_ve_bo_dem_trong_tien_trinh(self) -> None:
        """⭐⭐ Mệnh đề trung tâm: đường lui **là** hành vi hôm nay, không phải
        "không có hàng rào". Bộ đếm cục bộ vẫn chặn ở đúng trần."""
        local = RateLimiter()
        rl = RedisRateLimiter(
            _RedisHong(),
            fallback=local,
            breaker=CircuitBreaker(failure_threshold=10**9, cooldown_s=1.0),
        )
        cho_qua = [(await rl.check("acme", 3)).allowed for _ in range(5)]
        assert cho_qua == [True, True, True, False, False]
        assert rl.degraded == 5

    async def test_tran_chi_phi_tut_ve_bo_dem_trong_tien_trinh(self) -> None:
        spend = RedisDailySpend(
            _RedisHong(),
            cap_usd=1.0,
            breaker=CircuitBreaker(failure_threshold=10**9, cooldown_s=1.0),
        )
        assert (await spend.charge("acme", 0.6)).allowed
        d = await spend.charge("acme", 0.6)
        assert not d.allowed and d.degraded and d.spent_usd == pytest.approx(1.2)

    async def test_degraded_dem_duoc_chu_khong_im_lang(self) -> None:
        """⚠️ Không có bộ đếm này thì "hạn mức đang đúng" và "hạn mức đang là N×
        và không ai biết" trông giống hệt nhau từ bên ngoài."""
        rl = RedisRateLimiter(_RedisHong(), fallback=RateLimiter())
        await rl.check("a", 100)
        assert rl.degraded == 1


class TestRedisTreoThiHeKhongChamTheo:
    async def test_mach_mo_thi_KHONG_goi_redis_nua(self) -> None:
        """⭐⭐ Điều mà đường lui **không** phủ, và là lý do quyết định 4 tồn tại.

        Redis từ chối kết nối thì đường lui rẻ. Redis **treo** thì mỗi request
        trả giá bằng cả socket timeout *trước khi* được tụt về — API không trả
        5xx nhưng chậm tới mức không dùng được. Bài này ghim rằng sau ngưỡng
        hỏng, **không còn lời gọi Redis nào** được phát đi.
        """
        redis = _RedisHong()
        rl = RedisRateLimiter(
            redis,
            fallback=RateLimiter(),
            breaker=CircuitBreaker(failure_threshold=3, cooldown_s=60),
        )
        for _ in range(20):
            await rl.check("acme", 10_000)
        assert redis.calls == 3, f"mạch mở rồi mà vẫn gọi Redis {redis.calls} lần"
        assert rl.degraded == 20

    async def test_tran_chi_phi_cung_ngat_mach(self) -> None:
        redis = _RedisHong()
        spend = RedisDailySpend(
            redis, cap_usd=10.0, breaker=CircuitBreaker(failure_threshold=3, cooldown_s=60)
        )
        for _ in range(20):
            await spend.charge("acme", 0.001)
        assert redis.calls == 3

    async def test_hai_bo_dem_KHONG_dung_chung_mot_mach(self) -> None:
        """`EVAL` và `INCRBYFLOAT` hỏng độc lập; một mạch dùng chung sẽ tắt cả
        hai vì lỗi của một."""
        rl = RedisRateLimiter(_RedisHong(), fallback=RateLimiter())
        spend = RedisDailySpend(_RedisTot(), cap_usd=10.0)
        for _ in range(10):
            await rl.check("a", 100)
        assert (await spend.charge("a", 0.1)).degraded is False


class TestMachKhongDuocTuKHOAVinhVien:
    async def test_KHONG_ghi_failure_khi_mach_da_mo(self) -> None:
        """⭐⭐ Bản đầu gọi `allow()` **bên trong** `try`, nên nhánh `except` chạy
        `record("failure")` mỗi request trong lúc mạch **đã** mở — mà
        `record("failure")` đặt lại `_opened_at`, tức đồng hồ nguội không bao
        giờ chạy hết và Redis **không bao giờ được thử lại**. Một cơ chế thêm
        vào để một sự cố tạm thời đừng lan ra, tự biến nó thành vĩnh viễn.

        ⚠️ Bản đầu của **bài test này** dùng `cooldown_s=0.0` và đếm số lời gọi
        Redis — và nó **không thể đỏ**, vì với cooldown 0 thì mạch vào half-open
        ngay cả khi đồng hồ vừa bị đặt lại. Đo thứ quan sát được trực tiếp
        (`record` có được gọi không) thay vì một hệ quả gián tiếp qua đồng hồ.
        """
        ghi: list[str] = []

        class _Giam(CircuitBreaker):
            def record(self, outcome: Any) -> None:
                ghi.append(outcome)
                super().record(outcome)

        redis = _RedisHong()
        spend = RedisDailySpend(
            redis, cap_usd=10.0, breaker=_Giam(failure_threshold=2, cooldown_s=60.0)
        )
        for _ in range(2):
            await spend.charge("acme", 0.01)
        assert ghi == ["failure", "failure"], ghi
        for _ in range(5):
            await spend.charge("acme", 0.01)
        assert ghi == ["failure", "failure"], f"mạch đã mở mà vẫn ghi thêm: {ghi}"

    async def test_han_muc_cung_the(self) -> None:
        ghi: list[str] = []

        class _Giam(CircuitBreaker):
            def record(self, outcome: Any) -> None:
                ghi.append(outcome)
                super().record(outcome)

        rl = RedisRateLimiter(
            _RedisHong(),
            fallback=RateLimiter(),
            breaker=_Giam(failure_threshold=2, cooldown_s=60.0),
        )
        for _ in range(7):
            await rl.check("acme", 10_000)
        assert ghi == ["failure", "failure"], f"mạch đã mở mà vẫn ghi thêm: {ghi}"

    async def test_mot_lan_thanh_cong_dong_mach_lai(self) -> None:
        redis = _RedisTot()
        breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.0)
        rl = RedisRateLimiter(redis, fallback=RateLimiter(), breaker=breaker)
        await rl.check("a", 100)
        assert breaker.state == "closed"


class TestDuongVui:
    async def test_quyet_dinh_doc_tu_ket_qua_cua_lua(self) -> None:
        rl = RedisRateLimiter(_RedisTot(allowed=1, remaining="7.5"), fallback=RateLimiter())
        d = await rl.check("a", 60)
        assert d.allowed and d.remaining == 7 and d.retry_after_s == 0

    async def test_bi_chan_thi_retry_after_toi_thieu_1_giay(self) -> None:
        """⭐ `Retry-After: 0` biến một header sinh ra để giảm tải thành một
        vòng lặp nóng — cùng lý lẽ đã có ở `RateLimiter.check`."""
        rl = RedisRateLimiter(_RedisTot(allowed=0, remaining="0.99"), fallback=RateLimiter())
        d = await rl.check("a", 6000)
        assert not d.allowed and d.retry_after_s >= 1

    async def test_khoa_mang_tenant_va_tien_to(self) -> None:
        ghi: list[Any] = []

        class _Ghi(_RedisTot):
            async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
                ghi.append(args[0])
                return [1, "5"]

        rl = RedisRateLimiter(_Ghi(), fallback=RateLimiter(), prefix="rl")
        await rl.check("acme", 60)
        assert ghi == ["rl:acme"]

    async def test_limit_duoi_1_van_la_loi_lap_trinh(self) -> None:
        rl = RedisRateLimiter(_RedisTot(), fallback=RateLimiter())
        with pytest.raises(ValueError, match="≥ 1"):
            await rl.check("a", 0)


class TestTranChiPhiTheoTenant:
    async def test_mot_tenant_dot_het_KHONG_chan_tenant_khac(self) -> None:
        """⭐⭐ Nửa thứ hai của `TD-47`, và là nửa mà việc lên Redis **không** tự
        giải quyết: `DailyBudget` toàn cục chặn *mọi* tenant khi một tenant tiêu
        hết. Khoá phải mang `tenant_id`."""
        spend = RedisDailySpend(_RedisTot(), cap_usd=1.0)
        assert not (await spend.charge("acme", 2.0)).allowed
        assert (await spend.charge("globex", 0.1)).allowed

    async def test_khoa_doi_theo_ngay_UTC(self) -> None:
        from datetime import UTC, datetime

        spend = RedisDailySpend(_RedisTot(), cap_usd=1.0)
        assert spend._key("acme").endswith(datetime.now(UTC).date().isoformat())

    async def test_cap_0_nghia_la_khong_tran_va_khong_cham_redis(self) -> None:
        redis = _RedisTot()
        spend = RedisDailySpend(redis, cap_usd=0.0)
        assert (await spend.charge("acme", 999.0)).allowed
        assert redis.store == {}, "cap=0 không được ghi gì lên Redis"

    async def test_peek_KHONG_ghi_nhan(self) -> None:
        """`/admin` phải hỏi được số đã tiêu mà không làm nó tăng."""
        redis = _RedisTot()
        spend = RedisDailySpend(redis, cap_usd=1.0)
        await spend.charge("acme", 0.4)
        truoc = dict(redis.store)
        d = await spend.peek("acme")
        assert redis.store == truoc
        assert d.spent_usd == pytest.approx(0.4) and d.remaining_usd == pytest.approx(0.6)

    async def test_co_dat_TTL_de_khoa_ngay_cu_khong_tich_lai(self) -> None:
        redis = _RedisTot()
        spend = RedisDailySpend(redis, cap_usd=1.0)
        await spend.charge("acme", 0.1)
        assert redis.expires and redis.expires[0][1] > 86_400


class TestMacDinh:
    async def test_quota_shared_mac_dinh_BAT(self) -> None:
        """⭐ Một hạng mục về hạn mức phân tán mà mặc định tắt là một hạng mục
        không chạy ở production. Ghim con số, không ghim ý định."""
        from rag_core.settings import Settings

        assert Settings().quota_shared is True
