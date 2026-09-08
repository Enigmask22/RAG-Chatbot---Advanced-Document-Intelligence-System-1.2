"""`TD-39` + `TD-47` trên **Redis thật** — vì cái đang được kiểm là script Lua.

Bộ test đơn vị (`tests/unit/test_quota.py`) kiểm đường lui, bộ ngắt mạch và
hình dạng quyết định — tất cả đều đúng với một Redis giả. Thứ nó **không thể**
kiểm là chính lý do hạng mục này tồn tại:

* script Lua có chạy được không (cú pháp, kiểu trả về, `redis.call('TIME')`);
* hai tiến trình dùng chung **một** bucket có ra **một** trần không, hay ra hai;
* `INCRBYFLOAT` có thật sự cộng dồn qua các tiến trình không.

Một Redis giả trả lời cả ba câu ấy bằng chính mã của nó. Nên chúng ở đây.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import redis as redis_sync
import redis.asyncio as aioredis

from serving.core.quota import RedisDailySpend, RedisRateLimiter
from serving.core.ratelimit import RateLimiter

pytestmark = pytest.mark.integration

REDIS_URL = "redis://127.0.0.1:6379/0"
PREFIX = "test-quota"


@pytest.fixture(autouse=True)
def _don_khoa() -> Any:
    """Mỗi bài bắt đầu từ Redis sạch **phần của nó**.

    ⚠️ Không `FLUSHDB`: DB 0 cũng là chỗ semantic cache và `arq` đang dùng, và
    một bài test xoá sạch DB của hàng xóm là một bài test làm đỏ bài khác ở
    chỗ khác — kiểu phụ thuộc đắt nhất lúc gỡ.
    """

    def _xoa() -> None:
        client = redis_sync.Redis.from_url(REDIS_URL)
        for pattern in (f"{PREFIX}:*", f"{PREFIX}-spend:*"):
            for key in client.scan_iter(pattern):
                client.delete(key)
        client.close()

    _xoa()
    yield
    _xoa()


def _limiter(client: Any) -> RedisRateLimiter:
    return RedisRateLimiter(client, fallback=RateLimiter(), prefix=PREFIX)


class TestMotTranChuKhongPhaiNTran:
    @pytest.mark.asyncio
    async def test_HAI_replica_dung_chung_MOT_bucket(self) -> None:
        """⭐⭐ Đây là toàn bộ `TD-39`, viết thành một phép khẳng định.

        Hai `RedisRateLimiter` = hai tiến trình. Với bộ đếm trong tiến trình,
        trần 4/phút cho ra **8** lượt qua; với bucket dùng chung nó phải cho ra
        đúng **4**.
        """
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            a, b = _limiter(client), _limiter(client)
            qua = 0
            for i in range(10):
                rl = a if i % 2 == 0 else b
                if (await rl.check("acme", 4)).allowed:
                    qua += 1
            assert qua == 4, f"{qua} lượt qua — trần đang là bội của số replica"
            assert a.degraded == 0 and b.degraded == 0, "đã tụt về đường lui, không đo Redis"
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_10_request_DONG_THOI_qua_dung_bang_tran(self) -> None:
        """⭐⭐ Đọc–sửa–ghi qua ba lệnh sẽ hỏng **ở đây** chứ không ở bài trên:
        tuần tự thì một cuộc đua không bao giờ xảy ra. `EVAL` chạy nguyên tử
        trên server, nên phép trừ token là một bước không chia được."""
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            limiters = [_limiter(client) for _ in range(10)]
            ket_qua = await asyncio.gather(*(rl.check("acme", 3) for rl in limiters))
            assert sum(d.allowed for d in ket_qua) == 3
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_tenant_khac_nhau_khong_dung_chung_bucket(self) -> None:
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            rl = _limiter(client)
            for _ in range(2):
                await rl.check("acme", 2)
            assert not (await rl.check("acme", 2)).allowed
            assert (await rl.check("globex", 2)).allowed
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_bucket_co_TTL_de_tenant_im_lang_khong_tich_lai_mai(self) -> None:
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            await _limiter(client).check("acme", 5)
            assert 0 < await client.ttl(f"{PREFIX}:acme") <= 120
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_dong_ho_den_tu_REDIS_chu_khong_tu_tien_trinh(self) -> None:
        """⭐ Script ghi `updated` bằng `redis.call('TIME')`. Nếu nó lấy đồng hồ
        của tiến trình gọi thì hai replica lệch giờ sẽ làm bucket nhảy tới nhảy
        lui, và một replica chạy nhanh vài giây tự nạp đầy bucket của mình mỗi
        lần gọi — tức trần biến mất mà không ai thấy lỗi nào."""
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            await _limiter(client).check("acme", 5)
            updated = float(await client.hget(f"{PREFIX}:acme", "updated"))
            now_redis = (await client.time())[0]
            assert abs(updated - now_redis) < 5.0
        finally:
            await client.aclose()


class TestChiPhiNgayDungChung:
    @pytest.mark.asyncio
    async def test_HAI_replica_cong_vao_MOT_bo_dem(self) -> None:
        """⭐⭐ `TD-47` nửa thứ nhất: N replica ⇒ trần thật N×."""
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            a = RedisDailySpend(client, cap_usd=1.0, prefix=f"{PREFIX}-spend")
            b = RedisDailySpend(client, cap_usd=1.0, prefix=f"{PREFIX}-spend")
            assert (await a.charge("acme", 0.6)).allowed
            d = await b.charge("acme", 0.6)
            assert not d.allowed, "replica thứ hai không thấy phần đã tiêu của replica thứ nhất"
            assert d.spent_usd == pytest.approx(1.2) and not d.degraded
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_mot_tenant_dot_het_KHONG_chan_tenant_khac(self) -> None:
        """`TD-47` nửa thứ hai, và là nửa mà việc lên Redis **không** tự giải
        quyết — nó là chuyện của **khoá**."""
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            spend = RedisDailySpend(client, cap_usd=1.0, prefix=f"{PREFIX}-spend")
            assert not (await spend.charge("acme", 5.0)).allowed
            assert (await spend.charge("globex", 0.1)).allowed
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_peek_thay_dung_so_da_tieu_ma_khong_lam_no_tang(self) -> None:
        client = aioredis.from_url(REDIS_URL)  # type: ignore[no-untyped-call]
        try:
            spend = RedisDailySpend(client, cap_usd=2.0, prefix=f"{PREFIX}-spend")
            await spend.charge("acme", 0.25)
            assert (await spend.peek("acme")).spent_usd == pytest.approx(0.25)
            assert (await spend.peek("acme")).spent_usd == pytest.approx(0.25)
        finally:
            await client.aclose()


class TestRedisThatSuKHONGCoThiVanChay:
    @pytest.mark.asyncio
    async def test_cong_sai_thi_tut_ve_duong_lui_chu_khong_nem(self) -> None:
        """Redis thật, cổng sai — gần nhất với "Redis chết" mà không phải tắt
        service của các bài khác."""
        client = aioredis.from_url("redis://127.0.0.1:1/0")  # type: ignore[no-untyped-call]
        try:
            rl = _limiter(client)
            assert (await rl.check("acme", 2)).allowed
            assert rl.degraded == 1
        finally:
            await client.aclose()
