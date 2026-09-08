"""`TD-29` — đường "worker chết hẳn", kiểm bằng cách **giết tiến trình thật**.

`W3-08` có test cho ba trong bốn đường quay lại hàng đợi; đường thứ tư (worker
chết giữa job) tới giờ chỉ được xác minh bằng đọc mã nguồn arq, và bài
`test_attempt_lon_hon_1_di_vao_trang_thai` dựng lại **hậu quả** của nó bằng cách
đặt trước khoá đếm. Ở đây dựng **nguyên nhân**: một worker chạy ở tiến trình
con, bị giết cứng giữa chừng (`TerminateProcess` trên Windows, `SIGKILL` trên
POSIX), và worker thứ hai phải nhặt lại đúng job ấy với `job_try = 2`.

## ⭐⭐ Tiền đề của dòng nợ SAI — không cần "một đường code chỉ tồn tại cho test"

`TD-29` ghi: *"job phải đủ chậm để còn đang chạy lúc bị giết […] nên cần một
job cố ý chậm, tức một đường code chỉ tồn tại cho test."* Không đúng: job chậm
sống **trong module test này** và được đăng ký vào một worker của test, trên
một hàng đợi riêng. Thứ đang kiểm — khoá `in-progress` hết hạn, `job_try` được
`INCR` ở lần nhặt sau — là hành vi của **arq**, giống hệt nhau cho mọi hàm
được đăng ký; mã production không cần thêm nhánh nào. Đây là hạng mục thứ tư
trong hai ngày mà dòng nợ chứa sẵn một chẩn đoán chưa đo (`NEW-14`, `NEW-12`,
`TD-39`, và đây).

## Vì sao bài này chậm (~15–20 s), và vì sao con số ấy không giảm được

arq đặt khoá `in-progress:{job_id}` với hạn cứng `job_timeout + 10 s`
(`arq/worker.py:277`, hằng `+10` không cấu hình được). Worker chết không nhả
khoá — phục hồi CHỈ xảy ra khi khoá hết hạn, nên bài test phải chờ trọn TTL ấy.
`job_timeout = 5 s` ở đây cho TTL 15 s: nhỏ nhất có thể mà vẫn giữ khoảng cách
an toàn giữa "nhặt job" và "bị giết" (xem chú thích ⚠️ trong bài). Đó cũng chính
là con số vận hành mà `TD-29` cảnh báo: production để `job_timeout = 2 giờ`,
tức một worker chết làm job nằm im **tới 2 giờ** trước khi ai đó nhặt lại.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import redis as redis_sync
from arq.connections import RedisSettings, create_pool
from arq.worker import Worker

pytestmark = pytest.mark.integration

REDIS_URL = "redis://127.0.0.1:6379/0"
QUEUE = "arq:td29"
JOB_ID = "td29deadworker"
TRIES_KEY = "td29:tries"
#: `job_timeout` của worker TEST (không phải production). Khoá in-progress sẽ
#: sống `JOB_TIMEOUT_S + 10` giây — chính là thời gian bài test phải chờ.
JOB_TIMEOUT_S = 5
_REPO_ROOT = Path(__file__).resolve().parents[2]


async def viec_cham(ctx: dict[str, Any]) -> str:
    """Ngủ 60 s ở lần nhặt ĐẦU — đủ lâu để chắc chắn còn chạy lúc bị giết.

    Lần nhặt thứ hai trả về ngay: 60 s > `job_timeout` nên một lần chạy trọn
    vẹn sẽ bị chính arq huỷ vì quá giờ — nhưng lần đầu không bao giờ chạy trọn
    (worker bị giết trước), và lần hai không ngủ.
    """
    await ctx["redis"].rpush(TRIES_KEY, ctx["job_try"])
    if ctx["job_try"] == 1:
        await asyncio.sleep(60)
    return "song_sot_lan_hai"


def _worker_main() -> None:
    """Điểm vào của tiến trình con — worker arq thật, chạy tới khi bị giết."""
    worker = Worker(
        functions=[viec_cham],
        redis_settings=RedisSettings.from_dsn(REDIS_URL),
        queue_name=QUEUE,
        job_timeout=JOB_TIMEOUT_S,
        poll_delay=0.05,
        burst=False,
    )
    worker.run()


async def _tries(pool: Any) -> list[bytes]:
    """`redis-py` khai `lrange` là `Awaitable[list] | list` — ép kiểu một chỗ."""
    return list(await pool.lrange(TRIES_KEY, 0, -1))


def _don_dep() -> None:
    client = redis_sync.Redis.from_url(REDIS_URL)
    try:
        client.delete(
            TRIES_KEY,
            QUEUE,
            f"arq:job:{JOB_ID}",
            f"arq:result:{JOB_ID}",
            f"arq:retry:{JOB_ID}",
            f"arq:in-progress:{JOB_ID}",
        )
    finally:
        client.close()


@pytest.fixture(autouse=True)
def _khoa_sach() -> Any:
    _don_dep()
    yield
    _don_dep()


class TestWorkerChetHan:
    @pytest.mark.asyncio
    async def test_job_song_sot_qua_mot_worker_bi_giet_cung(self) -> None:
        """⭐⭐ Toàn bộ `TD-29` trong một kịch bản: nhặt → giết → hết hạn khoá →
        worker thứ hai nhặt lại với `job_try = 2`, không đặt trước khoá nào."""
        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL), default_queue_name=QUEUE)
        proc: subprocess.Popen[bytes] | None = None
        try:
            await pool.enqueue_job("viec_cham", _job_id=JOB_ID)

            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from tests.integration.test_worker_death import _worker_main; _worker_main()",
                ],
                cwd=_REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 1) Chờ worker con NHẶT job — bằng chứng là khoá in-progress.
            in_progress = f"arq:in-progress:{JOB_ID}"
            deadline = time.monotonic() + 30.0
            while not await pool.exists(in_progress):
                assert time.monotonic() < deadline, "worker con không nhặt job sau 30 s"
                assert proc.poll() is None, f"worker con chết sớm, mã {proc.returncode}"
                await asyncio.sleep(0.05)

            # 2) Ghim con số vận hành của `TD-29` bằng PHÉP ĐO, không bằng trích
            #    dẫn mã nguồn: TTL của khoá phải là `job_timeout + 10 s` — đây
            #    chính là "phục hồi mất tới job_timeout + 10 s" đo được.
            ttl_ms = await pool.pttl(in_progress)
            assert (JOB_TIMEOUT_S + 9) * 1000 < ttl_ms <= (JOB_TIMEOUT_S + 10) * 1000, (
                f"TTL khoá in-progress = {ttl_ms} ms — công thức job_timeout+10s đã đổi?"
            )

            # 3) Giết CỨNG. ⚠️ Ngay sau đó khoá phải CÒN — nếu nó đã biến mất
            #    thì job vừa kết thúc theo đường khác (vd. arq huỷ vì quá giờ)
            #    và phần còn lại của bài sẽ xanh mà không có worker nào chết
            #    giữa chừng cả — một bài test đúng kết luận, sai kịch bản.
            proc.kill()
            proc.wait(timeout=10)
            assert await pool.exists(in_progress), (
                "khoá in-progress biến mất ngay lúc giết — job không còn đang chạy"
            )
            assert await _tries(pool) == [b"1"], "lần nhặt đầu phải đã xảy ra, và chỉ một lần"

            # 4) Khoá không được nhả bởi cái chết — nó chỉ HẾT HẠN. Chờ trọn.
            deadline = time.monotonic() + JOB_TIMEOUT_S + 10.0 + 15.0
            while await pool.exists(in_progress):
                assert time.monotonic() < deadline, "khoá in-progress không hết hạn"
                await asyncio.sleep(0.2)

            # 5) Worker thứ hai — trong tiến trình, burst — nhặt lại và chạy nốt.
            hoi_sinh = Worker(
                functions=[viec_cham],
                redis_pool=pool,
                queue_name=QUEUE,
                job_timeout=JOB_TIMEOUT_S,
                poll_delay=0.05,
                burst=True,
            )
            try:
                assert await hoi_sinh.run_check(max_burst_jobs=5) == 1
            finally:
                await hoi_sinh.close()

            assert await _tries(pool) == [b"1", b"2"], (
                "lần nhặt thứ hai phải mang job_try = 2 — arq INCR ở mỗi lần nhặt"
            )
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
            await pool.aclose()
