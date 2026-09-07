"""Trần chi tiêu cho một demo công khai **không có xác thực**. `W6-02`.

Mọi hàng rào của hệ thống thật đều nằm sau một API key (`W4-04`), và §8 của
`reports/tasks/security-final.md` đã liệt kê: gần như mọi hàng rào hạ tầng còn
lại là `127.0.0.1`. Space không có cái nào trong hai thứ đó. Cái duy nhất đứng
giữa một người lạ và hoá đơn DeepSeek của chủ tài khoản là file này.

## Bốn quyết định, và cái thứ hai là cái phải đọc

**1. Đếm TRƯỚC khi gọi model, không phải sau khi có câu trả lời.** Đếm-khi-thành-
công nghe công bằng hơn với người dùng và nó sai theo hướng đắt tiền: một lượt
sinh chết ở giữa vẫn đã tiêu token, và nếu nó không được tính thì một vòng lặp
thử-lại là **miễn phí vô hạn**. `commit()` gộp xin phép và ghi nhận vào một
thao tác — người gọi cam kết trước, và không có đường nào hoàn lại.

**2. ⭐⭐ Trần theo IP là KHUYẾN CÁO, trần tổng mới là hàng rào.** Space đứng sau
proxy của HF, nên `request.client.host` là địa chỉ của proxy — dùng nó thì mọi
khách trở thành **một** người và trần theo IP hoặc vô dụng hoặc chặn tất cả.
Địa chỉ thật nằm ở `x-forwarded-for`, và header ấy **người gọi ghi được**: edge
nối thêm vào chứ không xoá, nên phần tử trái nhất là thứ client tự khai. Nói
cách khác: một người muốn vượt trần theo IP chỉ cần đổi một header.

Điều đó **không** làm trần theo IP vô nghĩa — nó chặn lượt lạm dụng vô tình và
lượt bấm-liên-tục, vốn là đa số. Nhưng nó có nghĩa là con số bảo vệ hoá đơn là
`daily_total`, và đó là con số duy nhất nên dùng để suy ra chi phí tối đa. Viết
ra ở đây thay vì để người sau tự phát hiện khi đọc hoá đơn.

**3. Chỉ giữ hash CÓ MUỐI của IP, không giữ IP.** Địa chỉ IP là dữ liệu cá nhân;
một demo công khai không có lý do gì để giữ nó, và bộ đếm chỉ cần biết "có phải
cùng một người không". Muối sinh ngẫu nhiên mỗi lần khởi động ⇒ hash không đối
chiếu được giữa hai lần chạy, kể cả bởi chính chủ Space.

**4. Bộ đếm nằm trong RAM, và một lần khởi động lại xoá nó.** Space miễn phí
không có ổ đĩa bền. Hệ quả thật: trần là "mỗi vòng đời tiến trình" chứ không
phải "mỗi ngày lịch" — chạm trần rồi Space ngủ và thức lại thì trần mở lại.
Không ai ngoài chủ Space kích được việc khởi động lại, nên đây là suy giảm chấp
nhận được; nó vẫn là một cách con số `daily_total` nói dối, nên nó được nói ra.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["SpendGuard", "Verdict", "client_key_of", "guard_from_env", "hash_client"]

#: Muối cho hash IP. Ngẫu nhiên mỗi tiến trình — xem quyết định 3.
_SALT = secrets.token_bytes(16)


def hash_client(raw: str) -> str:
    """IP thô -> 16 ký tự hex. Một chiều, và không so được giữa hai lần chạy."""
    return hashlib.blake2b(raw.encode("utf-8"), key=_SALT, digest_size=8).hexdigest()


def client_key_of(headers: object, fallback_host: str | None) -> str:
    """Khoá đếm cho một request Gradio.

    ⚠️ Ưu tiên `x-forwarded-for` **dù nó giả được** (quyết định 2): không dùng
    nó thì mọi khách gộp thành một khoá và trần theo IP mất hết ý nghĩa — hỏng
    chắc chắn, thay cho hỏng khi bị tấn công. Lấy phần tử **trái nhất** vì đó là
    chỗ edge của HF đặt địa chỉ khách; phần tử phải nhất đáng tin hơn nhưng nó
    là proxy, tức đúng cái giá trị vô dụng ta đang tránh.
    """
    forwarded = ""
    if headers is not None:
        getter = getattr(headers, "get", None)
        if callable(getter):
            forwarded = getter("x-forwarded-for") or ""
    first = forwarded.split(",")[0].strip()
    return hash_client(first or (fallback_host or "khong-ro"))


@dataclass(frozen=True, slots=True)
class Verdict:
    """Trả lời cho một lượt xin phép. `reason` đi thẳng ra mặt người dùng."""

    allowed: bool
    reason: str = ""


def _utc_day(now: float) -> int:
    return int(now // 86_400)


@dataclass
class SpendGuard:
    """Đếm lượt **sinh**. Truy hồi không tốn tiền nên không đi qua đây."""

    daily_total: int
    per_ip_daily: int
    per_ip_burst: int
    burst_window_s: float = 60.0
    #: Đơn giá quan sát được ở `W5-11` là $0,0010701/câu; lấy $0,002 làm trần
    #: trên. Chỉ dùng để **in ra** trần chi phí, không dùng để quyết định.
    usd_per_query: float = 0.002
    clock: Callable[[], float] = time.time

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _day: int = -1
    _total: int = 0
    _per_ip: dict[str, int] = field(default_factory=dict, repr=False)
    _burst: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _tripped: bool = False
    """Công tắc ngắt thủ công. Một chiều theo thiết kế: bật lại đòi khởi động
    lại Space, tức đòi một hành động của người, đúng như một cầu dao thật."""

    @property
    def max_usd_per_day(self) -> float:
        return round(self.daily_total * self.usd_per_query, 2)

    def _roll(self, now: float) -> None:
        """Sang ngày UTC mới thì mọi bộ đếm về 0. Gọi dưới `_lock`."""
        day = _utc_day(now)
        if day != self._day:
            self._day = day
            self._total = 0
            self._per_ip.clear()
            self._burst.clear()

    def trip(self) -> None:
        """Ngắt cứng. Dùng cho biến môi trường tắt sinh, hoặc khi vận hành cần."""
        with self._lock:
            self._tripped = True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._roll(self.clock())
            return {
                "tripped": self._tripped,
                "used_today": self._total,
                "daily_total": self.daily_total,
                "remaining": max(0, self.daily_total - self._total),
                "max_usd_per_day": self.max_usd_per_day,
            }

    def check(self, client: str) -> Verdict:
        """Có được phép gọi model không. **Không** thay đổi bộ đếm."""
        with self._lock:
            now = self.clock()
            self._roll(now)
            return self._check_locked(client, now)

    def _check_locked(self, client: str, now: float) -> Verdict:
        if self._tripped:
            return Verdict(False, "Sinh câu trả lời đang tắt trên bản demo này.")
        if self._total >= self.daily_total:
            return Verdict(
                False,
                f"Bản demo đã dùng hết hạn mức {self.daily_total} câu của hôm nay "
                f"(trần chi phí ~${self.max_usd_per_day}/ngày). Phần truy hồi vẫn "
                "chạy — bạn vẫn xem được nguồn cho câu hỏi của mình.",
            )
        if self._per_ip.get(client, 0) >= self.per_ip_daily:
            return Verdict(
                False,
                f"Bạn đã dùng {self.per_ip_daily} câu hôm nay — hạn mức cho mỗi "
                "khách. Phần truy hồi vẫn chạy.",
            )
        recent = [t for t in self._burst.get(client, ()) if now - t < self.burst_window_s]
        if len(recent) >= self.per_ip_burst:
            wait = int(self.burst_window_s - (now - recent[0])) + 1
            return Verdict(False, f"Hơi nhanh — thử lại sau {wait} giây.")
        return Verdict(True)

    def commit(self, client: str) -> Verdict:
        """Xin phép **và** ghi nhận trong một thao tác nguyên tử.

        ⚠️ Một `check()` rồi `commit()` ở hai lời gọi riêng là một cửa sổ đua:
        `W6-05` đo được 8 request đồng thời đi lọt qua đúng loại khe ấy ở tầng
        cache (`AU-11`). Đường dùng thật phải là hàm này.
        """
        with self._lock:
            now = self.clock()
            self._roll(now)
            verdict = self._check_locked(client, now)
            if not verdict.allowed:
                return verdict
            self._total += 1
            self._per_ip[client] = self._per_ip.get(client, 0) + 1
            recent = [t for t in self._burst.get(client, ()) if now - t < self.burst_window_s]
            recent.append(now)
            self._burst[client] = recent
            return verdict


def guard_from_env() -> SpendGuard:
    """Dựng từ biến môi trường — Space chỉnh được không cần sửa mã."""
    guard = SpendGuard(
        daily_total=int(os.environ.get("DEMO_DAILY_TOTAL", "500")),
        per_ip_daily=int(os.environ.get("DEMO_PER_IP_DAILY", "20")),
        per_ip_burst=int(os.environ.get("DEMO_PER_IP_BURST", "3")),
        burst_window_s=float(os.environ.get("DEMO_BURST_WINDOW_S", "60")),
    )
    if os.environ.get("DEMO_GENERATION", "on").strip().lower() in {"off", "0", "false"}:
        guard.trip()
    return guard
