"""Gộp N request **trùng nhau đang bay** thành một lượt sinh. `NEW-10`.

`W6-05` đo `AU-11` bằng một probe: gửi **8** lần cùng một câu hỏi đồng thời vào
`/chat` với cache bật ⇒ **8 lời gọi nhà cung cấp**, đỉnh 6 lượt chồng nhau,
**0 cache hit**. Cùng câu ấy gửi nối đuôi ⇒ 0 lời gọi, 50 ms. Semantic cache
của `W4-10` không hỏng; nó chỉ **không thể** giúp, vì cả tám lượt tra cache
trước khi lượt nào kịp ghi.

Cái giá không phải mỹ quan: 8× tiền, và 8× rerank trên đúng tài nguyên là trần
của hệ thống (`TD-63`). Một trang demo công khai có nút "gửi" bấm được hai lần
là đủ để dựng lại nguyên hình dạng ấy.

## Năm quyết định

1. **⭐⭐ Khoá phải CHÍNH XÁC, không được mờ.** Semantic cache khớp theo cosine
   ≥ 0,96 cộng hàng rào token chữ số — và nó *được phép* mờ vì mỗi lượt hit
   khai ra `matched_question` và `similarity` cho người đọc thấy. Single-flight
   **không có** bước ấy: người theo sau nhận thẳng câu trả lời như thể của
   mình. Nên khoá ở đây là `(tenant, namespace, câu hỏi NGUYÊN VĂN)`. Hai
   paraphrase đồng thời **không** được gộp — chấp nhận, vì `AU-11` đo trên tám
   câu **giống hệt** và đó là hình dạng thật của vấn đề (reload trang, bấm hai
   lần, một link được chia sẻ).

2. **⭐⭐ Người theo sau nhận câu trả lời TỪ BỘ NHỚ, không qua Redis.** Đường
   ghi cache là một task nền (`_PENDING` trong `chat.py`), nên một follower
   thức dậy rồi `lookup()` lại sẽ **đua với chính đường ghi ấy** và thường
   thua. Leader `resolve()` thẳng vào future. Redis vẫn được ghi như cũ, cho
   những lượt đến *sau*.

3. **⭐⭐ Leader hỏng thì follower KHÔNG được cùng chết.** Đây là chế độ hỏng
   nguy hiểm nhất của mọi cơ chế gộp: biến một lỗi thành N lỗi. `resolve(None)`
   nghĩa là *"không có câu trả lời, tự đi mà làm"* — và nó chạy trong `finally`
   của `stream_turn`, tức trên **cả ba** đường thoát (xong, lỗi, huỷ). Tệ nhất
   là N request đầy đủ: **đúng bằng hôm nay**, không tệ hơn.

4. **Có hạn giờ, và nó phải lớn hơn p99.** `exp-003` đo p99 end-to-end
   **11.142 ms**; mặc định 15 s. Quá hạn thì follower tự làm — nhưng lúc ấy nó
   đã tiêu 15 s **cộng** thời gian tự làm, tệ hơn là không chờ. Nên hạn giờ
   không phải một van an toàn rẻ tiền: đặt quá cao thì một leader treo kéo cả
   nhóm xuống. Nó là hàng rào cho ca leader **biến mất mà không chạy `finally`**
   (tiến trình bị giết) — ca duy nhất mà quyết định 3 không phủ.

5. **⚠️ Trong tiến trình, và nói ra.** Cùng lời khai với `CostBudget` của
   `W4-08` và hạn mức nhịp của `W4-04` (`TD-39`): 4 replica ⇒ **4** lượt sinh,
   không phải 1. Không dựng khoá phân tán trên Redis cho một triển khai chưa
   tồn tại — và `TD-63` vừa chứng minh (`NEW-09`, 08/09/2026) rằng lối ra
   multi-container còn chưa tới. Một `SETNX` + TTL mang theo cả họ chế độ hỏng
   của khoá phân tán (leader chết ⇒ follower chờ hết TTL, lệch đồng hồ, khoá bị
   cướp) để đổi lấy một lợi ích chưa đo được ở đây.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serving.core.semantic_cache import CachedAnswer

__all__ = ["Flight", "SingleFlight", "flight_key"]

logger = logging.getLogger(__name__)

#: Lớn hơn p99 end-to-end đo được ở `exp-003` (11.142 ms) một biên. Xem quyết
#: định 4: đây là hàng rào cho leader **biến mất**, không phải cho leader chậm.
DEFAULT_WAIT_S = 15.0


def flight_key(tenant: str, namespace: str, question: str) -> str:
    """Khoá gộp — **nguyên văn**, không chuẩn hoá mờ. Xem quyết định 1.

    ⚠️ Chỉ cắt khoảng trắng hai đầu. Không hạ chữ hoa, không bỏ dấu, không gập
    khoảng trắng bên trong: mỗi phép chuẩn hoá là một cách để hai câu hỏi khác
    nhau va vào cùng một khoá, và ở đây va nhau nghĩa là **trả lời sai người**.
    `namespace` đã mang bundle/prompt/top_k/generator/endpoint (`cache_namespace`),
    `tenant` là hàng rào dữ liệu — cùng luật với `AU-02`: một khoá phải chứa
    mọi đầu vào làm đổi câu trả lời.
    """
    return f"{tenant}\x00{namespace}\x00{question.strip()}"


@dataclass
class Flight:
    """Một vé. `is_leader` quyết định bên gọi phải làm gì với nó."""

    key: str
    is_leader: bool
    _future: asyncio.Future[CachedAnswer | None]
    _owner: SingleFlight
    _wait_s: float

    def resolve(self, answer: CachedAnswer | None) -> None:
        """Leader gọi khi đã có câu trả lời — hoặc khi biết mình không có.

        **Đồng bộ và idempotent** vì nó chạy trong `finally` của `stream_turn`,
        nơi có thể đang bị huỷ: docstring ở đó ghi *"đồng bộ, không `await`"*,
        và một `await` trong lúc bị huỷ không chạy tới nơi.
        """
        if not self.is_leader or self._future.done():
            return
        self._owner._retire(self.key)
        self._future.set_result(answer)

    async def wait(self) -> CachedAnswer | None:
        """Follower chờ leader. `None` = tự đi mà làm.

        ⚠️ Nuốt **mọi** kết cục xấu thành `None` chứ không ném: một cơ chế gộp
        được thêm vào để giảm hoá đơn không được phép trở thành một cách mới
        làm `/chat` trả lỗi. Cùng luật với `SemanticCache` (*"hỏng thì hỏng về
        phía miss"*).
        """
        if self.is_leader:
            return None
        try:
            answer = await asyncio.wait_for(asyncio.shield(self._future), self._wait_s)
        except TimeoutError:
            self._owner.timeouts += 1
            logger.info("single-flight quá hạn %.1fs, tự đi đường đầy đủ", self._wait_s)
            return None
        except asyncio.CancelledError:
            # Client của **follower** ngắt kết nối. Không đụng tới leader:
            # `shield` giữ future sống cho những người theo sau khác.
            raise
        except Exception:
            logger.exception("single-flight hỏng khi chờ — tự đi đường đầy đủ")
            return None
        if answer is not None:
            self._owner.served += 1
        return answer


class SingleFlight:
    """Sổ các lượt đang bay, theo khoá. Xem docstring module."""

    def __init__(self, *, wait_s: float = DEFAULT_WAIT_S) -> None:
        self.wait_s = wait_s
        self._inflight: dict[str, asyncio.Future[CachedAnswer | None]] = {}
        self.led = 0
        self.followed = 0
        self.served = 0
        self.timeouts = 0

    def join(self, key: str) -> Flight:
        """Nhận vé cho `key`. **Đồng bộ** — không có `await` giữa tra và ghi.

        ⭐ Đó là điều kiện đúng đắn, không phải tối ưu: một `await` ở giữa là
        đúng cửa sổ đua mà `guard.py` của `W6-02` đã phải đóng bằng
        `check()`+`commit()` nguyên tử, và là cùng hình dạng với chính `AU-11`.
        """
        existing = self._inflight.get(key)
        if existing is not None and not existing.done():
            self.followed += 1
            return Flight(key, False, existing, self, self.wait_s)
        future: asyncio.Future[CachedAnswer | None] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        self.led += 1
        return Flight(key, True, future, self, self.wait_s)

    def _retire(self, key: str) -> None:
        """Gỡ khỏi sổ.

        ⭐⭐ Bản đầu so `is` với future của chính mình, kèm một kịch bản nghe rất
        hợp lý: *"leader A hỏng, B nhận cùng khoá, rồi `finally` muộn của A xoá
        vé của B"*. **Tiêm lỗi bác bỏ kịch bản ấy** — thay bằng `pop()` trần thì
        không bài nào đỏ, và truy ra thì nó **không thể xảy ra**:

        Lý do thật là **tính nguyên tử**, không phải thứ tự: `resolve()` đồng
        bộ từ đầu tới cuối, nên giữa `_retire` và `set_result` **không coroutine
        nào chen vào được**, và `join()` không bao giờ quan sát được trạng thái
        trung gian. Suy ra một khoá không bao giờ có hai leader sống cùng lúc,
        và A không thể "về muộn" sau B.

        ⚠️ Thứ tự hai dòng ấy cũng đã được tiêm thử (`M8`, đảo lại) và **sống
        sót** — nó là một *mutant tương đương*, không phải một lỗ test. Giữ thứ
        tự hiện tại vì nó đọc ra bất biến rõ hơn, nhưng đừng viết một bài test
        canh nó: bài test ấy sẽ canh một thứ không quan sát được.

        Đó là hai dòng mã chết cộng một câu chuyện bịa để biện minh cho chúng.
        Cùng bài học `W5-11`/`M2`: một điều kiện không đổi được hành vi là một
        chú thích viết bằng cú pháp `if` — và nguy hiểm hơn chú thích, vì nó
        trông như đã có ai đó nghĩ tới ca ấy.
        """
        self._inflight.pop(key, None)

    @property
    def inflight(self) -> int:
        return len(self._inflight)

    def stats(self) -> dict[str, int]:
        """Bốn con số kể đủ chuyện: dẫn / theo / được phục vụ / quá hạn.

        `followed - served - timeouts` = số người theo sau nhận `None` vì
        leader không có câu trả lời (lỗi, huỷ, hoặc câu rỗng). Không cộng sẵn
        hiệu ấy: một chỉ số dẫn xuất nằm cạnh chỉ số gốc là chỗ để hai con số
        lệch nhau khi ai đó sửa một trong hai.
        """
        return {
            "led": self.led,
            "followed": self.followed,
            "served": self.served,
            "timeouts": self.timeouts,
            "inflight": self.inflight,
        }
