"""Trần kích thước thân request, ở tầng ASGI — `NEW-12`.

## ⭐⭐ Tiền đề của nợ này **sai**, và phép đo là thứ chỉ ra điều đó

`W6-06` §8 ghi: *"lỗ duy nhất `W6-06` **không đóng được trong mã**: Pydantic chỉ
thấy thân request sau khi nó đã được đọc trọn vào bộ nhớ."* Vế sau đúng. Vế
trước sai — nó đúng với **Pydantic**, không đúng với **ASGI**. Middleware ASGI
thuần chạy *dưới* tầng ấy: nó thấy `content-length` trước khi một byte thân nào
được đọc, và nó thấy từng khung `http.request` khi chúng tới.

Đo trên cùng ngăn xếp (uvicorn + starlette + pydantic), `POST` 200 MB:

| | status | RSS đỉnh |
|---|---|---:|
| có `max_length=8000` của Pydantic | 422 | **852,3 MB** |
| không Pydantic, chunked | 200 | 421,4 MB |
| **+ trần ASGI**, chunked | **413** | **55,0 MB** |
| không Pydantic, khai `content-length` | 200 | 406,3 MB |
| **+ trần ASGI**, khai `content-length` | **413** | **54,8 MB** |

⚠️ Hàng đầu là hàng đáng đọc kỹ: trần **từng trường** không giảm nhẹ gì — nó
**khuếch đại**. 855 MB là **4,3× payload** và cao hơn cả đường không có Pydantic,
vì đường 422 còn dựng thêm bản giải mã JSON và thông điệp lỗi. Dòng
*"không trần (ngoài trần từng trường)"* của `security-final` đọc như một giảm
nhẹ một phần; trên trục kích thước request nó là **số âm**.

## ⭐⭐ Và nó là thứ thật sự đóng `SEC-04`

`W6-06` chặn `filters` ở **100 giá trị mỗi trường** (`MAX_FILTER_VALUES`) —
nhưng không chặn **độ dài từng giá trị**. Đo:

    {"message": …, "filters": {"chunk_id": [<1 MB> × 100]}}  →  100.048.510 byte

Thân ấy **hợp lệ với mọi phép kiểm hệ thống có hôm nay**. `W6-06` đóng đúng trục
nó nhìn vào và để hở trục ngay bên cạnh — đó là chế độ hỏng cố hữu của mọi hàng
rào đếm-theo-trường: **phải liệt kê đúng và đủ mọi trục**. Một trần tính bằng
**byte của cả thân** không cần liệt kê gì: nó chặn mọi trục cùng lúc, kể cả
những trục chưa ai nghĩ ra.

## ⭐ Một header thêm vào "cho chắc" đã làm 413 không đọc được

Bản đầu của `_refuse` gửi kèm `connection: close` — nghe như việc đúng phải làm
khi từ chối một thân request khổng lồ. Đo thì client nhận **`ReadError`**, không
nhận 413: máy chủ đóng kết nối trong lúc client còn đang gửi, nên phản hồi chưa
kịp được đọc. Bỏ header ấy đi thì **cả hai** ca trả về 413 thật, và RSS vẫn nằm
ở mức nền. Một hàng rào mà nạn nhân của nó không đọc được lý do là một hàng rào
buộc người ta phải đoán.

⚠️ Đây **không** thay được reverse proxy, và nói rõ để không ai đọc nhầm: byte
vẫn đi hết qua socket vào tiến trình ứng dụng trước khi bị đếm. Proxy hấp thụ
chúng sớm hơn một tầng, và chỉ proxy mới chặn được một client mở trăm kết nối
rồi nhỏ giọt từng byte. Cái này là hàng rào **tồn tại ngay hôm nay**, ở mọi chỗ
hệ chạy — kể cả Space HF, nơi không có proxy nào thuộc quyền ta.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["DEFAULT_MAX_BODY_BYTES", "BodyLimitMiddleware"]

logger = logging.getLogger(__name__)

#: Thân request **hợp lệ** lớn nhất đo được là **204.090 byte** — `/ingest` với
#: 1000 `doc_ids` × 200 ký tự, tức đúng trần mà chính `StartRequest` khai. `/chat`
#: đầy đủ tiếng Việt là 24.083 byte. Trần 1 MiB để dư **5,1×** so với cái lớn
#: nhất, chứ không phải một con số tròn chọn cho đẹp.
DEFAULT_MAX_BODY_BYTES = 1024 * 1024


def _declared_length(scope: dict[str, Any]) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                # Header hỏng: để tầng HTTP xử. Không đoán hộ.
                return None
    return None


class BodyLimitMiddleware:
    """Từ chối thân request vượt trần, **trước** khi nó nằm trọn trong bộ nhớ.

    ⭐⭐ Vị trí trong chồng là một quyết định, và **bản đầu đặt sai**.

    Chồng đúng: `RequestContext` → `Auth` → **`BodyLimit`** → router.

    * **Trong `AuthMiddleware`.** Bản đầu đặt nó ra ngoài auth, kèm lý do nghe
      rất xuôi: *"một thân 200 MB không khoá là đúng hình dạng tấn công, nên nó
      không được chờ xác thực."* Câu ấy **sai ở chỗ "chờ"**: `AuthMiddleware`
      quyết định hoàn toàn bằng header và **không bao giờ chạm `receive`**, nên
      với một request không khoá nó từ chối sau **0 byte**, trong khi trần này
      phải đếm tới `max_bytes` mới biết. Phép từ chối rẻ hơn phải ra ngoài hơn.
      Đặt ngược lại còn tiết lộ con số trần cho người chưa xác thực. Đường công
      khai (`/`, `/health`, `/ready`) vẫn được phủ vì auth cho chúng đi qua
      thẳng xuống đây.
    * **Trong `RequestContextMiddleware`** — để 413 vẫn mang `X-Request-ID` và
      vẫn được `rag_http_*` đếm. Cùng lý lẽ đã đặt context ra ngoài cùng ở
      `W4-03`: *phản hồi mà người vận hành cần truy vết nhất không được là phản
      hồi duy nhất không truy được*.

    ⚠️ **Không** thêm bộ đếm riêng. `RequestContextMiddleware` bọc ngoài nên nó
    đã đếm mọi 413 ở đây theo `status`; một chỉ số thứ hai cho cùng sự kiện là
    một chỗ để hai con số lệch nhau khi ai đó sửa một trong hai — cùng lý lẽ với
    `SingleFlight.stats()` không cộng sẵn hiệu dẫn xuất.
    """

    def __init__(self, app: Any, *, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            # Đường rẻ nhất: từ chối mà **không đọc một byte thân nào**.
            logger.warning(
                "từ chối thân request %d byte (trần %d) — khai qua content-length",
                declared,
                self.max_bytes,
            )
            await self._refuse(send)
            return

        # Không khai `content-length` (chunked) — hoặc khai rồi nhưng ta vẫn
        # đếm, vì một trần chỉ dựa vào lời khai của client là một trần dựa vào
        # lời khai của client.
        con_lai = self.max_bytes
        vuot_tran = False
        da_bat_dau = False

        async def receive_dem() -> dict[str, Any]:
            nonlocal con_lai, vuot_tran
            message = await receive()
            if message["type"] == "http.request":
                con_lai -= len(message.get("body", b""))
                if con_lai < 0:
                    vuot_tran = True
                    # ⚠️ `http.disconnect`, không phải ném exception: ta đang ở
                    # trong `receive` của ứng dụng, và một exception ở đây đi
                    # lên qua đúng những đường mà mỗi framework xử một kiểu.
                    # `http.disconnect` là từ vựng ASGI cho *"đừng chờ thêm"*,
                    # và mọi tầng ở trên đều đã biết cách dừng.
                    return {"type": "http.disconnect"}
            return message

        async def send_ghi_de(message: dict[str, Any]) -> None:
            nonlocal da_bat_dau
            if message["type"] == "http.response.start":
                da_bat_dau = True
                if vuot_tran:
                    # Ứng dụng đã dựng một phản hồi cho ca "client ngắt kết
                    # nối" (Starlette cho 400). Sự thật là 413, và client cần
                    # biết đúng lý do để sửa được request của mình.
                    await self._refuse(send)
                    return
            if vuot_tran and message["type"] == "http.response.body":
                # Đã vượt trần: khung thân của ứng dụng không được ra nữa. Nếu
                # phản hồi **chưa** bắt đầu thì `_refuse` ở trên đã gửi đủ cặp
                # start+body; nếu nó **đã** bắt đầu (SSE đang chảy) thì status
                # không sửa được nữa và cắt là lựa chọn trung thực duy nhất.
                return
            await send(message)

        try:
            await self.app(scope, receive_dem, send_ghi_de)
        except Exception:
            if not vuot_tran:
                raise
            # Ứng dụng ném vì thân request đứt giữa chừng (Starlette:
            # `ClientDisconnect`). Đó là hệ quả của chính ta, không phải một
            # lỗi máy chủ — nuốt nó và nói ra sự thật.
            logger.warning("từ chối thân request vượt trần %d byte", self.max_bytes)

        # ⚠️ **Ngoài** `try`, không nằm trong nhánh `except`. Bản đầu để nó ở
        # trong ấy, nên một ứng dụng **trả về mà không phản hồi gì** sau khi
        # vượt trần sẽ không nhận được 413 nào — request treo cho tới lúc client
        # bỏ cuộc. Chỗ này phải phủ **cả hai** đường thoát, không chỉ đường ném.
        if vuot_tran and not da_bat_dau:
            await self._refuse(send)

    async def _refuse(self, send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": (
                    b'{"detail":"than request vuot tran '
                    + str(self.max_bytes).encode()
                    + b' byte"}'
                ),
            }
        )
