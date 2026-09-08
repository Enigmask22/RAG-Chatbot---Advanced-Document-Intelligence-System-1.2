"""`NEW-12` — trần thân request, và ba cách một trần có thể **làm hỏng hệ**.

Một hàng rào chặn theo kích thước chỉ đáng tin khi nó chứng minh được ba điều
**âm**, không phải điều dương "thân to thì bị chặn":

1. **Không chặn nhầm cái hợp lệ.** Thân request hợp lệ lớn nhất của hệ này là
   204.090 byte (`/ingest`). Một trần đặt sai làm `/ingest` chết mà không ai
   nối được nguyên nhân với hàng rào — nó biểu hiện thành "ingest thỉnh thoảng
   413".
2. **Không phá SSE.** Middleware này nằm trên đường của `POST /chat`, và
   `W4-06` là stream. `middleware.py` đã ghi rõ vì sao `BaseHTTPMiddleware` là
   sai ở đây; một bọc `send` viết cẩu thả tái lập đúng lỗi ấy.
3. **Không nuốt lỗi thật.** Nó bắt `Exception` để nuốt `ClientDisconnect` do
   chính nó gây ra. Nếu nó nuốt luôn lỗi của ứng dụng thì mọi 500 biến thành
   413 — và bảng sẽ báo "người dùng gửi thân request quá to" cho một sự cố
   máy chủ.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from serving.api.body_limit import DEFAULT_MAX_BODY_BYTES, BodyLimitMiddleware

pytestmark = pytest.mark.asyncio


def _scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict[str, Any]:
    return {"type": "http", "method": "POST", "path": "/chat", "headers": headers or []}


class _Bat:
    """Ghi lại mọi khung `send`, để khẳng định trên **khung**, không trên chuỗi."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.frames.append(message)

    @property
    def status(self) -> int | None:
        for f in self.frames:
            if f["type"] == "http.response.start":
                return int(f["status"])
        return None

    @property
    def body(self) -> bytes:
        return b"".join(
            f.get("body", b"") for f in self.frames if f["type"] == "http.response.body"
        )


def _khung_than(*chunks: bytes) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Sinh chuỗi khung ASGI cho một thân gửi làm nhiều mẩu."""
    seq = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]

    async def receive() -> dict[str, Any]:
        return seq.pop(0) if seq else {"type": "http.disconnect"}

    return receive


class _UngDung:
    """Ứng dụng đọc hết thân rồi trả 200 — đủ để phân biệt đọc-được với bị-chặn."""

    def __init__(self) -> None:
        self.doc_duoc = 0
        self.duoc_goi = False

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.duoc_goi = True
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            self.doc_duoc += len(message.get("body", b""))
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class TestKhongChanNhamCaiHopLe:
    async def test_tran_mac_dinh_du_cho_than_hop_le_lon_nhat(self) -> None:
        """⭐ 204.090 byte là `/ingest` với 1000 `doc_ids` × 200 ký tự — đúng
        trần mà chính `StartRequest` khai, nên nó là thân **hợp lệ** lớn nhất
        hệ có thể nhận. Con số này đo được, không ước lượng."""
        assert DEFAULT_MAX_BODY_BYTES > 204_090

    async def test_than_duoi_tran_di_qua_nguyen_ven(self) -> None:
        app = _UngDung()
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1000)
        await mw(_scope(), _khung_than(b"a" * 400, b"b" * 400), bat)
        assert bat.status == 200
        assert app.doc_duoc == 800, "ứng dụng phải nhận đủ thân, không thiếu byte nào"

    async def test_than_dung_bang_tran_van_di_qua(self) -> None:
        """Ranh giới là `>`, không phải `>=`: một trần khai 1 MiB mà từ chối
        đúng 1 MiB là một trần nói dối về con số của chính nó."""
        app = _UngDung()
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1000)
        await mw(_scope(), _khung_than(b"x" * 1000), bat)
        assert bat.status == 200

    async def test_khai_content_length_dung_bang_tran_van_di_qua(self) -> None:
        app = _UngDung()
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1000)
        await mw(_scope([(b"content-length", b"1000")]), _khung_than(b"x" * 1000), bat)
        assert bat.status == 200

    async def test_khong_phai_http_thi_KHONG_BOC_receive_va_send(self) -> None:
        """⭐⭐ Bản đầu của bài này khẳng định `app.duoc_goi` — và nó **không thể
        đỏ**: tiêm bỏ hẳn nhánh `scope["type"] != "http"` thì `lifespan` vẫn
        chạy trót lọt, vì `receive_dem` chỉ đếm khung `http.request` và
        `send_ghi_de` chỉ động vào `http.response.start`. Kèm theo đó là một lời
        biện minh tôi viết ra mà không đo (*"bọc chúng là cách làm chết
        `startup`"*) — đúng lỗi `NEW-10` §4.

        ⚠️ Nên nói thẳng: phép kiểm ấy hôm nay **không quan sát được bằng hành
        vi**. Nó được giữ vì **hợp đồng ASGI** (mọi middleware phải để scope
        không-http đi thẳng), không vì một kịch bản hỏng có thật — và nó được
        ghim bằng phép so **danh tính**, thứ duy nhất quan sát được, thay vì
        bằng một câu chuyện. Ngày có route WebSocket đầu tiên, phép đếm byte
        của HTTP không được áp lên nó.
        """
        nhan: dict[str, object] = {}

        async def app(scope: Any, receive: Any, send: Any) -> None:
            nhan["receive"] = receive
            nhan["send"] = send

        recv = _khung_than(b"")
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1)
        await mw({"type": "lifespan"}, recv, bat)
        assert nhan["receive"] is recv, "scope không-http phải nhận ĐÚNG `receive` gốc"
        assert nhan["send"] is bat, "scope không-http phải nhận ĐÚNG `send` gốc"


class TestChanCaiQuaTran:
    async def test_khai_content_length_qua_tran_bi_chan_TRUOC_khi_doc(self) -> None:
        """⭐⭐ Mệnh đề trung tâm của cả hạng mục: **ứng dụng không được gọi**.

        Nếu nó được gọi thì thân request đã đi vào bộ nhớ, và toàn bộ lý do tồn
        tại của middleware này biến mất — kể cả khi status trả về vẫn là 413.
        """
        app = _UngDung()
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1000)
        await mw(_scope([(b"content-length", b"999999")]), _khung_than(b"x" * 1000), bat)
        assert bat.status == 413
        assert not app.duoc_goi, "chặn phải xảy ra TRƯỚC khi ứng dụng chạm thân request"

    async def test_khong_khai_content_length_van_bi_chan_theo_so_byte_that(self) -> None:
        """Chunked: không có lời khai nào để tin. Đây là đường mà một trần chỉ
        đọc `content-length` sẽ bỏ lọt hoàn toàn."""
        app = _UngDung()
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1000)
        await mw(_scope(), _khung_than(*[b"x" * 400] * 10), bat)
        assert bat.status == 413
        # ⭐ Đếm **khung**, không chỉ đọc status. Ứng dụng vẫn gửi cặp
        # start+body 200 của nó sau khi bị ngắt; nếu chúng lọt ra thì đây là
        # hai phản hồi chồng lên nhau trên một kết nối — lỗi giao thức mà
        # `bat.status` (chỉ đọc khung `start` **đầu tiên**) không thấy.
        assert [f["type"] for f in bat.frames] == [
            "http.response.start",
            "http.response.body",
        ], bat.frames

    async def test_khong_doc_them_sau_khi_vuot_tran(self) -> None:
        """⭐ Trần là trần **bộ nhớ**, nên nó phải dừng ngay ở mẩu làm tràn —
        không phải ở cuối thân request. Một bản vá đếm đủ rồi mới từ chối vẫn
        cho phép 200 MB đi vào RAM và chỉ đổi được status."""
        app = _UngDung()
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1000)
        await mw(_scope(), _khung_than(*[b"x" * 400] * 100), bat)
        assert bat.status == 413
        assert app.doc_duoc <= 1000 + 400, f"đã đọc {app.doc_duoc} byte sau khi vượt trần"

    async def test_than_qua_tran_bao_ra_con_so_tran(self) -> None:
        """Người bị chặn phải sửa được request của mình. Một 413 trống buộc họ
        đoán, và họ sẽ đoán bằng cách thử lại — cùng lý lẽ với thông điệp của
        `_drain_saves` ở `NEW-14`."""
        bat = _Bat()
        mw = BodyLimitMiddleware(_UngDung(), max_bytes=4242)
        await mw(_scope([(b"content-length", b"999999")]), _khung_than(b""), bat)
        assert b"4242" in bat.body
        assert json.loads(bat.body)["detail"]

    async def test_content_length_hong_khong_lam_no_middleware(self) -> None:
        """Header rác không được thành 500. Bỏ qua lời khai, đếm byte thật."""
        app = _UngDung()
        bat = _Bat()
        mw = BodyLimitMiddleware(app, max_bytes=1000)
        await mw(_scope([(b"content-length", b"khong-phai-so")]), _khung_than(b"x" * 10), bat)
        assert bat.status == 200


class TestKhongPhaSSEVaKhongNuotLoiThat:
    async def test_khung_phan_hoi_di_qua_TUNG_CAI_MOT(self) -> None:
        """⚠️ `W4-06` là SSE. `middleware.py` ghi rõ vì sao
        `BaseHTTPMiddleware` sai ở đây — nó gom phản hồi lại. Bọc `send` mà
        buffer thì tái lập đúng lỗi ấy, và triệu chứng ("stream về một cục ở
        cuối") trông như lỗi của tầng sinh, cách xa nguyên nhân."""
        khung_gui: list[dict[str, Any]] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            for i in range(3):
                await send(
                    {"type": "http.response.body", "body": f"m{i}".encode(), "more_body": True}
                )
                # Nếu middleware gom lại thì tới đây `khung_gui` còn rỗng.
                assert len(khung_gui) == i + 2, "khung phải ra ngoài NGAY, không đợi khung sau"
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def send(message: dict[str, Any]) -> None:
            khung_gui.append(message)

        mw = BodyLimitMiddleware(app, max_bytes=1_000_000)
        await mw(_scope(), _khung_than(b"q"), send)
        assert len(khung_gui) == 5

    async def test_loi_that_cua_ung_dung_van_noi_len(self) -> None:
        """⭐⭐ `except Exception` ở đây tồn tại để nuốt `ClientDisconnect` **do
        chính middleware gây ra**. Nếu nó nuốt cả lỗi của ứng dụng thì mọi 500
        thành 413, và bảng RAG Health sẽ báo "người dùng gửi thân quá to" cho
        một sự cố máy chủ — đúng họ với `refusals_suspected` của `W5-07`."""

        async def app_no(scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError("Qdrant chết")

        mw = BodyLimitMiddleware(app_no, max_bytes=1_000_000)
        with pytest.raises(RuntimeError, match="Qdrant chết"):
            await mw(_scope(), _khung_than(b"q"), _Bat())

    async def test_loi_sau_khi_vuot_tran_thi_thanh_413_chu_khong_no(self) -> None:
        """Mặt còn lại của bài trên: `ClientDisconnect` do ta gây ra **phải**
        bị nuốt, không thì hàng rào tự biến mình thành 500."""

        async def app_no(scope: Any, receive: Any, send: Any) -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    raise RuntimeError("client disconnect")

        bat = _Bat()
        mw = BodyLimitMiddleware(app_no, max_bytes=100)
        await mw(_scope(), _khung_than(*[b"x" * 60] * 5), bat)
        assert bat.status == 413

    async def test_ung_dung_tra_ve_ma_KHONG_phan_hoi_gi_van_nhan_413(self) -> None:
        """⭐⭐ Đường thoát thứ ba, và bản đầu để hở nó.

        Ứng dụng thấy `http.disconnect` rồi **return** — không ném, không gửi
        khung nào. Bản đầu đặt phép cứu trong nhánh `except`, nên đường này
        không có phản hồi nào cả và request treo tới lúc client bỏ cuộc. Cùng
        hình dạng với `NEW-10`: một `finally` đặt sau chỗ nó cần phủ.
        """

        async def app_im_lang(scope: Any, receive: Any, send: Any) -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        bat = _Bat()
        mw = BodyLimitMiddleware(app_im_lang, max_bytes=100)
        await mw(_scope(), _khung_than(*[b"x" * 60] * 5), bat)
        assert bat.status == 413

    async def test_ung_dung_da_bat_dau_phan_hoi_thi_khong_ghi_de_status(self) -> None:
        """⚠️ Ca không sửa được: nếu stream đã bắt đầu thì status đã đi rồi.
        Ghim để không ai "sửa" bằng cách gửi thêm một `response.start` thứ hai —
        đó là lỗi giao thức, và nó sẽ đóng kết nối giữa một câu trả lời."""
        khung: list[dict[str, Any]] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"da-chay", "more_body": True})
            while True:
                if (await receive())["type"] == "http.disconnect":
                    break
            # Sau khi bị ngắt, ứng dụng vẫn cố nhả nốt — khung này KHÔNG được ra.
            await send({"type": "http.response.body", "body": b"sau-khi-ngat"})

        async def send(message: dict[str, Any]) -> None:
            khung.append(message)

        mw = BodyLimitMiddleware(app, max_bytes=100)
        await mw(_scope(), _khung_than(*[b"x" * 60] * 5), send)
        starts = [f for f in khung if f["type"] == "http.response.start"]
        assert len(starts) == 1 and starts[0]["status"] == 200
        than = [f.get("body") for f in khung if f["type"] == "http.response.body"]
        assert than == [b"da-chay"], f"khung thân sau khi vượt trần vẫn lọt ra: {than}"
