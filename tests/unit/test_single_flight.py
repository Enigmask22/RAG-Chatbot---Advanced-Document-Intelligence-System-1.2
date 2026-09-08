"""`NEW-10` — gộp request trùng nhau, và ba cách nó có thể **làm tệ hơn**.

Một cơ chế gộp được thêm vào để giảm hoá đơn chỉ đáng tin khi nó chứng minh
được ba điều **âm**, không phải điều dương "hai request thành một":

1. Leader hỏng ⇒ follower **không** cùng chết. Biến một lỗi thành N lỗi là chế
   độ hỏng đắt nhất mà một cơ chế gộp mang lại.
2. Follower ngắt kết nối ⇒ leader **không** bị kéo theo, và những follower khác
   vẫn nhận được câu trả lời.
3. Hai câu hỏi **khác nhau** ⇒ không bao giờ dùng chung một khoá. Ở đây va khoá
   nghĩa là trả lời sai người, và không có bước xác minh nào ở sau để bắt.

Bộ test này viết theo ba điều ấy trước, rồi mới tới đường vui.
"""

from __future__ import annotations

import asyncio

import pytest

from serving.core.semantic_cache import CachedAnswer
from serving.core.single_flight import DEFAULT_WAIT_S, SingleFlight, flight_key

pytestmark = pytest.mark.asyncio


def _answer(text: str = "xong") -> CachedAnswer:
    return CachedAnswer(
        question="RRF là gì?",
        text=text,
        sources=[],
        citations_frame=None,
        model="fake",
        similarity=1.0,
    )


class TestKhoaPhaiChinhXacChuKhongMo:
    """Quyết định 1 của module. Semantic cache **được phép** mờ vì mỗi hit khai
    ra `matched_question`; single-flight thì không có bước ấy."""

    def test_hai_cau_khac_nhau_khong_dung_chung_khoa(self) -> None:
        a = flight_key("t", "ns", "Tỉ lệ nghèo năm 1993?")
        b = flight_key("t", "ns", "Tỉ lệ nghèo năm 1998?")
        assert a != b

    def test_khac_hoa_thuong_la_khac_khoa(self) -> None:
        """Không hạ chữ hoa: mỗi phép chuẩn hoá là một cách để hai câu va nhau."""
        assert flight_key("t", "ns", "RRF") != flight_key("t", "ns", "rrf")

    def test_khac_tenant_la_khac_khoa(self) -> None:
        """Cùng luật với namespace của `SemanticCache`: trả câu của tenant này
        cho tenant kia là **rò dữ liệu**, không phải một lượt gộp."""
        assert flight_key("a", "ns", "q") != flight_key("b", "ns", "q")

    def test_khac_namespace_la_khac_khoa(self) -> None:
        assert flight_key("t", "ns1", "q") != flight_key("t", "ns2", "q")

    def test_chi_cat_khoang_trang_hai_dau(self) -> None:
        assert flight_key("t", "ns", "  q  ") == flight_key("t", "ns", "q")
        assert flight_key("t", "ns", "a b") != flight_key("t", "ns", "a  b")

    def test_ranh_gioi_truong_khong_gia_mao_duoc_bang_noi_dung(self) -> None:
        """⭐⭐ Bản đầu của bài này **không thể đỏ**: nó so hai khoá vốn khác
        nhau dưới *mọi* dấu ngăn cách, nên phép tiêm đổi `\\x00` thành `:` sống
        sót. Luật thật là **không có nhập nhằng**, và cách duy nhất kiểm được nó
        là dựng đúng một cặp **va nhau** dưới dấu ngăn cách tồi.

        Với `:` thì `("a", "b:c", "q")` và `("a", "b", "c:q")` cho **cùng một
        chuỗi** — hai namespace khác nhau dùng chung một lượt sinh. Với `\\x00`
        thì không, vì một câu hỏi đi qua JSON/HTTP không mang byte NUL.
        """
        assert flight_key("a", "b:c", "q") != flight_key("a", "b", "c:q")


class TestLeaderHongThiFollowerKhongCungChet:
    async def test_leader_resolve_None_thi_follower_di_duong_day_du(self) -> None:
        sf = SingleFlight()
        leader = sf.join("k")
        follower = sf.join("k")
        leader.resolve(None)
        assert await follower.wait() is None
        assert sf.served == 0

    async def test_leader_khong_bao_gio_resolve_thi_follower_qua_han_chu_khong_treo(
        self,
    ) -> None:
        sf = SingleFlight(wait_s=0.05)
        sf.join("k")  # leader biến mất, không giữ tham chiếu
        follower = sf.join("k")
        assert await follower.wait() is None
        assert sf.timeouts == 1

    async def test_qua_han_KHONG_dem_la_duoc_phuc_vu(self) -> None:
        """Nhóm chứng cho bài trên: `served` là số người **thật sự** đỡ được một
        lượt sinh. Gộp nó với `timeouts` là biến sổ thành lời quảng cáo."""
        sf = SingleFlight(wait_s=0.05)
        sf.join("k")
        await sf.join("k").wait()
        assert sf.stats() == {"led": 1, "followed": 1, "served": 0, "timeouts": 1, "inflight": 1}


class TestFollowerNgatKetNoiKhongKeoLeaderTheo:
    async def test_mot_follower_bi_huy_khong_lam_hong_follower_khac(self) -> None:
        """⚠️ Đây là lý do `wait()` dùng `asyncio.shield`. Không có nó, một
        client bấm Esc sẽ huỷ **future dùng chung** và mọi người còn lại nhận
        `CancelledError` từ một request không phải của mình."""
        sf = SingleFlight()
        leader = sf.join("k")
        bo_cuoc = asyncio.create_task(sf.join("k").wait())
        kien_nhan = asyncio.create_task(sf.join("k").wait())
        await asyncio.sleep(0)
        bo_cuoc.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bo_cuoc

        leader.resolve(_answer("của leader"))
        got = await kien_nhan
        assert got is not None and got.text == "của leader"


class TestDuongVui:
    async def test_follower_nhan_dung_cau_tra_loi_cua_leader(self) -> None:
        sf = SingleFlight()
        leader = sf.join("k")
        follower = sf.join("k")
        assert leader.is_leader and not follower.is_leader
        leader.resolve(_answer("từ bộ nhớ, không qua Redis"))
        got = await follower.wait()
        assert got is not None and got.text == "từ bộ nhớ, không qua Redis"
        assert sf.stats()["served"] == 1

    async def test_nhieu_follower_cung_nhan_mot_cau_tra_loi(self) -> None:
        sf = SingleFlight()
        leader = sf.join("k")
        followers = [asyncio.create_task(sf.join("k").wait()) for _ in range(7)]
        await asyncio.sleep(0)
        leader.resolve(_answer("một lượt sinh"))
        got = await asyncio.gather(*followers)
        assert [g.text for g in got if g] == ["một lượt sinh"] * 7
        assert sf.stats() == {"led": 1, "followed": 7, "served": 7, "timeouts": 0, "inflight": 0}

    async def test_khoa_khac_nhau_thi_ai_cung_la_leader(self) -> None:
        sf = SingleFlight()
        assert sf.join("a").is_leader
        assert sf.join("b").is_leader
        assert sf.led == 2

    async def test_sau_khi_leader_xong_lan_toi_lai_la_leader(self) -> None:
        """Sổ phải **rỗng lại** — không thì lượt thứ hai của cùng câu hỏi sẽ
        theo một future đã xong và nhận lại câu trả lời cũ mãi mãi."""
        sf = SingleFlight()
        sf.join("k").resolve(_answer())
        assert sf.inflight == 0
        assert sf.join("k").is_leader


class TestResolveLaHopDongChuKhongPhaiGoiY:
    async def test_resolve_hai_lan_khong_no_va_khong_doi_ket_qua(self) -> None:
        """`finally` của `stream_turn` có thể chạy sau một `resolve` sớm; và một
        `set_result` lần hai là `InvalidStateError` — tức một cơ chế tiết kiệm
        tiền giết chính request nó vừa phục vụ."""
        sf = SingleFlight()
        leader = sf.join("k")
        follower = sf.join("k")
        leader.resolve(_answer("đầu"))
        leader.resolve(_answer("sau"))
        got = await follower.wait()
        assert got is not None and got.text == "đầu"

    async def test_follower_goi_resolve_la_khong_lam_gi(self) -> None:
        """Vé của follower không được phép trả lời thay leader."""
        sf = SingleFlight()
        leader = sf.join("k")
        follower = sf.join("k")
        follower.resolve(_answer("giả mạo"))
        leader.resolve(_answer("thật"))
        got = await follower.wait()
        assert got is not None and got.text == "thật"

    async def test_leader_goi_wait_thi_nhan_None_chu_khong_tu_khoa_minh(self) -> None:
        sf = SingleFlight()
        assert await sf.join("k").wait() is None


class TestMotKhoaKhongBaoGIOCoHAILeaderSong:
    """⭐⭐ Bất biến này thay cho một phép so `is` mà **tiêm lỗi đã bác bỏ**.

    Bản đầu của `_retire` so future với chính mình, kèm kịch bản *"A về muộn
    xoá vé của B"*. Thay bằng `pop()` trần thì không bài nào đỏ — vì kịch bản
    ấy không thể xảy ra. Nên bộ test này ghim **lý do** nó không thể, chứ không
    ghim một hàng rào thừa.
    """

    async def test_future_dang_giu_khoa_thi_chua_bao_gio_done(self) -> None:
        """`resolve()` gỡ sổ **trước** `set_result` — đó là toàn bộ cơ chế."""
        sf = SingleFlight()
        leader = sf.join("k")
        assert sf.inflight == 1
        leader.resolve(_answer())
        assert sf.inflight == 0, "gỡ sổ phải xong TRƯỚC khi future thành done"

    async def test_trong_khi_leader_chua_xong_moi_nguoi_deu_la_follower(self) -> None:
        sf = SingleFlight()
        sf.join("k")
        assert [sf.join("k").is_leader for _ in range(5)] == [False] * 5

    async def test_ve_ve_muon_cua_leader_cu_khong_dung_toi_leader_moi(self) -> None:
        """Hệ quả: A đã xong ⇒ `resolve()` lần hai thoát ngay ở `future.done()`,
        không chạm sổ. B an toàn **nhờ bất biến**, không nhờ một phép so."""
        sf = SingleFlight()
        a = sf.join("k")
        a.resolve(None)
        b = sf.join("k")
        theo_b = sf.join("k")
        assert not theo_b.is_leader

        a.resolve(_answer("của A, về muộn"))
        b.resolve(_answer("của B"))
        got = await theo_b.wait()
        assert got is not None and got.text == "của B"


class TestHanGioMacDinh:
    def test_lon_hon_p99_do_duoc(self) -> None:
        """`exp-003`: p99 end-to-end 11.142 ms. Hạn giờ nhỏ hơn con số ấy biến
        cơ chế gộp thành một máy sinh độ trễ: follower chờ rồi vẫn tự làm."""
        assert DEFAULT_WAIT_S > 11.142
