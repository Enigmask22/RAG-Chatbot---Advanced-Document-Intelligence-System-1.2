"""`W6-02` — trần chi tiêu của demo công khai.

Đây là **hàng rào duy nhất** giữa một người lạ và hoá đơn DeepSeek: Space không
có xác thực, không có `127.0.0.1`, không có middleware nào của `W4-04`. Nên bộ
test này soi hai thứ mà một bộ test "gọi hàm xem có chạy không" bỏ qua:

* `check()` **không được** đếm, `commit()` **phải** đếm. Lẫn hai cái là mở cửa
  cho một vòng lặp miễn phí.
* Từ chối vì trần tổng, vì trần khách, và vì bấm nhanh là **ba** thông điệp
  khác nhau — gộp chúng lại thì người vận hành đọc log không biết cái nào chạm.
"""

from __future__ import annotations

import pytest

from guard import SpendGuard, client_key_of, guard_from_env, hash_client

DAY = 86_400.0


class _Clock:
    """Đồng hồ giả. Ngày UTC tính bằng `now // 86400` nên nhảy ngày là cộng DAY."""

    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _guard(clock: _Clock, **kw: object) -> SpendGuard:
    opts: dict[str, object] = {
        "daily_total": 5,
        "per_ip_daily": 3,
        "per_ip_burst": 2,
        "burst_window_s": 60.0,
    }
    opts.update(kw)
    return SpendGuard(clock=clock, **opts)  # type: ignore[arg-type]


class TestCheckKhongDemCommitMoiDem:
    def test_check_goi_bao_nhieu_lan_cung_khong_lam_thay_doi_bo_dem(self) -> None:
        guard = _guard(_Clock())
        for _ in range(50):
            assert guard.check("a").allowed
        assert guard.snapshot()["used_today"] == 0

    def test_commit_dem_ngay_ca_khi_luot_sinh_sau_do_that_bai(self) -> None:
        """⭐ Không có đường hoàn lại, và đó là chủ đích.

        Một lượt sinh chết giữa chừng **đã** tiêu token. Tính-khi-thành-công
        biến mọi lỗi thành một lần thử miễn phí.
        """
        guard = _guard(_Clock())
        assert guard.commit("a").allowed
        assert guard.snapshot()["used_today"] == 1


class TestBaLyDoTuChoiLaBaThongDiepKhacNhau:
    def test_tran_tong_chan_ke_ca_khach_moi_toanh(self) -> None:
        guard = _guard(_Clock(), daily_total=2, per_ip_daily=99, per_ip_burst=99)
        assert guard.commit("a").allowed
        assert guard.commit("b").allowed
        verdict = guard.commit("nguoi-chua-hoi-lan-nao")
        assert not verdict.allowed
        assert "hạn mức" in verdict.reason and "hôm nay" in verdict.reason

    def test_tran_khach_chan_dung_khach_do_va_khong_chan_nguoi_khac(self) -> None:
        guard = _guard(_Clock(), daily_total=99, per_ip_daily=2, per_ip_burst=99)
        assert guard.commit("a").allowed
        assert guard.commit("a").allowed
        assert not guard.commit("a").allowed
        assert guard.commit("b").allowed, "trần theo khách không được rò sang khách khác"

    def test_bam_nhanh_chan_tam_thoi_va_noi_ro_bao_lau(self) -> None:
        clock = _Clock()
        guard = _guard(clock, daily_total=99, per_ip_daily=99, per_ip_burst=2)
        assert guard.commit("a").allowed
        assert guard.commit("a").allowed
        verdict = guard.commit("a")
        assert not verdict.allowed
        assert "giây" in verdict.reason
        clock.now += 61
        assert guard.commit("a").allowed, "hết cửa sổ thì phải mở lại"

    def test_ba_ly_do_khong_dung_chung_mot_cau_chu(self) -> None:
        """Gộp thông điệp là bỏ mất thứ duy nhất phân biệt được ba chế độ."""
        reasons = set()
        g1 = _guard(_Clock(), daily_total=1, per_ip_daily=99, per_ip_burst=99)
        g1.commit("a")
        reasons.add(g1.commit("b").reason)
        g2 = _guard(_Clock(), daily_total=99, per_ip_daily=1, per_ip_burst=99)
        g2.commit("a")
        reasons.add(g2.commit("a").reason)
        g3 = _guard(_Clock(), daily_total=99, per_ip_daily=99, per_ip_burst=1)
        g3.commit("a")
        reasons.add(g3.commit("a").reason)
        g4 = _guard(_Clock())
        g4.trip()
        reasons.add(g4.commit("a").reason)
        assert len(reasons) == 4, reasons


class TestCauDaoVaVongDoiNgay:
    def test_trip_chan_tat_ca_va_khong_co_duong_bat_lai(self) -> None:
        guard = _guard(_Clock())
        guard.trip()
        assert not guard.check("a").allowed
        assert not guard.commit("a").allowed
        assert guard.snapshot()["tripped"] is True
        assert not hasattr(guard, "reset"), "cầu dao một chiều: bật lại phải là khởi động lại"

    def test_sang_ngay_utc_moi_thi_moi_bo_dem_ve_khong(self) -> None:
        clock = _Clock()
        guard = _guard(clock, daily_total=2, per_ip_daily=1, per_ip_burst=1)
        assert guard.commit("a").allowed
        assert not guard.commit("a").allowed
        clock.now += DAY
        assert guard.commit("a").allowed
        assert guard.snapshot()["used_today"] == 1

    def test_cau_dao_KHONG_mo_lai_khi_sang_ngay_moi(self) -> None:
        """⚠️ Ngắt thủ công là quyết định của người, không phải một bộ đếm."""
        clock = _Clock()
        guard = _guard(clock)
        guard.trip()
        clock.now += DAY * 3
        assert not guard.commit("a").allowed


class TestTranChiPhiInRaDungConSoBaoVeHoaDon:
    def test_max_usd_tinh_tu_tran_TONG_chu_khong_tu_tran_khach(self) -> None:
        guard = SpendGuard(daily_total=500, per_ip_daily=20, per_ip_burst=3)
        assert guard.max_usd_per_day == pytest.approx(1.0)
        assert guard.snapshot()["max_usd_per_day"] == pytest.approx(1.0)

    def test_don_gia_mac_dinh_la_tran_tren_cua_so_do_W5_11(self) -> None:
        """$0,0010701/câu đo được; $0,002 là trần trên có chủ ý, không phải số bịa."""
        assert SpendGuard(daily_total=1, per_ip_daily=1, per_ip_burst=1).usd_per_query >= 0.0010701


class TestKhoaDemLayTuDau:
    def test_uu_tien_x_forwarded_for_va_lay_phan_tu_TRAI_NHAT(self) -> None:
        """Trái nhất = chỗ edge của HF đặt địa chỉ khách. Xem `guard.py` quyết định 2."""
        key = client_key_of({"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, "10.0.0.1")
        assert key == hash_client("203.0.113.7")
        assert key != hash_client("10.0.0.1")

    def test_khong_co_header_thi_roi_ve_host_cua_ket_noi(self) -> None:
        assert client_key_of({}, "198.51.100.4") == hash_client("198.51.100.4")
        assert client_key_of(None, "198.51.100.4") == hash_client("198.51.100.4")

    def test_khong_co_gi_ca_thi_van_ra_mot_khoa_chu_khong_no(self) -> None:
        assert client_key_of(None, None)

    def test_khoa_KHONG_phai_dia_chi_ip(self) -> None:
        """IP là dữ liệu cá nhân; bộ đếm chỉ cần biết 'có cùng một người không'."""
        ip = "203.0.113.7"
        key = client_key_of({"x-forwarded-for": ip}, None)
        assert ip not in key
        assert len(key) == 16 and int(key, 16) >= 0

    def test_hai_khach_khac_nhau_cho_hai_khoa_khac_nhau(self) -> None:
        a = client_key_of({"x-forwarded-for": "203.0.113.7"}, None)
        b = client_key_of({"x-forwarded-for": "203.0.113.8"}, None)
        assert a != b


class TestDungTuBienMoiTruong:
    def test_mac_dinh_500_cau_tuc_toi_da_mot_do_la_mot_ngay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in (
            "DEMO_DAILY_TOTAL",
            "DEMO_PER_IP_DAILY",
            "DEMO_PER_IP_BURST",
            "DEMO_GENERATION",
        ):
            monkeypatch.delenv(name, raising=False)
        guard = guard_from_env()
        assert guard.daily_total == 500
        assert guard.max_usd_per_day == pytest.approx(1.0)
        assert guard.snapshot()["tripped"] is False

    @pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "False"])
    def test_DEMO_GENERATION_tat_lam_cau_dao_nhay_ngay_luc_dung(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("DEMO_GENERATION", value)
        assert guard_from_env().snapshot()["tripped"] is True

    def test_gia_tri_khong_phai_tat_thi_KHONG_lam_nhay_cau_dao(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEMO_GENERATION", "on")
        assert guard_from_env().snapshot()["tripped"] is False
