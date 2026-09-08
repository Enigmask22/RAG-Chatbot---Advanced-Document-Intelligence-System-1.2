"""`TD-62` — khai ra phiên bản thư viện, và ba cách khai sai.

Sự cố gốc: container chạy `transformers 5.16.1 / sentence-transformers 6.0.1 /
torch 2.14.0` thay vì `5.15.0 / 5.7.0 / 2.13.0` của lock, cross-encoder **không
phục vụ được**, và `GET /admin/bundle` trả `runtime_drift: null` suốt thời gian
ấy — vì phép so danh tính của `W4-02` chỉ nhìn `(model, device, dtype)`.

Ba điều phải đúng, và cả ba đều là cách một khối *"khai thông tin"* trở thành
vô dụng hoặc gây hại:

1. **Không được ném.** Một triển khai dense-only hợp lệ **sẽ** thiếu
   `sentence-transformers`; ném ở đây biến một cấu hình đúng thành 500.
2. **Phải đúng ba gói ấy**, không phải cả môi trường: một danh sách 200 dòng là
   một danh sách không ai đọc, và thứ cần đọc chìm trong đó.
3. **Phải nằm ngoài `active_detail`.** Trong khối mô tả bundle thì nó đọc như
   thể bundle khai ra nó — mà bundle **không** khai, và đó chính là phần `TD-62`
   cố ý *không* làm.
"""

from __future__ import annotations

from pathlib import Path

from serving.core.libversions import MODEL_PACKAGES, library_versions

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestKhongDuocNem:
    def test_goi_khong_ton_tai_cho_None_chu_khong_ném(self) -> None:
        """⭐ `None` là một câu trả lời **có nghĩa**: nó nói *"gói này không có
        mặt"*, thứ mà một triển khai dense-only hợp lệ sẽ báo. Ném ở đây biến
        một cấu hình đúng thành một `/admin/bundle` 500."""
        ra = library_versions(("khong-he-ton-tai-goi-nay",))
        assert ra == {"khong-he-ton-tai-goi-nay": None}

    def test_tron_goi_co_va_khong_co_thi_van_tra_du(self) -> None:
        ra = library_versions(("pytest", "khong-he-ton-tai-goi-nay"))
        assert ra["pytest"] is not None
        assert ra["khong-he-ton-tai-goi-nay"] is None


class TestDungBaGoiCanDoc:
    def test_dung_ba_goi_cua_su_co(self) -> None:
        """Ghim **danh sách**, không ghim ý định. Thêm gói thứ tư là một quyết
        định (nó làm loãng ba dòng cần đọc), nên nó phải đi qua bài test này."""
        assert MODEL_PACKAGES == ("torch", "transformers", "sentence-transformers")

    def test_mac_dinh_doc_dung_MODEL_PACKAGES(self) -> None:
        assert set(library_versions()) == set(MODEL_PACKAGES)

    def test_torch_va_transformers_co_that_trong_moi_truong_nay(self) -> None:
        """Nhóm chứng: nếu hàm luôn trả `None` thì mọi bài trên vẫn xanh."""
        ra = library_versions()
        assert ra["torch"] and ra["transformers"], ra


class TestKhaiODungCho:
    def test_o_ngoai_active_detail_chu_khong_o_trong(self) -> None:
        """⭐⭐ Phiên bản thư viện là thuộc tính của **tiến trình**, không của
        bundle. Đặt nó trong `active_detail` đọc như thể manifest khai ra nó —
        mà manifest **không**, và việc để manifest khai là thứ `TD-62` cố ý
        không làm (nó đòi thêm trường vào `RagBundle`, và `TD-36` đã trả giá
        một lần cho bài học "thêm trường có mặc định làm vỡ chữ ký bundle cũ").

        Bài này đọc **mã nguồn** của route: dựng một app thật chỉ để xem một
        khoá nằm ở tầng nào của JSON là trả giá integration cho một phép so
        chuỗi, và `_describe` là hàm thuần nên nó không có cách nào khác để bị
        quan sát.
        """
        # ⚠️ Neo vào gốc repo, **không** dùng đường dẫn tương đối: bản nháp
        # viết `Path("serving/api/admin.py")`, thứ chỉ đúng khi pytest được gọi
        # từ gốc — cùng họ với ba lỗi khác của ngày hôm nay, đều là *"kết quả
        # do một thứ ngoài bài test quyết định"*.
        nguon = (_REPO_ROOT / "serving" / "api" / "admin.py").read_text(encoding="utf-8")
        than_describe = nguon.split("def _describe(")[1].split("\n@router")[0]
        assert "library_versions" not in than_describe
        assert 'payload["library_versions"] = library_versions()' in nguon
