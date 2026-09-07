"""`W6-02` — những gì rời khỏi repo này để lên một Space công khai.

Space là **repo thứ hai**, và mọi repo thứ hai là một chỗ để hai bản sao lệch
nhau mà không ai thấy. Bộ test này canh ba loại lệch:

1. **Lệch mã.** `requirements.txt` phải ghim một *commit*, không phải `@main`.
   `@main` nghĩa là mỗi lần Space build lại là một hệ thống khác — và số đo
   trong README của Space nói về hệ thống nào thì không ai biết.
2. **Lệch cấu hình.** Frontmatter phải khai một `python_version` mà ZeroGPU
   thật sự chạy, và **không** được ghim `gradio`/`spaces` ở nơi thứ hai.
3. **Lệch nghĩa vụ.** Corpus là CC BY 3.0 IGO của World Bank. Ghi công là điều
   kiện của giấy phép, không phải một dòng trang trí — nên nó có test.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SPACE = REPO / "space"
DEPLOY = REPO / "scripts" / "deploy_space.py"

#: Tài liệu ZeroGPU (07/09/2026) liệt kê đúng hai bản. Ghim vào test vì một
#: `python_version` ngoài danh sách này làm Space **không build được**, và ta
#: chỉ biết điều đó sau vài phút build.
ZEROGPU_PYTHON = {"3.12.12", "3.10.13"}


def _deploy_module() -> Any:
    spec = importlib.util.spec_from_file_location("deploy_space", DEPLOY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frontmatter() -> dict[str, str]:
    text = (SPACE / "README.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "README của Space phải mở đầu bằng frontmatter YAML"
    out: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip().strip('"')
    return out


class TestRequirementsGhimMotHeThongCuThe:
    def test_ghim_commit_chu_khong_phai_nhanh(self) -> None:
        text = (SPACE / "requirements.txt").read_text(encoding="utf-8")
        pinned = [ln for ln in text.splitlines() if "git+https://github.com" in ln]
        assert pinned, "phải cài rag-platform từ git, không chép mã vào Space"
        for line in pinned:
            assert "@__GIT_SHA__" in line, f"chưa để chỗ điền SHA: {line}"
            assert "@main" not in line and "@master" not in line, line

    def test_cai_tu_dung_repo_cong_khai(self) -> None:
        text = (SPACE / "requirements.txt").read_text(encoding="utf-8")
        assert "github.com/Enigmask22/RAG-Chatbot" in text

    @pytest.mark.parametrize("forbidden", ["spaces", "gradio"])
    def test_KHONG_ghim_thu_ma_nen_tang_tu_ghim(self, forbidden: str) -> None:
        """⚠️ `spaces` do nền tảng ghim (ghim lần hai ⇒ pip resolve hỏng);
        `gradio` do `sdk_version` trong frontmatter quyết định."""
        for line in (SPACE / "requirements.txt").read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name = re.split(r"[=<>@\[ ]", stripped, maxsplit=1)[0].strip().lower()
            assert name != forbidden, f"{forbidden} không được ghim ở đây: {line}"


class TestFrontmatterKhaiThuZeroGpuChayDuoc:
    def test_sdk_la_gradio(self) -> None:
        """ZeroGPU **chỉ** nhận Gradio SDK — Docker và Static không lên được."""
        assert _frontmatter()["sdk"] == "gradio"

    def test_python_version_nam_trong_danh_sach_ZeroGPU_ho_tro(self) -> None:
        version = _frontmatter().get("python_version")
        assert version in ZEROGPU_PYTHON, (
            f"python_version={version!r} — ZeroGPU chỉ chạy {sorted(ZEROGPU_PYTHON)}"
        )

    def test_co_sdk_version_va_no_la_mot_so_phien_ban(self) -> None:
        assert re.fullmatch(r"\d+\.\d+\.\d+", _frontmatter().get("sdk_version", ""))

    def test_app_file_tro_dung_file_co_that(self) -> None:
        assert (SPACE / _frontmatter()["app_file"]).is_file()


class TestNghiaVuGhiCongCuaGiayPhepCorpus:
    def test_readme_neu_dung_giay_phep_va_chu_so_huu(self) -> None:
        """CC BY = ghi công là **điều kiện**. `data/README.md` đã ghi luật này."""
        text = (SPACE / "README.md").read_text(encoding="utf-8")
        assert "CC BY 3.0 IGO" in text
        assert "World Bank" in text

    def test_giao_dien_cung_neu_giay_phep_chu_khong_chi_readme(self) -> None:
        """Người dùng demo đọc giao diện, không đọc README của repo Space."""
        text = (SPACE / "app.py").read_text(encoding="utf-8")
        assert "CC BY 3.0 IGO" in text and "World Bank" in text

    def test_readme_noi_ro_cau_tra_loi_co_the_sai(self) -> None:
        assert "có thể sai" in (SPACE / "README.md").read_text(encoding="utf-8")


class TestAppKhongDungLaiNhungThuDaCoMotNguon:
    def test_doc_con_tro_bang_read_pointer_chu_khong_tu_doc_CURRENT(self) -> None:
        """⭐ `W5-10` dựng `read_pointer` để xoá một bản sao thứ hai (`AU-12`);
        tự `read_text("CURRENT")` là dựng lại đúng bản sao ấy. Bản đầu của
        `app.py` làm đúng như vậy.

        ⚠️ Soi **AST**, không soi văn bản. Bản đầu của bài test này hỏi
        `'"CURRENT"' not in source` và đỏ ngay — vì đúng dòng chú thích *giải
        thích tại sao không được làm thế* có chứa chuỗi ấy. Cùng họ với ba lỗ
        mà lượt tiêm của `W6-03` tìm ra: "chuỗi này có xuất hiện ở đâu không"
        không bao giờ là câu hỏi đang cần hỏi. Chú thích không có trong AST.
        """
        import ast

        tree = ast.parse((SPACE / "app.py").read_text(encoding="utf-8"))
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        assert "CURRENT" not in literals, (
            "app.py không được tự dựng đường dẫn tới CURRENT — dùng read_pointer"
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "read_pointer" in called

    def test_KHONG_bat_runtime_drift(self) -> None:
        """`TD-57`: bật drift là đúng kiểu nói dối mà `TD-38` sinh ra để chặn."""
        text = (SPACE / "app.py").read_text(encoding="utf-8")
        assert "allow_runtime_drift=False" in text
        assert "allow_runtime_drift=True" not in text

    def test_tat_warmup_vi_TD_72_khong_hop_le_tren_ZeroGPU(self) -> None:
        assert "warmup=False" in (SPACE / "app.py").read_text(encoding="utf-8")


class TestNoiChuaNoiDungKhongTinDuocPhaiKhaiSanitize:
    @pytest.mark.parametrize("component", ["gr.Chatbot(", 'gr.Markdown("_Chưa có lượt nào._"'])
    def test_khai_sanitize_html_tuong_minh(self, component: str) -> None:
        """⚠️ Mặc định đã là True — và một mặc định là thứ đổi được ở phiên bản
        sau mà không ai đọc changelog. `W6-01` đặt cùng luật cho trang HTML."""
        text = (SPACE / "app.py").read_text(encoding="utf-8")
        start = text.index(component)
        window = text[start : start + 400]
        assert "sanitize_html=True" in window, f"{component} thiếu khai sanitize_html"

    def test_noi_dung_chunk_di_vao_code_fence(self) -> None:
        """Markdown không diễn giải bên trong ``` — đường tương đương của
        `textContent` ở `W6-01`."""
        text = (SPACE / "app.py").read_text(encoding="utf-8")
        assert "```text" in text


class TestCuaChanTruocKhiDay:
    def test_lap_thay_SHA_va_bo_file_lock(self, tmp_path: Path) -> None:
        module = _deploy_module()
        index = tmp_path / "idx"
        (index / "collection").mkdir(parents=True)
        (index / "collection" / "storage.sqlite").write_bytes(b"gia-lap")
        (index / ".lock").write_text("", encoding="utf-8")
        dest = tmp_path / "out"
        dest.mkdir()

        module._stage(dest, "deadbeefcafe", index, "0.2.1")

        req = (dest / "requirements.txt").read_text(encoding="utf-8")
        assert "deadbeefcafe" in req
        assert module.SHA_PLACEHOLDER not in req
        assert not (dest / "index" / ".lock").exists(), (
            ".lock là khoá tiến trình, không phải dữ liệu"
        )
        assert (dest / "index" / "collection" / "storage.sqlite").is_file()
        assert (dest / "bundles" / "CURRENT").read_text(encoding="utf-8").strip() == "0.2.1"
        assert "filter=lfs" in (dest / ".gitattributes").read_text(encoding="utf-8")

    def test_moi_file_trong_danh_sach_trang_deu_ton_tai(self) -> None:
        module = _deploy_module()
        for name in module.APP_FILES:
            assert (SPACE / name).is_file(), name

    def test_danh_sach_trang_phu_het_ma_python_cua_space(self) -> None:
        """⚠️ Danh sách **trắng**: thêm một module mà quên khai thì Space thiếu
        file và chỉ biết lúc nó không import được."""
        module = _deploy_module()
        on_disk = {p.name for p in SPACE.glob("*.py")}
        assert on_disk <= set(module.APP_FILES), on_disk - set(module.APP_FILES)

    def test_cua_3_bat_lech_so_point(self, tmp_path: Path) -> None:
        """⭐⭐ Lệch ⇒ `_check_size` giết Space lúc khởi động. Biết ở đây trong
        hai giây, thay vì ở đó sau một lần upload 239 MB."""
        from qdrant_client import QdrantClient, models

        module = _deploy_module()
        index = tmp_path / "idx"
        client = QdrantClient(path=str(index))
        client.create_collection(
            "c",
            vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)},
        )
        client.upsert(
            "c", points=[models.PointStruct(id=1, vector={"dense": [1.0, 0.0]}, payload={})]
        )
        client.close()

        manifest = {"components": {"index": {"collection": "c", "n_chunks": 999}}}
        with pytest.raises(SystemExit, match="cửa 3"):
            module._gate_index(index, manifest)

        manifest["components"]["index"]["n_chunks"] = 1
        assert module._gate_index(index, manifest) == 1


class TestBundleGuiDiKhopConTro:
    def test_manifest_cua_bundle_CURRENT_khai_dung_phien_ban_do(self) -> None:
        """Cửa 4 — `NEW-13` là bài học về hai nguồn sự thật cho một con số."""
        from rag_core.bundle.store import read_pointer

        version = read_pointer(REPO / "bundles")
        assert version
        manifest = json.loads(
            (REPO / "bundles" / f"rag-bundle-v{version}" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["bundle_version"] == version
