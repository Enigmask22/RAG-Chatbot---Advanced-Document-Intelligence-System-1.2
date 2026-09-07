"""`W6-08` — bảng bằng chứng của CV phải còn trỏ vào thứ có thật.

DoD của `W6-08` là *"mọi dòng trong mục project đều trả lời được bằng file trong
repo"*. Một bảng viết tay thoả điều kiện ấy **vào ngày viết**, rồi thôi thoả —
âm thầm — ở lần đầu tiên ai đó đổi tên một báo cáo hoặc bỏ một target Makefile.
Đó là họ `AU-12`: hai nơi giữ cùng một sự thật, và chỉ một nơi được cập nhật.

Chỗ hỏng ở đây tệ hơn một liên kết chết trong tài liệu nội bộ, vì bảng ấy là
thứ đứng sau **một con số trong CV**. Câu hỏi phỏng vấn *"chứng minh đi"* mà
nhận lại một đường dẫn 404 thì tệ hơn hẳn việc không đưa con số ấy ra.

⚠️ Bài này **đã đỏ ngay lần chạy đầu**: bản đầu của báo cáo trỏ vào
`tasks/w4-12-guardrails.md` (báo cáo thật tên `security-w4.md`) và
`data/goldenset/golden_v1.jsonl` (thật là `data/golden/`). Hai lỗi trong một
bảng vừa viết xong — chính xác kiểu mục nát mà nó sinh ra để bắt, chỉ là sớm
hơn dự kiến.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "plans" / "reports" / "tasks" / "w6-07-08-cv.md"
REPORTS_ROOT = REPO / "plans" / "reports"

#: Đuôi file được coi là "một đường dẫn cần tồn tại". Cố ý **không** nhận mọi
#: chuỗi có dấu `/`: `provider:model`, `CC BY 3.0 IGO` và `1/rerank-time` cũng
#: có, và bắt chúng sẽ làm bài test ồn tới mức bị tắt.
SUFFIXES = (".md", ".json", ".jsonl", ".py", ".yaml", ".yml", ".sha256")

#: Đường dẫn tới thứ **không** nằm trong git: cache cục bộ, và file CV sống ở
#: repo khác. Cả hai được nêu tên có chủ đích trong báo cáo, nên loại theo tiền
#: tố chứ không im lặng bỏ qua mọi thứ không tìm thấy.
NGOAI_REPO = (".cache/", "main.tex")


def _khong_rao(text: str) -> str:
    """Bỏ khối rào ```` ``` ```` **trước** khi ghép cặp backtick.

    ⚠️ Không bỏ thì một rào ba-backtick làm **lệch cặp** cho toàn bộ phần còn
    lại của file: cả khối LaTeX bị nuốt thành một "token" và mọi đường dẫn sau
    nó ghép sai. Lần chạy đầu bắt được 2/17 đường dẫn và ba bài dưới vẫn xanh —
    nhóm chứng `test_bang_bang_chung_khong_rong` tồn tại đúng vì lượt ấy.
    """
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _paths_in(text: str) -> list[str]:
    """Chỉ soi **hàng bảng**, không soi văn xuôi.

    ⚠️⚠️ Luật này không phải để cho dễ. Bản đầu quét cả file và đỏ ở đúng đoạn
    §6 **kể lại hai đường dẫn hỏng mà nó vừa tìm ra** — báo cáo phải được phép
    gọi tên `tasks/w4-12-guardrails.md` để giải thích rằng nó *không* tồn tại.

    Cùng khuôn với bài học `W6-02` (một phép kiểm hỏi *"chuỗi này có xuất hiện
    ở đâu không"* đỏ vì chú thích giải thích luật) và với bài kiểm khối LaTeX ở
    cuối file. Phân biệt được là **vai trò**, không phải nội dung: trong bảng,
    một đường dẫn là **bằng chứng được đưa ra**; trong văn xuôi nó có thể là
    **tang vật**. Bảng là thứ người phỏng vấn sẽ bấm vào.
    """
    hang_bang = "\n".join(
        line for line in _khong_rao(text).splitlines() if line.lstrip().startswith("|")
    )
    found = []
    for token in re.findall(r"`([^`]+)`", hang_bang):
        token = token.strip()
        if "/" not in token or not token.endswith(SUFFIXES):
            continue
        if any(token.startswith(x) for x in NGOAI_REPO):
            continue
        found.append(token)
    return sorted(set(found))


def _resolve(token: str) -> Path | None:
    """Repo-root trước, rồi `plans/reports/`.

    Bảng viết tắt `runs/…` và `tasks/…` vì trong ngữ cảnh báo cáo chúng rõ
    nghĩa; giữ cách viết ấy và cho bài test biết luật, thay vì bắt người viết
    gõ tiền tố đầy đủ ở mỗi ô.
    """
    for base in (REPO, REPORTS_ROOT):
        candidate = base / token
        if candidate.exists():
            return candidate
    return None


def _make_targets() -> set[str]:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([a-z][a-z0-9-]*):", text, flags=re.MULTILINE))


class TestBaoCaoCvTonTai:
    def test_bao_cao_co_mat(self) -> None:
        assert REPORT.is_file(), "W6-07/W6-08 không có báo cáo thì không có bảng để kiểm"


class TestMoiDuongDanConTro:
    def test_bang_bang_chung_khong_rong(self) -> None:
        """Nhóm chứng. Một regex hỏng cho 0 đường dẫn và **mọi** bài dưới xanh."""
        assert len(_paths_in(REPORT.read_text(encoding="utf-8"))) >= 15

    def test_moi_duong_dan_duoc_neu_ten_deu_ton_tai(self) -> None:
        chet = [t for t in _paths_in(REPORT.read_text(encoding="utf-8")) if _resolve(t) is None]
        assert not chet, f"báo cáo CV trỏ vào file không tồn tại: {chet}"

    def test_bo_phan_giai_co_the_that_bai(self) -> None:
        """⚠️ Một bài kiểm "không có gì hỏng" phải chứng minh được nó **biết**
        hỏng trông thế nào — nếu không nó chỉ đang khẳng định rằng nó chạy."""
        assert _resolve("tasks/khong-he-co-bao-cao-nay.md") is None


class TestMoiLenhMakeConLaTarget:
    def test_moi_lenh_make_trong_bao_cao_la_target_that(self) -> None:
        text = REPORT.read_text(encoding="utf-8")
        goi = set(re.findall(r"`make ([a-z][a-z0-9-]*)", _khong_rao(text)))
        assert goi, "nhóm chứng: không bắt được lệnh make nào"
        thieu = sorted(goi - _make_targets())
        assert not thieu, f"báo cáo CV nêu target Makefile không tồn tại: {thieu}"


class TestNhungThuKHONGDuocQuayLaiBaoCao:
    """Năm claim đã bị bỏ ở §0. Bảng ⛔ chỉ là chữ; đây là hàng rào."""

    @pytest.mark.parametrize(
        ("cam", "ly_do"),
        [
            ("signed RagBundle", "bundle được checksum sha256, không ai ký"),
            ("Qwen3-8B", "ablation bộ sinh là DeepSeek vs GLM, chưa từng chạy vLLM"),
            ("250-query", "golden_v1 có 242 câu"),
        ],
    )
    def test_claim_da_bo_khong_quay_lai_trong_khoi_latex(self, cam: str, ly_do: str) -> None:
        """Chỉ soi **khối LaTeX** — §0 phải được phép gọi tên thứ nó vừa bác bỏ.

        Một bài hỏi "chuỗi này có xuất hiện ở đâu không" sẽ đỏ vì chính đoạn
        văn giải thích tại sao nó sai; cùng cái bẫy `W6-02` đã gặp.
        """
        khoi = re.findall(r"```latex\n(.*?)```", REPORT.read_text(encoding="utf-8"), re.DOTALL)
        assert khoi, "nhóm chứng: không tìm thấy khối latex nào"
        for block in khoi:
            assert cam not in block, f"{cam!r} quay lại CV — {ly_do}"
