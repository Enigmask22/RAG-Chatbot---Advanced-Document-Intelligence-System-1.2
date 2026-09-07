"""CI chạy đúng những gì nó nói là chạy — `W5-09`.

## ⭐⭐ Một CI xanh chứng nhận thứ nó **đã chạy**, không phải thứ nó tên là

Bộ test đầy đủ không chạy được trên runner GitHub: `weights` cần ~4 GB
docling/EasyOCR cộng trọng số tải về, `gpu` cần GPU, `e2e` cần image 7,35 GB.
Bỏ chúng ra là bắt buộc; bỏ **mà không nói** thì dấu tick xanh trên PR trở
thành một lời khai không có nội dung.

Nên module này đọc **chính `.github/workflows/ci.yml`** — không phải một danh
sách chép tay — và đòi ba tính chất:

1. **Phủ.** Mọi tổ hợp marker *chạy được ở CI* phải được ít nhất một tầng nhận.
   Một bài không thuộc tầng nào là một bài không bao giờ chạy, và không có gì
   báo ra: `pytest -m` đơn giản là chọn ra rỗng.
2. **Kín.** Không tầng nào được nhận một bài mang marker đã khai là không chạy
   được. Nếu có, job sẽ đỏ vì thiếu phụ thuộc chứ không vì mã sai.
3. **Chính tả.** Mọi tên trong biểu thức `-m` phải là marker đã đăng ký.
   ⚠️ Đây là chỗ hỏng câm nhất: `-m "not weigths"` **không** báo lỗi, nó chọn
   đúng mọi bài — và `--strict-markers` không cứu được, vì nó chỉ gác marker
   *gắn lên test*, không gác marker *viết trong biểu thức chọn*.

Cùng lý lẽ với `test_metrics_endpoint.py` của `W5-07`: đọc artifact thật, không
đọc một bản sao của nó.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = Path(".github/workflows/ci.yml")
PYPROJECT = Path("pyproject.toml")

WEIGHTS_MODULES = {
    "tests/integration/test_ocr_fallback.py",
    "tests/unit/test_loaders.py",
    "tests/unit/test_parse_pin.py",
    "tests/unit/test_scan_detection.py",
    "tests/unit/test_structure_chunker.py",
}
"""Toàn bộ lỗ của tầng CI nhanh, viết ra thành một danh sách.

⭐ Danh sách này là **hợp đồng**, không phải tài liệu: thêm một module cần
docling mà quên khai ở đây thì `test_the_weights_set_is_exactly_declared` đỏ.
Không có nó, một module mới lặng lẽ rơi khỏi CI và không ai biết cho tới lúc
production hỏng ở đúng chỗ ấy.
"""

NOT_RUN_IN_CI = {
    "gpu": "runner GitHub không có GPU",
    "e2e": "cần image serving 7,35 GB đã build (`make up-api`)",
    "weights": "cần ~4 GB docling/EasyOCR + trọng số tải về mỗi lượt chạy",
}
"""Marker mà CI **cố ý** không chạy, kèm lý do. Xem `W5-10` cho nơi chúng chạy."""

_APPLIES_WEIGHTS = re.compile(
    r"(?m)^(?:pytestmark\s*=.*|\s*@)pytest\.mark\.weights|^pytestmark\s*=\s*\[[^\]]*weights"
)
"""Marker được **áp dụng**, không phải chỉ được nhắc tới.

⚠️ Bản đầu tìm chuỗi `"pytest.mark.weights"` và nó khớp **chính file này** —
một bài test grep một chuỗi thì bản thân nó cũng chứa chuỗi ấy. Đỏ ngay lần
chạy đầu, may hơn là im lặng đúng.
"""

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS = {"and", "or", "not"}


def _registered_markers() -> set[str]:
    """Tên marker đọc từ `pyproject.toml` — nguồn sự thật của `--strict-markers`."""
    text = PYPROJECT.read_text(encoding="utf-8")
    block = text.split("markers = [", 1)[1].split("\n]", 1)[0]
    return {
        line.strip().strip('",').split(":", 1)[0]
        for line in block.splitlines()
        if line.strip().startswith('"')
    }


def _tier_expressions() -> dict[str, str]:
    """`{tên job: biểu thức -m}` bóc từ mọi bước `run:` của workflow."""
    workflow: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            command = str(step.get("run", ""))
            match = re.search(r'pytest\s+-m\s+"([^"]+)"', command)
            if match:
                found[job_name] = match.group(1)
    return found


def _matches(expression: str, active: frozenset[str], universe: set[str]) -> bool:
    """`pytest -m` dùng cú pháp con của Python, nên `eval` là bản dịch đúng."""
    scope = {name: (name in active) for name in universe}
    return bool(eval(expression, {"__builtins__": {}}, scope))


@pytest.fixture(scope="module")
def markers() -> set[str]:
    return _registered_markers()


@pytest.fixture(scope="module")
def tiers() -> dict[str, str]:
    found = _tier_expressions()
    assert found, f"không tìm thấy lệnh `pytest -m` nào trong {WORKFLOW}"
    return found


class TestTheWorkflowIsReadable:
    def test_the_file_exists_and_parses(self) -> None:
        assert WORKFLOW.exists(), "W5-09 chưa có workflow"
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        assert set(workflow["jobs"]) >= {"lint", "unit", "integration", "smoke-eval"}

    def test_it_runs_on_pull_requests(self) -> None:
        """Nửa đầu DoD: *"PR mở ra là CI chạy"*.

        ⚠️ `yaml.safe_load` biến khoá `on:` thành `True` (chuẩn YAML 1.1 đọc
        `on` là boolean). Không biết điều đó thì bài này tìm khoá `"on"` và
        không thấy — rồi người ta sửa bài test thay vì hiểu.
        """
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        triggers = workflow.get("on", workflow.get(True))
        assert triggers is not None, "không đọc được khối trigger"
        assert "pull_request" in triggers


class TestEveryTestBelongsToATier:
    def test_no_marker_combination_falls_through(
        self, tiers: dict[str, str], markers: set[str]
    ) -> None:
        """Tính chất **phủ**: không tổ hợp nào rơi ra ngoài mọi tầng."""
        runnable = sorted(markers - set(NOT_RUN_IN_CI))
        orphans = []
        for size in range(len(runnable) + 1):
            for combo in itertools.combinations(runnable, size):
                active = frozenset(combo)
                if not any(_matches(expr, active, markers) for expr in tiers.values()):
                    orphans.append(sorted(active) or ["(không marker nào)"])
        assert not orphans, f"tổ hợp marker không tầng nào chạy: {orphans}"

    def test_no_tier_claims_something_ci_cannot_run(
        self, tiers: dict[str, str], markers: set[str]
    ) -> None:
        """Tính chất **kín**: một tầng nhận bài `weights` sẽ đỏ vì thiếu 4 GB
        phụ thuộc, tức đỏ vì hạ tầng chứ không vì diff — thứ dạy người ta bỏ
        qua màu đỏ."""
        others = sorted(markers - set(NOT_RUN_IN_CI))
        violations = []
        for blocked in NOT_RUN_IN_CI:
            # ⚠️ Phải quét MỌI tổ hợp chứa `blocked`, không chỉ `{blocked}` một
            # mình. Bản đầu chỉ thử singleton và nó xanh — trong khi
            # `test_reranked_retriever.py` mang `{integration, gpu}` và biểu
            # thức `integration and not weights` nhận nó. Repro trên Linux tìm
            # ra 9 bài chết vì CUDA; bài canh này thì không.
            for size in range(len(others) + 1):
                for combo in itertools.combinations(others, size):
                    active = frozenset({blocked, *combo})
                    for job, expr in tiers.items():
                        if _matches(expr, active, markers):
                            violations.append(f"{job} nhận {sorted(active)} qua biểu thức {expr!r}")
        assert not violations, "; ".join(sorted(set(violations))[:6])

    def test_every_name_in_a_tier_expression_is_a_registered_marker(
        self, tiers: dict[str, str], markers: set[str]
    ) -> None:
        """⚠️ Gõ sai một marker trong `-m` **không** báo lỗi — nó chọn tất cả."""
        for job, expr in tiers.items():
            names = {n for n in _NAME.findall(expr) if n not in _KEYWORDS}
            unknown = names - markers
            assert not unknown, f"job {job}: marker không đăng ký {sorted(unknown)}"


class TestTheHoleIsDeclared:
    def test_the_weights_set_is_exactly_declared(self) -> None:
        """Module nào mang `weights` — đọc từ mã, so với danh sách khai."""
        found = {
            str(path).replace("\\", "/")
            for path in Path("tests").rglob("test_*.py")
            if _APPLIES_WEIGHTS.search(path.read_text(encoding="utf-8"))
        }
        assert found == WEIGHTS_MODULES, (
            f"lỗ của CI đã đổi. Thêm: {sorted(found - WEIGHTS_MODULES)} · "
            f"Bớt: {sorted(WEIGHTS_MODULES - found)}. Sửa `WEIGHTS_MODULES` "
            "**và** nói ra trong PR — đây là phần CI không gác."
        )

    def test_each_excluded_marker_carries_a_reason(self, markers: set[str]) -> None:
        assert set(NOT_RUN_IN_CI) <= markers
        assert all(reason.strip() for reason in NOT_RUN_IN_CI.values())

    def test_the_smoke_tier_is_not_a_pytest_tier(self, tiers: dict[str, str]) -> None:
        """`smoke-eval` cố ý **không** là một job pytest: nó chạy
        `python -m pipeline.eval.smoke` và trả exit code của riêng nó. Nếu một
        ngày nó thành `pytest -m smoke` thì bài phủ ở trên phải tính cả nó."""
        assert "smoke-eval" not in tiers


class TestAFailedTierIsDiagnosable:
    """⭐⭐ Một lượt CI đỏ mà không đọc được tên bài đỏ thì chỉ là một lời đồn.

    Repo này không có `gh` CLI và không có token, nên API **log** của Actions trả
    403: một lượt đỏ đọc được đúng chữ *"Process completed with exit code 1"*. Đã
    phải đoán **hai lần trong một phiên** 07/09/2026, và lần thứ hai là một bài
    chập chờn ở tầng integration mà cùng mã ấy vừa xanh ở commit ngay trước.

    ⭐ Thứ đọc được **không cần token** là annotation của check run
    (`/check-runs/{id}/annotations`), và `::error::` đi thẳng vào đó. Nên mỗi tầng
    pytest phải ghi log ra tệp rồi phát tên bài đỏ thành annotation.

    ⚠️ Phép kiểm ở đây là **quan hệ**: mọi job chạy `pytest` phải có bước ấy. Ghim
    tên job thì bài test này sẽ đúng cho tới đúng ngày ai đó thêm tầng thứ ba.
    """

    @staticmethod
    def _jobs_running_pytest() -> dict[str, list[dict[str, object]]]:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        found = {}
        for name, job in workflow["jobs"].items():
            steps = job.get("steps", [])
            if any("pytest -m" in str(step.get("run", "")) for step in steps):
                found[name] = steps
        return found

    def test_every_pytest_job_writes_its_output_to_a_file(self) -> None:
        for name, steps in self._jobs_running_pytest().items():
            runs = " ".join(str(step.get("run", "")) for step in steps)
            assert "pytest.log" in runs, (
                f"job `{name}` chạy pytest nhưng không ghi log ra tệp — "
                "không có gì để trích tên bài đỏ ra annotation"
            )
            assert "set -o pipefail" in runs, (
                f"job `{name}` dẫn pytest qua `tee` mà không có `pipefail`: "
                "exit code của `tee` (luôn 0) sẽ che mất pytest đỏ"
            )

    def test_every_pytest_job_emits_failures_as_annotations(self) -> None:
        for name, steps in self._jobs_running_pytest().items():
            emitters = [step for step in steps if "::error::" in str(step.get("run", ""))]
            assert emitters, f"job `{name}` không phát `::error::` nào khi đỏ"
            assert all("failure()" in str(step.get("if", "")) for step in emitters), (
                f"bước annotation của job `{name}` phải chạy `if: failure()`; "
                "chạy luôn thì mỗi lượt xanh cũng đẻ ra annotation rỗng"
            )
