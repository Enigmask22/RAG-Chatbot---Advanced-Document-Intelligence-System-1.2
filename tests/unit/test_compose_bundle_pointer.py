"""`bundles/CURRENT` phải là con trỏ mà container thật sự đi theo.

⭐⭐ **Một cơ chế ba tầng mà tầng 2 không bao giờ chạy được trong môi trường duy
nhất giống production.** `serving/api/app.py::_startup_version` giải quyết "bundle
nào chạy" theo ba tầng, giảm dần độ tường minh:

1. `BUNDLE_VERSION` — người vận hành ghim tay, thắng tất cả;
2. `bundles/CURRENT` — con trỏ phát hành của `W5-10`, thứ mà promote ghi vào và
   rollback ghi ngược lại, **và là đường mà production đi**;
3. bản semver cao nhất — một phép **đoán**, kèm `logger.warning` nói rõ là đoán.

`infra/docker-compose.yml` từng viết `BUNDLE_VERSION=${BUNDLE_VERSION:-0.2.0}`.
Hằng số ấy làm **tầng 1 luôn luôn nổ**, nên tầng 2 là mã chết trong container —
đúng chỗ nó được thiết kế để chạy.

⚠️ Cái giá đo được ngày 07/09/2026: `CURRENT` là `0.2.1`, container phục vụ
`0.2.0`, `/ready` báo `ready: true` suốt, và **mọi** tài liệu, mọi lượt eval cùng
`make gate` mô tả một bundle khác với bundle đang trả lời người dùng. Không test
nào đỏ, vì bài e2e hỏi *"container có phục vụ bundle nó KHAI không"* — và nó khai
đúng cái sai.

Đây là họ `AU-12`: hai nguồn sự thật cho một câu hỏi, và bản sao thứ hai mục đi
trong im lặng. Phép kiểm phải là **quan hệ** (compose không được tự khai một
phiên bản), không phải một giá trị chép sang file thứ ba.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = sorted((_REPO_ROOT / "infra").glob("docker-compose*.yml"))

# `- BUNDLE_VERSION=…` với bất kỳ giá trị nào, kể cả qua `${VAR:-mặc_định}`.
_ASSIGNED = re.compile(r"^\s*-\s*BUNDLE_VERSION\s*=\s*(\S.*)$", re.MULTILINE)


def test_there_is_a_current_pointer_at_all() -> None:
    pointer = _REPO_ROOT / "bundles" / "CURRENT"

    assert pointer.exists(), "không có `bundles/CURRENT`; tầng 2 không có gì để trỏ"
    assert pointer.read_text(encoding="utf-8").strip(), "`bundles/CURRENT` rỗng"


def test_the_pointer_names_a_bundle_that_exists() -> None:
    version = (_REPO_ROOT / "bundles" / "CURRENT").read_text(encoding="utf-8").strip()

    manifest = _REPO_ROOT / "bundles" / f"rag-bundle-v{version}" / "manifest.json"

    assert manifest.exists(), f"`CURRENT` trỏ tới {version} nhưng không có {manifest}"


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
def test_no_compose_file_hardcodes_a_bundle_version(compose: Path) -> None:
    """⚠️ Dạng trần `- BUNDLE_VERSION` là được: nó chỉ truyền biến khi biến ĐƯỢC ĐẶT.

    Dạng gán — kể cả `${BUNDLE_VERSION:-0.2.0}` trông như "chỉ là mặc định" — làm
    tầng 1 luôn thắng, và tầng 2 không bao giờ chạy.
    """
    assigned = _ASSIGNED.findall(compose.read_text(encoding="utf-8"))

    assert not assigned, (
        f"{compose.name} gán BUNDLE_VERSION={assigned} — tầng `bundles/CURRENT` "
        "thành mã chết. Dùng dạng trần `- BUNDLE_VERSION` và ghim trong `.env` "
        "nếu thật sự cần ghim."
    )


def test_the_example_env_does_not_pin_a_bundle_either() -> None:
    """`.env.example` là thứ người mới chép thành `.env`, nên một dòng ghim ở đó
    đi thẳng vào mọi máy dev và tái lập đúng lỗi trên."""
    example = _REPO_ROOT / ".env.example"
    if not example.exists():  # pragma: no cover - repo luôn có file này
        pytest.skip("không có .env.example")

    pinned = [
        line
        for line in example.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("BUNDLE_VERSION=") and line.split("=", 1)[1].strip()
    ]

    assert not pinned, f".env.example ghim sẵn bundle: {pinned}"
