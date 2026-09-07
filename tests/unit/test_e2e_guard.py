"""Chốt skip của e2e phải skip ĐÚNG LÚC — và một bài test skip vô điều kiện thì im lặng.

⚠️ Đây là lỗ mà chính lượt tiêm lỗi của `W6-03` chỉ ra. `_require_api_container`
được thêm để `test_the_container_runs_the_versions_the_lockfile_pins` **skip** thay
vì **đỏ** khi chưa `make up-api`. Nhưng nếu ai đó sau này đơn giản hoá nó thành
`pytest.skip(...)` vô điều kiện thì **không có gì bắt được**: bộ e2e vẫn "xanh"
(21 skip), CI vẫn xanh, và phép kiểm duy nhất nối "cái đã đo" với "cái đang chạy"
im lặng biến mất khỏi dự án.

Một chốt skip là mã, và mã không có test thì nó tự do trôi về phía luôn-skip — vì
phía ấy không bao giờ làm ai khó chịu.

⭐ Nên bài test này khẳng định **cả hai chiều**:
  * không có container ⇒ `Skipped`
  * có container       ⇒ **không** skip, để bài kiểm phiên bản thật sự chạy

Nó là unit test và không cần Docker: `subprocess.run` bị thay bằng một hàm giả trả
đúng cái `docker compose ps -q` sẽ trả.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from tests.e2e.test_smoke import _require_api_container


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def _fake_run(stdout: str) -> Callable[..., _Completed]:
    def run(*_args: Any, **_kwargs: Any) -> _Completed:
        return _Completed(stdout)

    return run


def test_it_skips_when_no_container_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docker compose ps -q api` trả rỗng ⇒ điều kiện môi trường, không phải lỗi."""
    monkeypatch.setattr(subprocess, "run", _fake_run(""))

    with pytest.raises(pytest.skip.Exception, match="make up-api"):
        _require_api_container()


def test_it_does_NOT_skip_when_a_container_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """⭐⭐ Chiều quan trọng hơn: chốt không được nuốt bài test khi môi trường có sẵn.

    Một `pytest.skip()` vô điều kiện làm bộ e2e xanh mãi mãi và xoá sổ phép kiểm
    phiên bản thư viện mà không để lại dấu vết nào.

    ⚠️⚠️ Phải **bắt** `Skipped` rồi `fail`, không được chỉ gọi hàm trần. Một lượt
    tiêm lỗi đã chứng minh: biến chốt thành `pytest.skip()` vô điều kiện thì lời
    gọi trần ném `Skipped` ra giữa thân test, pytest ghi bài này là **skipped**, và
    exit code là **0**. Tức một bài test khẳng định "không skip" tự nó không thể
    đỏ — nó im lặng biến mất đúng lúc nó cần lên tiếng.
    """
    monkeypatch.setattr(subprocess, "run", _fake_run("f86bf73c56f4\n"))

    try:
        _require_api_container()
    except pytest.skip.Exception as exc:  # pragma: no cover - chỉ chạy khi hồi quy
        pytest.fail(f"chốt đã skip dù container đang chạy: {exc}")


def test_a_broken_docker_is_a_skip_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Không có `docker` trên PATH là điều kiện môi trường, không phải lỗi hồi quy."""

    def boom(*_args: Any, **_kwargs: Any) -> _Completed:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(pytest.skip.Exception, match="compose"):
        _require_api_container()


def test_it_asks_compose_only_for_RUNNING_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚠️ `docker compose ps -q api` (không `--status running`) trả id cả container ĐÃ DỪNG.

    Chốt sẽ để bài test chạy tiếp rồi `exec` vào một container không chạy, tức đỏ
    vì đúng lý do nó vừa được viết ra để tránh.
    """
    seen: list[list[str]] = []
    seen_kwargs: list[dict[str, Any]] = []

    def record(cmd: list[str], **kwargs: Any) -> _Completed:
        seen.append(cmd)
        seen_kwargs.append(kwargs)
        return _Completed("abc123\n")

    monkeypatch.setattr(subprocess, "run", record)
    try:
        _require_api_container()
    except pytest.skip.Exception as exc:  # pragma: no cover - chỉ chạy khi hồi quy
        pytest.fail(f"chốt đã skip dù container đang chạy: {exc}")

    assert seen, "chốt không hỏi compose lần nào"
    assert "--status" in seen[0] and "running" in seen[0], seen[0]
    # ⚠️ `check=True` là thứ khiến "compose lỗi thật" khác "không có container":
    # bỏ nó đi thì một lỗi compose trả `stdout` rỗng và bị đọc thành "chưa bật",
    # tức bài test phiên bản thư viện skip mất mà không ai biết.
    assert seen_kwargs[0].get("check") is True, seen_kwargs[0]


def test_it_propagates_a_real_compose_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """`check=True` phải giữ nguyên: compose lỗi thật thì thành skip có lời giải thích,
    chứ không phải một `stdout` rỗng bị đọc nhầm thành "không có container"."""

    def failing(cmd: list[str], **_kwargs: Any) -> _Completed:
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="no such service")

    monkeypatch.setattr(subprocess, "run", failing)

    with pytest.raises(pytest.skip.Exception, match="không hỏi được compose"):
        _require_api_container()
