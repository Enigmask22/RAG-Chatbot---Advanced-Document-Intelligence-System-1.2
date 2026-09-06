"""`W0-08` — đóng gói job cho GPU thuê, và quét bí mật trước khi nó rời máy.

Quy tắc cứng #2 nói: *pod chỉ chạy job GPU-bound tự chứa, **không mang API
key**.* Lời hứa ấy cần một cơ chế, không phải một thói quen — vì thứ làm hỏng nó
sẽ là một lần `tar -czf` vội vàng lúc 11 giờ đêm khi pod đang tính tiền.

## Cơ chế: `git archive`, không phải `tar`

Phần mã nguồn của gói dựng bằng `git archive HEAD`. Điều đó **không thể** chứa
file chưa commit hoặc bị `.gitignore` — mà `.env`, `.venv/`, `.cache/`,
`data/corpus/` (DVC) đều thuộc nhóm đó. Bảo đảm này là tính chất của công cụ,
không phải của danh sách loại trừ do tôi nhớ ra; danh sách loại trừ thì luôn
thiếu đúng cái mình chưa nghĩ tới.

Hai lớp còn lại là để bắt cái mà lớp thứ nhất không bắt được:

* **Danh sách tên cấm** — bắt file *đã commit nhầm* từ trước (`git archive` sẽ
  vui vẻ đóng gói một `.env` đã lỡ commit).
* **Quét nội dung** — bắt bí mật nằm *bên trong* một file hợp lệ, kể cả bên
  trong `requests.jsonl.gz` (nó được giải nén ra để quét, không quét vỏ nén).

⚠️ Quét mẫu là lớp **cuối**, không phải lớp đầu. Nó chỉ bắt được thứ trông giống
mẫu đã biết. Bảo đảm thật nằm ở `git archive`.
"""

from __future__ import annotations

import gzip
import logging
import re
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

# ⭐ `W6-06` dời bảng nhận dạng credential xuống `rag_core`: bộ che log cần
# **cùng** tri thức này, và một bản sao thứ hai là bản sẽ không được cập nhật
# khi ai đó thêm nhà cung cấp mới — hỏng theo hướng *báo an toàn* (họ `AU-12`).
# ⚠️ `scan_text` quét TỪNG DÒNG, nên mọi luật trong bảng phải khớp trong phạm
# vi một dòng; bảng chung giữ đúng ràng buộc ấy, xem docstring bên đó.
from rag_core.credentials import SECRET_PATTERNS

__all__ = [
    "BUNDLE_EXCLUDE",
    "FORBIDDEN_NAMES",
    "REQUIRED_MEMBERS",
    "SECRET_PATTERNS",
    "BundleIncomplete",
    "BundleScan",
    "Finding",
    "build_bundle",
    "scan_bundle",
    "scan_text",
]

logger = logging.getLogger(__name__)


FORBIDDEN_NAMES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)\.env(\.|$)"),
    re.compile(r"(^|/)\.git/"),
    re.compile(r"(^|/)\.venv/"),
    re.compile(r"(^|/)\.ssh/"),
    re.compile(r"(^|/)\.aws/"),
    re.compile(r"(^|/)\.netrc$"),
    re.compile(r"(^|/)_netrc$"),
    re.compile(r"(^|/)\.git-credentials$"),
    re.compile(r"(^|/)id_(?:rsa|ed25519|ecdsa)(\.|$)"),
    re.compile(r"\.(?:pem|pfx|p12|keystore|sqlite3)$"),
)
"""Tên file cấm mang lên máy thuê. `.sqlite3` không phải bí mật — nó là cache
chunk, và nó ở đây vì gói job không có lý do gì để mang theo trạng thái cục bộ."""

_TEXT_SUFFIXES = frozenset(
    {".py", ".txt", ".md", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".cfg", ".sh", ".csv", ""}
)
_MAX_SCAN_BYTES = 400 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    member: str
    rule: str
    line: int
    excerpt: str

    def __str__(self) -> str:
        return f"{self.member}:{self.line} [{self.rule}] {self.excerpt}"


@dataclass
class BundleScan:
    members: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    scanned_bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        head = f"{len(self.members)} file · {self.scanned_bytes / 1e6:.1f} MB đã quét"
        if self.clean:
            return f"{head} · SẠCH"
        return f"{head} · {len(self.findings)} PHÁT HIỆN\n" + "\n".join(
            f"  {f}" for f in self.findings[:20]
        )


def scan_text(member: str, text: str) -> list[Finding]:
    """Quét một khối văn bản. Tách riêng để test gọi thẳng, không cần dựng tarball."""
    out: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for rule, pattern in SECRET_PATTERNS.items():
            match = pattern.search(line)
            if match:
                found = match.group(0)
                out.append(
                    Finding(
                        member=member,
                        rule=rule,
                        line=line_no,
                        # Che phần lớn giá trị: báo cáo lỗi bí mật không được
                        # trở thành chỗ thứ hai làm lộ nó.
                        excerpt=f"{found[:8]}…{len(found)} ký tự",
                    )
                )
    return out


def scan_bundle(path: Path, *, deep: bool = True) -> BundleScan:
    """Mở tarball, kiểm tên cấm, rồi quét nội dung từng file văn bản.

    `deep=False` bỏ qua phần giải nén `.gz` bên trong — 285 MB văn bản cho gói
    thật, khoảng một phút. Mặc định là **bật**: gói này rời khỏi máy, và một
    phút là cái giá rẻ nhất trong toàn bộ quy trình.
    """
    scan = BundleScan()
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            scan.members.append(member.name)

            for forbidden in FORBIDDEN_NAMES:
                if forbidden.search(member.name):
                    scan.findings.append(
                        Finding(member.name, "forbidden_name", 0, forbidden.pattern)
                    )

            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover - thư mục đã lọc ở trên
                continue
            name = Path(member.name)

            if name.suffix == ".gz":
                if not deep:
                    scan.skipped.append(member.name)
                    continue
                try:
                    payload = gzip.decompress(handle.read()).decode("utf-8", errors="replace")
                except (OSError, EOFError):
                    scan.skipped.append(member.name)
                    continue
            elif name.suffix in _TEXT_SUFFIXES:
                payload = handle.read().decode("utf-8", errors="replace")
            else:
                scan.skipped.append(member.name)
                continue

            if len(payload) > _MAX_SCAN_BYTES:  # pragma: no cover - chặn gói bất thường
                scan.skipped.append(member.name)
                continue
            scan.scanned_bytes += len(payload)
            scan.findings.extend(scan_text(member.name, payload))
    return scan


REQUIRED_MEMBERS: tuple[str, ...] = (
    "pipeline/__init__.py",
    "pipeline/indexing/__init__.py",
    "pipeline/indexing/contextualize.py",
    "packages/rag_core/chunking/contextual.py",
    "packages/rag_core/chunking/tokens.py",
    "packages/rag_core/llm/base.py",
    "packages/rag_core/llm/budget.py",
    "packages/rag_core/llm/openai_compat.py",
    "packages/rag_core/schemas.py",
    "packages/rag_core/settings.py",
    "pyproject.toml",
)
"""File mà đường `run --backend vllm` chạm tới. Thiếu một cái là pod đứng.

⚠️ `git archive HEAD` chỉ đóng gói thứ **đã commit** — đó là bảo đảm khiến `.env`
không lọt vào, và cũng là cái bẫy đối xứng: code vừa viết xong mà chưa commit
thì cũng không lọt vào. Lần dựng gói đầu tiên ở đây thiếu đúng
`contextualize.py`, tức là chính cái job. Không có phép kiểm này thì triệu chứng
là `ModuleNotFoundError` trên một cái pod đang tính tiền theo giờ.
"""


BUNDLE_EXCLUDE: tuple[re.Pattern[str], ...] = (
    re.compile(r"^tests/"),
    re.compile(r"^plans/"),
    re.compile(r"^legacy/"),
    re.compile(r"^apps/"),
    re.compile(r"^serving/"),
    re.compile(r"^infra/"),
    re.compile(r"^\.github/"),
    re.compile(r"^\.dvc/"),
    re.compile(r"^data/"),
)
"""Thứ pod không chạy tới. Cắt đi vì gói nhỏ hơn, và vì một lý do cụ thể hơn.

⭐ Lần quét đầu tiên trên gói thật báo **6 phát hiện**, cả sáu nằm trong
`tests/security/test_no_secret_in_job_bundle.py` — chính các bí mật giả mà test
cắm vào để chứng minh bộ quét hoạt động. Đó là true positive, và nó chứng minh
đường quét chạy thật trên gói thật chứ không chỉ trên đầu vào tổng hợp.

Cách chữa **không** phải là miễn trừ file đó: một cơ chế miễn trừ là thứ sau này
sẽ che mất một bí mật thật. Cách chữa là bỏ hẳn `tests/` ra khỏi gói, vì pod
chạy đúng một job và không chạy test bao giờ.

Danh sách loại trừ này an toàn theo cả hai chiều nhờ đi cặp với
`REQUIRED_MEMBERS`: loại trừ quá tay thì `BundleIncomplete` chặn ngay ở laptop,
loại trừ thiếu thì gói chỉ to hơn cần thiết.
"""


class BundleIncomplete(RuntimeError):
    """Gói thiếu file mà pod cần — gần như luôn là "chưa commit"."""


def build_bundle(out: Path, *, requests_path: Path, repo: Path | None = None) -> Path:
    """Dựng `runpod-job.tar.gz` = `git archive HEAD` + gói request + script chạy.

    Raises:
        RuntimeError: `git archive` thất bại (không phải repo, hoặc chưa commit).
        BundleIncomplete: thiếu file trong `REQUIRED_MEMBERS`.
    """
    repo = repo or Path.cwd()
    out.parent.mkdir(parents=True, exist_ok=True)
    source = out.with_name("_source.tar")
    try:
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(source), "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"git archive thất bại: {exc}") from exc

    packed: set[str] = set()
    try:
        with tarfile.open(out, "w:gz") as bundle:
            with tarfile.open(source, "r:") as archive:
                for member in archive:
                    if member.isfile() and not _skip_from_bundle(member.name):
                        handle = archive.extractfile(member)
                        if handle is not None:
                            bundle.addfile(member, handle)
                            packed.add(member.name)
            missing = [name for name in REQUIRED_MEMBERS if name not in packed]
            if missing:
                raise BundleIncomplete(
                    "Gói thiếu file mà pod cần: "
                    + ", ".join(missing)
                    + ". `git archive HEAD` chỉ đóng gói thứ đã commit — commit rồi dựng lại."
                )
            bundle.add(requests_path, arcname=f"job/{requests_path.name}")
            script = out.with_name("run_on_pod.sh")
            script.write_text(POD_SCRIPT, encoding="utf-8", newline="\n")
            bundle.add(script, arcname="job/run_on_pod.sh")
            script.unlink()
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)
    return out


def _is_forbidden(name: str) -> bool:
    return any(pattern.search(name) for pattern in FORBIDDEN_NAMES)


def _skip_from_bundle(name: str) -> bool:
    return _is_forbidden(name) or any(p.search(name) for p in BUNDLE_EXCLUDE)


POD_SCRIPT = """\
#!/usr/bin/env bash
# Chạy TRÊN POD. Không cần API key, không cần git, không cần corpus.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
REQUESTS="${REQUESTS:-job/requests.jsonl.gz}"
OUT="${OUT:-job/contexts.jsonl}"
CONCURRENCY="${CONCURRENCY:-32}"
GPU_HOURLY_USD="${GPU_HOURLY_USD:-0}"

echo "==> vLLM: $MODEL"
vllm serve "$MODEL" \\
  --max-model-len 8192 \\
  --enable-prefix-caching \\
  --gpu-memory-utilization 0.92 \\
  --disable-log-requests &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
echo "==> vLLM sẵn sàng"

python -m pipeline.indexing.contextualize run \\
  --backend vllm \\
  --model "$MODEL" \\
  --requests "$REQUESTS" \\
  --out "$OUT" \\
  --concurrency "$CONCURRENCY" \\
  --gpu-hourly-usd "$GPU_HOURLY_USD" \\
  --cost-cap 0 \\
  --report job/run-report.json

echo "==> Xong. Kéo về: $OUT + job/run-report.json"
"""
"""Script chạy trên pod. `--max-model-len 8192` vì prompt dài nhất đo được là
**5.248 token**: khai 40.960 là bắt vLLM giữ chỗ KV cache cho một độ dài không
bao giờ dùng tới, và trên 24GB thì chỗ ấy chính là thứ quyết định batch lớn hay
nhỏ. `--enable-prefix-caching` là cái mà thứ tự prompt được dựng để phục vụ —
không bật thì 31,5 triệu token tiền tố dùng chung bị prefill lại từ đầu."""
