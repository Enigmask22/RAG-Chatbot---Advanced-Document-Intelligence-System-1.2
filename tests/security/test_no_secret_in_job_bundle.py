"""`W0-08` — không có bí mật nào rời khỏi máy này trong gói job GPU thuê.

DoD: *quét tarball, fail nếu thấy pattern API key.*

⚠️ Một bộ quét luôn trả "sạch" cũng qua được DoD viết như trên. Nên nhóm test
đầu tiên ở đây là **positive control**: cắm bí mật thật vào rồi đòi bộ quét phải
đỏ. Không có nhóm đó thì phần còn lại chỉ chứng minh rằng file không tồn tại.
"""

from __future__ import annotations

import gzip
import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from pipeline.indexing.job_bundle import (
    FORBIDDEN_NAMES,
    SECRET_PATTERNS,
    BundleIncomplete,
    build_bundle,
    scan_bundle,
    scan_text,
)

REPO = Path(__file__).resolve().parents[2]

# Chuỗi có **hình dạng** bí mật thật nhưng không phải bí mật của ai. Đủ dài để
# vượt ngưỡng của từng mẫu; nếu rút ngắn thì test tự nhiên xanh mà không ai biết.
PLANTED = {
    "openai_style_key": "DEEPSEEK_KEY = sk-a1b2c3d4e5f6a7b8c9d0e1f2",
    "hf_token": "export HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    "github_pat": "ghp_0123456789abcdefghijklmnopqrstuvwxyz",
    "aws_access_key": "aws_key: AKIAIOSFODNN7EXAMPLE",
    "private_key_block": "-----BEGIN OPENSSH PRIVATE KEY-----",
    "assigned_secret": 'password = "hunter2hunter2hunter2hunter2"',
    # `W6-06` dời bảng xuống `rag_core.credentials` và thêm hai luật: khoá của
    # CHÍNH hệ thống này, và `Bearer …` (vá `AU-09` ở tầng che log).
    "platform_api_key": "Authorization header cũ: rag_" + "0123456789abcdef" * 3,
    "bearer_token": "curl -H 'Authorization: Bearer abcdef0123456789xyz'",
}


# --------------------------------------------------------------------------
# 1. Positive control — bộ quét có thật sự bắt được gì không
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rule", sorted(SECRET_PATTERNS))
def test_every_pattern_catches_a_planted_secret(rule: str) -> None:
    """Mỗi mẫu phải có ít nhất một mẫu thử làm nó đỏ — không mẫu nào là trang trí."""
    assert rule in PLANTED, f"mẫu {rule!r} chưa có ca thử; thêm vào PLANTED"
    findings = scan_text("planted.txt", PLANTED[rule])
    assert rule in {f.rule for f in findings}


def test_finding_does_not_reprint_the_secret() -> None:
    """Báo cáo lỗi bí mật không được trở thành chỗ thứ hai làm lộ nó."""
    findings = scan_text("planted.txt", PLANTED["hf_token"])
    assert findings
    for finding in findings:
        assert "AbCdEfGhIjKlMnOpQrStUvWxYz" not in finding.excerpt
        assert "AbCdEfGhIjKlMnOpQrStUvWxYz" not in str(finding)


@pytest.mark.parametrize(
    "benign",
    [
        "DEEPSEEK_API_KEY=",  # .env.example — biến rỗng
        "api_key: str",  # chữ ký hàm
        "api_key=api_key,  # truyền xuống",
        "# sk-... là tiền tố của DeepSeek",
        "password field is redacted in repr",
    ],
)
def test_benign_source_lines_do_not_trip_the_scanner(benign: str) -> None:
    """Bộ quét kêu oan là bộ quét bị tắt sau lần thứ ba."""
    assert scan_text("ok.py", benign) == []


def test_scanner_finds_a_secret_planted_inside_a_gzip_member(tmp_path: Path) -> None:
    """⭐ Gói request là file `.gz`; quét vỏ nén thì không thấy gì bên trong.

    Ca hỏng có thật: `requests.jsonl.gz` là thứ **duy nhất** trong gói được sinh
    ra từ dữ liệu cục bộ, tức là chỗ duy nhất một bí mật có thể lọt vào mà
    `git archive` không chặn được.
    """
    bundle = tmp_path / "job.tar.gz"
    payload = gzip.compress(
        ('{"key":"a","user":"' + PLANTED["openai_style_key"] + '"}\n').encode("utf-8")
    )
    with tarfile.open(bundle, "w:gz") as tar:
        info = tarfile.TarInfo("job/requests.jsonl.gz")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    assert not scan_bundle(bundle).clean
    assert scan_bundle(bundle, deep=False).clean, "deep=False phải bỏ qua — và nói rõ là bỏ qua"
    assert "job/requests.jsonl.gz" in scan_bundle(bundle, deep=False).skipped


@pytest.mark.parametrize(
    "name",
    [".env", ".env.local", ".ssh/id_rsa", ".git-credentials", ".netrc", "server.pem"],
)
def test_forbidden_names_are_rejected_even_when_the_content_is_clean(
    tmp_path: Path, name: str
) -> None:
    bundle = tmp_path / "job.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        body = b"noi dung hoan toan vo hai\n"
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))

    scan = scan_bundle(bundle)
    assert not scan.clean
    assert {f.rule for f in scan.findings} == {"forbidden_name"}


def test_every_forbidden_pattern_has_a_case_here() -> None:
    """Thêm mẫu cấm mà quên ca thử thì mẫu ấy không bao giờ được chạy."""
    covered = {
        ".env",
        ".env.local",
        ".ssh/id_rsa",
        ".git-credentials",
        ".netrc",
        "server.pem",
        "_netrc",
        ".git/config",
        ".venv/pyvenv.cfg",
        ".aws/credentials",
        "cache.sqlite3",
    }
    for pattern in FORBIDDEN_NAMES:
        assert any(pattern.search(name) for name in covered), f"{pattern.pattern} chưa có ca thử"


# --------------------------------------------------------------------------
# 2. Gói thật, dựng bằng đúng đường sẽ dùng
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Gói dựng từ repo thật, với một gói request giả nhỏ (gói thật 8,5 MB)."""
    workspace = tmp_path_factory.mktemp("bundle")
    requests = workspace / "requests.jsonl.gz"
    requests.write_bytes(
        gzip.compress(b'{"key":"k","chunk_id":"d::0","doc_id":"d","system":"s","user":"u"}\n')
    )
    return build_bundle(workspace / "runpod-job.tar.gz", requests_path=requests, repo=REPO)


def test_real_bundle_is_clean(real_bundle: Path) -> None:
    scan = scan_bundle(real_bundle)
    assert scan.clean, scan.summary()


def test_real_bundle_carries_no_env_file_of_any_kind(real_bundle: Path) -> None:
    """`git archive` **không thể** đóng gói file bị `.gitignore` — nhưng `.env.example`
    thì đã commit, nên nó phải bị chặn bởi danh sách tên cấm chứ không phải bởi git."""
    with tarfile.open(real_bundle) as tar:
        names = tar.getnames()
    assert not [n for n in names if ".env" in n], [n for n in names if ".env" in n]


def test_real_bundle_carries_what_the_pod_actually_needs(real_bundle: Path) -> None:
    """Ngược lại của mọi test trên: gói sạch mà thiếu file thì pod đứng, tính tiền."""
    with tarfile.open(real_bundle) as tar:
        names = set(tar.getnames())
    for required in (
        "job/requests.jsonl.gz",
        "job/run_on_pod.sh",
        "pipeline/indexing/contextualize.py",
        "packages/rag_core/chunking/contextual.py",
        "packages/rag_core/llm/openai_compat.py",
        "pyproject.toml",
    ):
        assert required in names, f"thiếu {required}"


def test_pod_script_never_mentions_an_api_key_variable(real_bundle: Path) -> None:
    """Quy tắc cứng #2 ở dạng test: đường chạy trên pod không có chỗ nào nhận key."""
    with tarfile.open(real_bundle) as tar:
        handle = tar.extractfile("job/run_on_pod.sh")
        assert handle is not None
        script = handle.read().decode("utf-8")
    for forbidden in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "--backend deepseek"):
        assert forbidden not in script
    assert "--backend vllm" in script


def test_bundle_excludes_the_test_suite_that_carries_planted_secrets(real_bundle: Path) -> None:
    """⭐ Lần quét đầu trên gói thật báo 6 phát hiện — cả sáu là bí mật giả của
    chính file này. True positive, và bằng chứng bộ quét chạy trên đường thật.

    Cách chữa là bỏ `tests/` khỏi gói, **không** phải miễn trừ file này: một cơ
    chế miễn trừ là thứ về sau sẽ che mất một bí mật thật.
    """
    with tarfile.open(real_bundle) as tar:
        names = tar.getnames()
    assert not [n for n in names if n.startswith("tests/")]
    assert not [n for n in names if n.startswith("plans/")]


def test_bundle_refuses_to_build_when_a_required_file_is_missing(tmp_path: Path) -> None:
    """Bẫy đối xứng của `git archive`: code chưa commit cũng không vào gói.

    Không có phép kiểm này thì triệu chứng là `ModuleNotFoundError` trên một cái
    pod đang tính tiền theo giờ — đúng chỗ đắt nhất để phát hiện ra.
    """
    repo = tmp_path / "empty-repo"
    repo.mkdir()
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.org"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(args, cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("khong co gi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True, capture_output=True)

    requests = tmp_path / "requests.jsonl.gz"
    requests.write_bytes(gzip.compress(b'{"key":"k"}\n'))
    out = tmp_path / "job.tar.gz"

    with pytest.raises(BundleIncomplete, match=r"contextualize\.py"):
        build_bundle(out, requests_path=requests, repo=repo)
    assert not out.exists(), "gói hỏng không được để lại trên đĩa"
