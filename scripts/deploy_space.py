"""Lắp và đẩy HF Space của `W6-02`.

Space là một repo thứ hai, và mọi repo thứ hai là một chỗ để hai bản sao lệch
nhau. Script này tồn tại để **không có bước nào làm bằng tay**: nó lắp thư mục
gửi đi từ nguồn trong repo này, điền SHA, và từ chối chạy khi một trong năm điều
kiện dưới đây sai.

## Năm cửa, và bốn cửa đầu là lỗi đã tưởng tượng được ra hậu quả

1. **Cây làm việc sạch.** SHA ghim trong `requirements.txt` trỏ về một commit;
   nếu còn thay đổi chưa commit thì SHA ấy mô tả một hệ thống khác hệ thống vừa
   thử.
2. **⭐ SHA đã có trên GitHub.** Đây là cửa dễ quên nhất và hậu quả của nó chỉ
   lộ ra trong log build của Space: `pip install git+…@<sha>` với một commit mới
   chỉ có ở máy sẽ hỏng ở phút thứ 5 của một lần build vài GB.
3. **⭐⭐ Số point của index khớp `n_chunks` của manifest.** Nếu lệch,
   `QdrantRuntimeBuilder._check_size` sẽ giết Space lúc khởi động — đúng như nó
   nên làm, nhưng ta biết được điều đó ở đây trong hai giây thay vì ở đó sau
   một lần upload 239 MB.
4. **Manifest gửi đi đúng bản `bundles/CURRENT` trỏ tới.** Gửi một bundle khác
   con trỏ là dựng lại đúng lỗi `NEW-13` ở một repo thứ hai.
5. **⚠️ Frontmatter hợp lệ với Hub.** Cửa này được thêm **sau khi bốn cửa kia
   cho qua một lượt deploy hỏng**: `short_description` dài 69 ký tự, Hub chặn ở
   60, và lỗi trả về là một `BadRequestError` lồng ba tầng traceback từ
   `/api/validate-yaml`. Bốn cửa đầu đều hỏi *"thứ tôi gửi có đúng thứ tôi đã
   đo không"* — không cửa nào hỏi *"phía kia có nhận không"*. Ràng buộc thì
   thuộc về Hub, nhưng **thời điểm** biết được nó thì thuộc về ta.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPACE_SRC = REPO / "space"
GITHUB = "https://github.com/Enigmask22/RAG-Chatbot"
SHA_PLACEHOLDER = "__GIT_SHA__"

#: Chỉ những file này rời khỏi `space/` — danh sách trắng, không phải danh sách
#: đen. Một `ignore_patterns` bỏ sót nghĩa là một file lọt lên chỗ công khai.
APP_FILES = ("app.py", "guard.py", "zerogpu.py", "README.md", "requirements.txt")

#: Hub từ chối `short_description` dài hơn ngần này (đo được 07/09/2026 bằng
#: một lượt deploy đỏ). Ghim ở đây, không gõ lại trong test.
SHORT_DESCRIPTION_MAX = 60


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _gate_clean() -> None:
    if _git("status", "--porcelain"):
        raise SystemExit("cây làm việc còn thay đổi chưa commit — cửa 1")


def _gate_pushed(sha: str) -> None:
    import urllib.error
    import urllib.request

    url = f"https://api.github.com/repos/Enigmask22/RAG-Chatbot/commits/{sha}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            if resp.status != 200:
                raise SystemExit(f"GitHub trả {resp.status} cho {sha} — cửa 2")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"commit {sha[:10]} chưa có trên GitHub ({exc.code}) — đẩy trước đã. Cửa 2"
        ) from exc


def _gate_index(index_dir: Path, manifest: dict) -> int:
    from qdrant_client import QdrantClient

    collection = manifest["components"]["index"]["collection"]
    want = int(manifest["components"]["index"]["n_chunks"])
    client = QdrantClient(path=str(index_dir))
    try:
        got = int(client.count(collection, exact=True).count)
    finally:
        client.close()
    if got != want:
        raise SystemExit(f"index có {got} point nhưng manifest khai {want} — cửa 3")
    return got


def _gate_frontmatter() -> None:
    """Cửa 5 — xem docstring module.

    Cố ý **không** gọi `HfApi._validate_yaml`: nó cần mạng và một token, tức
    biến một phép kiểm hai mili giây thành một phụ thuộc. Ràng buộc duy nhất
    từng chặn ta là độ dài, nên đó là thứ được ghim; nếu Hub thêm luật mới thì
    cửa này im lặng và ta lại biết qua một lượt đỏ — chấp nhận, và nói ra.
    """
    text = (SPACE_SRC / "README.md").read_text(encoding="utf-8")
    match = re.search(r"^short_description:[ 	]*(.+?)[ 	]*$", text, re.MULTILINE)
    if match is None:
        return
    mo_ta = match.group(1)
    if len(mo_ta) > SHORT_DESCRIPTION_MAX:
        raise SystemExit(
            f"short_description dài {len(mo_ta)} ký tự, Hub chặn ở "
            f"{SHORT_DESCRIPTION_MAX} — cửa 5\n  {mo_ta}"
        )


def _stage(dest: Path, sha: str, index_dir: Path, version: str) -> None:
    for name in APP_FILES:
        shutil.copy2(SPACE_SRC / name, dest / name)

    req = dest / "requirements.txt"
    text = req.read_text(encoding="utf-8")
    if SHA_PLACEHOLDER not in text:
        raise SystemExit(f"requirements.txt không còn chỗ điền {SHA_PLACEHOLDER}")
    req.write_text(text.replace(SHA_PLACEHOLDER, sha), encoding="utf-8")

    bundles = dest / "bundles"
    (bundles / f"rag-bundle-v{version}").mkdir(parents=True)
    shutil.copy2(
        REPO / "bundles" / f"rag-bundle-v{version}" / "manifest.json",
        bundles / f"rag-bundle-v{version}" / "manifest.json",
    )
    (bundles / "CURRENT").write_text(f"{version}\n", encoding="utf-8")

    target = dest / "index"
    shutil.copytree(index_dir, target, ignore=shutil.ignore_patterns(".lock"))

    # `.gitattributes`: sqlite của local mode là 239 MB, phải đi qua LFS.
    (dest / ".gitattributes").write_text(
        "*.sqlite filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="johnenigmask12/rag-platform-demo")
    ap.add_argument("--index", type=Path, default=REPO / "dist" / "space-index")
    ap.add_argument("--private", action="store_true", help="tạo Space riêng tư")
    ap.add_argument("--dry-run", action="store_true", help="chỉ lắp và in ra, không đẩy")
    ap.add_argument("--message", default="deploy W6-02")
    args = ap.parse_args()

    from rag_core.bundle.store import read_pointer

    version = read_pointer(REPO / "bundles")
    if not version:
        raise SystemExit("bundles/CURRENT không trỏ vào đâu — cửa 4")
    manifest = json.loads(
        (REPO / "bundles" / f"rag-bundle-v{version}" / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["bundle_version"] != version:
        raise SystemExit(
            f"CURRENT nói {version} nhưng manifest khai {manifest['bundle_version']} — cửa 4"
        )

    if not args.dry_run:
        _gate_clean()
    sha = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _gate_pushed(sha)
    _gate_frontmatter()
    points = _gate_index(args.index, manifest)

    print(f"bundle {version} · {points} point · commit {sha[:10]} · {GITHUB}")

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "space"
        dest.mkdir()
        _stage(dest, sha, args.index, version)
        total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print(
            f"đã lắp {sum(1 for _ in dest.rglob('*') if _.is_file())} file, "
            f"{total / 1024 / 1024:.1f} MB"
        )
        for f in sorted(dest.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(dest)}  {f.stat().st_size / 1024:.0f} KB")
        if args.dry_run:
            print("\n--dry-run: dừng ở đây, không đẩy")
            return 0

        from huggingface_hub import HfApi, SpaceHardware

        api = HfApi(token=_token())
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=SpaceHardware.ZERO_A10G,
            private=args.private,
            exist_ok=True,
        )
        url = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(dest),
            commit_message=f"{args.message} — repo @ {sha[:10]}",
        )
        print(f"\nxong: {url}")
        print(f"Space: https://huggingface.co/spaces/{args.repo_id}")
    return 0


def _token() -> str:
    """Token HF của người dùng.

    ⚠️ `HF_HOME` trên máy này trỏ sang ổ D, nên `get_token()` không thấy token
    nằm ở `~/.cache/huggingface/token`. Đọc cả hai chỗ thay vì bắt người dùng
    đăng nhập lại vào một cache thứ hai.
    """
    from huggingface_hub import get_token

    token = get_token()
    if token:
        return token
    fallback = Path.home() / ".cache" / "huggingface" / "token"
    if fallback.is_file():
        value = fallback.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise SystemExit("không tìm thấy token HF — chạy `hf auth login`")


if __name__ == "__main__":
    sys.exit(main())
