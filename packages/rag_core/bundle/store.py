"""Đọc/ghi `RagBundle` trên đĩa. Ba luật, và cả ba đều là luật *từ chối*.

1. **Không ghi đè một version đã tồn tại.** "Bundle bất biến" là một câu về hệ
   thống file, không phải về schema. Nếu `v1.4.0` ghi đè được thì câu "số đo này
   thuộc về `v1.4.0`" mất nghĩa, và gate ở `W5` gác một cái tên chứ không gác
   một artifact.
2. **Không nạp bundle chưa ký hoặc sai chữ ký**, trừ khi người gọi nêu tường
   minh là muốn bỏ qua. Mặc định phải là chặt, vì đường mặc định là đường mà
   serving đi.
3. **Không nạp bundle mà tên thư mục khác `bundle_version` bên trong.** Đây là
   cách một bản rollback đi nhầm chỗ: thư mục nói `v1.3.2`, manifest nói
   `v1.4.0`, checksum khớp hoàn toàn — vì checksum bảo vệ nội dung, không bảo vệ
   chỗ đặt.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import BundleValidationError, RagBundle, parse_semver

__all__ = [
    "BUNDLE_DIR_PREFIX",
    "MANIFEST_NAME",
    "POINTER_NAME",
    "bundle_dir_name",
    "current_bundle",
    "latest_bundle",
    "list_bundles",
    "load_bundle",
    "read_pointer",
    "save_bundle",
    "write_pointer",
]

MANIFEST_NAME = "manifest.json"
BUNDLE_DIR_PREFIX = "rag-bundle-v"
POINTER_NAME = "CURRENT"
"""Tên file chứa **một** version: bundle đang được chọn để phục vụ.

## ⭐⭐ Vì sao cần một con trỏ khi đã có `latest_bundle`

`latest_bundle` trả bản semver cao nhất trong thư mục. Ba hệ quả, cả ba đều là
lỗi thật chứ không phải chuyện thẩm mỹ:

1. **`save_bundle` trở thành một lần deploy.** Đúc một release candidate để
   chạy gate — thứ mà `W5-10` làm mỗi đêm — là đủ để đổi cái mà serving nạp ở
   lần restart kế tiếp, kể cả khi bundle ấy vừa **trượt** gate.
2. **Rollback không sống qua restart.** `POST /admin/bundle` đổi bundle đang
   chạy trong bộ nhớ; lần khởi động sau `latest_bundle` lại chọn bản cao nhất
   và lặng lẽ huỷ kết quả của lần rollback ấy.
3. **Cổng PR gác một cấu hình khác cấu hình đang phục vụ** (`AU-12`):
   `pipeline/eval/smoke.py` từng hardcode `bundles/rag-bundle-v0.2.1`, còn CI
   và Makefile không truyền `--bundle`. Bump bundle mà quên sửa hằng số ấy là
   một cổng xanh chứng nhận một hệ thống không còn tồn tại.

Một con trỏ trả lời cả ba: kho artifact vẫn bất biến, phần **thay đổi được** co
lại đúng một dòng chữ, và dòng ấy nằm trong git nên mỗi lần promote/rollback là
một commit đọc được.
"""


def bundle_dir_name(version: str) -> str:
    parse_semver(version)  # từ chối sớm: tên thư mục sai không sửa được sau khi ghi
    return f"{BUNDLE_DIR_PREFIX}{version}"


def _version_from_dir(path: Path) -> str | None:
    if not path.name.startswith(BUNDLE_DIR_PREFIX):
        return None
    return path.name[len(BUNDLE_DIR_PREFIX) :]


def save_bundle(bundle: RagBundle, root: Path, *, overwrite: bool = False) -> Path:
    """Ký rồi ghi vào `root/rag-bundle-v<version>/manifest.json`.

    Ký ở đây chứ không bắt người gọi tự ký: một đường ghi mà quên ký sinh ra
    bundle không nạp được, và lỗi ấy chỉ lộ ra ở phía đọc, thường là trên máy
    khác, thường là lúc deploy.

    `overwrite` có mặt cho test và cho việc sinh lại bundle mẫu; nó **không** có
    cờ dòng lệnh tương ứng, để việc ghi đè luôn là một câu viết trong mã chứ
    không phải một phím gõ nhầm.
    """
    directory = root / bundle_dir_name(bundle.bundle_version)
    manifest = directory / MANIFEST_NAME
    if manifest.exists() and not overwrite:
        raise BundleValidationError(
            f"bundle {bundle.bundle_version} đã tồn tại: {manifest}. "
            "Bundle là bất biến — tăng version thay vì ghi đè, nếu không thì "
            "mọi số đo đã công bố cho version này không còn trỏ vào đâu cả."
        )
    directory.mkdir(parents=True, exist_ok=True)
    signed = bundle.signed()
    manifest.write_text(
        json.dumps(json.loads(signed.model_dump_json()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_bundle(path: Path, *, verify: bool = True) -> RagBundle:
    """Nạp từ file manifest hoặc từ thư mục bundle.

    `verify=False` tồn tại cho đúng một việc: chẩn đoán một bundle đã hỏng
    (`đọc được nhưng sai chữ ký` khác `không đọc nổi`). Không dùng nó ở đường
    serving.
    """
    manifest = path / MANIFEST_NAME if path.is_dir() else path
    if not manifest.is_file():
        raise FileNotFoundError(f"không thấy manifest bundle: {manifest}")

    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"manifest không phải JSON hợp lệ: {manifest} ({exc})") from exc

    bundle = RagBundle.model_validate(raw)

    declared = _version_from_dir(manifest.parent)
    if declared is not None and declared != bundle.bundle_version:
        raise BundleValidationError(
            f"thư mục nói version {declared!r} nhưng manifest nói "
            f"{bundle.bundle_version!r}: {manifest}. Checksum bảo vệ *nội dung*, "
            "không bảo vệ *chỗ đặt* — nên phép kiểm này phải nằm ngoài checksum."
        )

    if verify:
        # ⭐ Truyền `raw` chứ không để nó băm lại model đã validate: đó là toàn bộ
        # cách giải `TD-36`. Xem docstring của `RagBundle.verify_checksum`.
        bundle.verify_checksum(raw)
    return bundle


def list_bundles(root: Path, *, verify: bool = True) -> list[RagBundle]:
    """Mọi bundle trong `root`, **sắp theo thứ tự semver** chứ không theo tên file.

    Sắp theo tên thì `v1.10.0` đứng trước `v1.9.0`, và "bản trước đó" — thứ mà
    rollback cần — trỏ vào nhầm bundle. Lỗi này chỉ xuất hiện ở lần release thứ
    mười, tức lâu sau khi mọi test thủ công đã thôi được chạy.
    """
    if not root.is_dir():
        return []
    found = [
        load_bundle(child, verify=verify)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / MANIFEST_NAME).is_file()
    ]
    return sorted(found, key=lambda item: item.version_key)


def latest_bundle(root: Path, *, verify: bool = True) -> RagBundle | None:
    found = list_bundles(root, verify=verify)
    return found[-1] if found else None


def read_pointer(root: Path) -> str | None:
    """Version mà `root/CURRENT` trỏ tới, hoặc `None` nếu chưa có con trỏ.

    `None` khác `""`: chưa có con trỏ là trạng thái hợp lệ của một checkout mới
    hoặc một thư mục tạm trong test. Con trỏ **rỗng** thì không — nó là một
    lần ghi hỏng, và im lặng coi nó như "chưa có" sẽ đẩy hệ thống về đúng cái
    hành vi `latest_bundle` mà con trỏ sinh ra để thay.
    """
    pointer = root / POINTER_NAME
    if not pointer.is_file():
        return None
    version = pointer.read_text(encoding="utf-8").strip()
    if not version:
        raise BundleValidationError(
            f"con trỏ {pointer} rỗng. Một file trống không phải 'chưa chọn' — "
            "xoá hẳn file nếu thật sự muốn quay về suy luận theo semver."
        )
    parse_semver(version)  # con trỏ trỏ vào một cái tên sai thì hỏng ngay ở đây
    return version


def write_pointer(root: Path, version: str) -> Path:
    """Trỏ `CURRENT` sang `version`. Từ chối nếu bundle ấy không nạp được.

    Kiểm **trước khi** ghi, và kiểm bằng đúng đường mà serving đi (`load_bundle`
    với `verify=True`). Một con trỏ trỏ vào chỗ trống hay vào một manifest sai
    chữ ký là cách biến một lần promote thành một sự cố lúc khởi động — và lúc
    ấy thông tin duy nhất còn lại là một tiến trình không lên được.
    """
    load_bundle(root / bundle_dir_name(version))
    pointer = root / POINTER_NAME
    pointer.write_text(version + "\n", encoding="utf-8")
    return pointer


def current_bundle(root: Path, *, verify: bool = True) -> RagBundle | None:
    """Bundle mà con trỏ chọn. `None` nếu **chưa có con trỏ** — không tự đoán.

    Cố ý không fallback về `latest_bundle` ở đây: hàm này trả lời câu hỏi *"đã
    chọn cái nào chưa"*, và trộn nó với *"đoán xem cái nào"* là cách con trỏ mất
    hết tác dụng ngay lần đầu ai đó quên tạo nó. Chỗ nào cần đoán thì phải viết
    ra là mình đang đoán — xem `serving.api.app._startup_version`.
    """
    version = read_pointer(root)
    if version is None:
        return None
    return load_bundle(root / bundle_dir_name(version), verify=verify)
