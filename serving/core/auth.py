"""API key, principal, và chỗ **ép** tenant — `W4-04`.

## Vì sao SHA-256 chứ không bcrypt/argon2

Phản xạ "băm bí mật thì phải dùng KDF chậm" đúng với **mật khẩu**, và sai ở đây.
KDF chậm tồn tại để bù cho **entropy thấp**: người ta chọn `Hanoi2024!`, nên phải
làm mỗi lần thử tốn 100 ms. API key ở đây là **32 byte ngẫu nhiên từ
`secrets`** — không gian 2²⁵⁶, không có từ điển để duyệt, và không có bảng cầu
vồng nào dựng được.

Đổi lại, cái giá của KDF chậm ở đây là thật và trả **mỗi request**: bcrypt cost
12 là ~250 ms, tức nó sẽ là thành phần chậm nhất của toàn bộ đường `/chat` —
chậm hơn cả reranker (`W2-05` đo 524 ms cho 50 cặp, nhưng đó là *một* lần trên
GPU, còn đây là mỗi lượt gọi trên CPU của tiến trình API).

Một lượt SHA-256 cho phép tra **thẳng bằng digest** trong dict, tức O(1) và
không phải so tuần tự với từng key. Đó cũng là lý do không cần
`hmac.compare_digest` ở đây: không có phép so chuỗi nào chạy trên bí mật cả —
digest được dùng làm *khoá tra*, và đoán trúng khoá tra tức là đã đoán trúng key.

## Bí mật thô không bao giờ nằm trên đĩa

`mint()` in key ra **đúng một lần** rồi ghi lại digest. Mất thì cấp key mới, chứ
không có đường đọc lại — đó là tính chất, không phải bất tiện.

## ⭐⭐ Chỗ ép tenant mà `W2-06` không ép được

`rag_core.retrieval.filters` nói thẳng giới hạn của nó: nó không bắt được người
gọi *phải* truyền `tenant_id`, vì `rag_core` không phân biệt được "không lọc" là
đúng (eval chạy toàn corpus) hay là một lỗ rò (serving quên). Chỗ phân biệt được
là đây, nơi tenant đến từ **token đã xác thực** chứ không từ thân request.

`tenant_filter()` là hàm duy nhất được phép dựng filter tenant cho một request,
và nó **không nhận** `tenant_id` từ người gọi bên ngoài. Hai hướng hỏng của filter
tenant không đối xứng — quá chặt thì người dùng thấy và báo lại, quá lỏng thì
**dữ liệu tenant khác lọt ra và không ai thấy, kể cả người bị rò** — nên mọi thứ
ở đây nghiêng về hướng thứ nhất.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_core.retrieval.filters import MetadataFilter

__all__ = [
    "ADMIN_SCOPE",
    "ApiKeyStore",
    "CrossTenantError",
    "Principal",
    "digest_of",
    "load_entries",
    "main",
    "mint",
    "revoke",
    "tenant_filter",
]

logger = logging.getLogger(__name__)

ADMIN_SCOPE = "admin"
"""Đổi bundle đang phục vụ là quyền **khác hẳn** quyền hỏi.

Một key của tenant gọi được `POST /admin/bundle/reload` nghĩa là khách hàng đổi
được hệ thống phục vụ mọi khách hàng khác. Đó là lý do scope tồn tại ngay từ
`W4-04` thay vì để lại cho một hạng mục "phân quyền" về sau.
"""

_KEY_BYTES = 32
_KEY_PREFIX = "rag_"

_DEFAULT_STORE = "secrets/api-keys.json"
"""Một hằng cho cả ba lệnh con. `TD-58` thêm `list`/`revoke`, và ba bản sao của
cùng một đường dẫn mặc định là ba chỗ để lệch nhau — lệch ở đây nghĩa là `revoke`
báo "đã xoá" trên một file mà server không đọc."""


class CrossTenantError(PermissionError):
    """Request xin dữ liệu của tenant khác tenant trong token."""


def digest_of(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def key_hint(raw_key: str) -> str:
    """Phần key được phép ghi vào log: đủ để biết **key nào** trong lúc xoay vòng,
    không đủ để dùng lại."""
    return raw_key[: len(_KEY_PREFIX) + 6] + "…"


@dataclass(frozen=True)
class Principal:
    """Ai đang gọi. Bất biến — nó là nguồn của mọi quyết định phân quyền sau đó."""

    tenant_id: str
    key_id: str
    scopes: frozenset[str] = frozenset()
    rate_limit_per_minute: int = 60

    @property
    def is_admin(self) -> bool:
        return ADMIN_SCOPE in self.scopes


@dataclass
class ApiKeyStore:
    """Tra `Principal` từ key thô. Nội dung nạp từ JSON, khoá là **digest**."""

    by_digest: dict[str, Principal] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.by_digest)

    def lookup(self, raw_key: str) -> Principal | None:
        return self.by_digest.get(digest_of(raw_key))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ApiKeyStore:
        store: dict[str, Principal] = {}
        for entry_digest, spec in raw.items():
            store[entry_digest] = Principal(
                tenant_id=spec["tenant_id"],
                key_id=spec.get("key_id", entry_digest[:8]),
                scopes=frozenset(spec.get("scopes", ())),
                rate_limit_per_minute=int(spec.get("rate_limit_per_minute", 60)),
            )
        return cls(store)

    @classmethod
    def load(cls, path: Path | None) -> ApiKeyStore:
        """Không có file = **không có key nào**, không phải "cho qua hết".

        ⚠️ Cố ý không có key mặc định cho tiện lúc dev. Một key mặc định là
        đường ngắn nhất để credential của môi trường test đi thẳng vào
        production, và nó không để lại dấu vết nào trong diff.
        """
        if path is None or not path.is_file():
            logger.error(
                "không có kho API key (%s) — mọi route cần xác thực sẽ trả 401. "
                "Tạo bằng `python -m serving.core.auth mint --tenant <id>`",
                path,
            )
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        store = cls.from_mapping(raw)
        logger.info("nạp %d API key từ %s", len(store), path)
        return store


def mint(
    path: Path, *, tenant_id: str, scopes: Sequence[str] = (), rate_limit_per_minute: int = 60
) -> str:
    """Sinh một key mới, ghi **digest** vào `path`, trả key thô cho người gọi in ra.

    Key thô không bao giờ chạm đĩa từ hàm này. Người gọi in nó một lần; mất thì
    cấp key mới.
    """
    raw_key = _KEY_PREFIX + secrets.token_urlsafe(_KEY_BYTES)
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing[digest_of(raw_key)] = {
        "tenant_id": tenant_id,
        "key_id": f"{tenant_id}-{secrets.token_hex(3)}",
        "scopes": list(scopes),
        "rate_limit_per_minute": rate_limit_per_minute,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return raw_key


def load_entries(path: Path) -> dict[str, Any]:
    """Nội dung thô của kho khoá. `{}` nếu chưa có file."""
    if not path.is_file():
        return {}
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return raw


def revoke(path: Path, *, key_id: str) -> int:
    """Xoá mọi mục có `key_id` ấy. Trả về số mục đã xoá — `TD-58`.

    ## ⭐⭐ Thu hồi theo `key_id`, không theo digest

    Digest là thứ **không ai cầm**: người vận hành có key thô (thì đã không cần
    thu hồi) hoặc chỉ có dòng log. Và dòng log cố ý chỉ mang `key_id` — xem
    `key_hint`. Bắt họ tự băm một key để xoá nó là bắt họ đi tìm lại đúng cái bí
    mật mà quy trình này tồn tại để loại bỏ.

    ⚠️⚠️ **Thu hồi chưa có hiệu lực cho tới khi server khởi động lại.**
    `ApiKeyStore.load` chạy **một lần** lúc khởi động (`W4-04`), nên hàm này sửa
    đĩa chứ không sửa tiến trình đang chạy. Điều đó phải nằm trong lời in ra của
    CLI, không chỉ trong docstring: một người vận hành vừa xoá một khoá bị lộ và
    tin rằng mình đã xong là tình huống tệ hơn cả việc không có lệnh thu hồi.
    """
    entries = load_entries(path)
    doomed = [digest for digest, spec in entries.items() if spec.get("key_id") == key_id]
    for digest in doomed:
        del entries[digest]
    if doomed:
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(doomed)


def tenant_filter(principal: Principal, requested: MetadataFilter | None = None) -> MetadataFilter:
    """Filter cho một request, với `tenant_id` **luôn** lấy từ token.

    Ba luật, và luật giữa là luật đáng nghĩ nhất:

    1. Không truyền gì ⇒ vẫn lọc theo tenant của token. Đây là chỗ đóng đúng lỗ
       mà `W2-06` để lại: "quên lọc" không còn là một trạng thái viết ra được.
    2. ⭐ Truyền `tenant_id` **khác** ⇒ `CrossTenantError`, **không** phải lặng lẽ
       ghi đè. Ghi đè an toàn về dữ liệu nhưng biến một request sai thành một
       kết quả rỗng hợp lệ — người gọi tưởng "tenant kia không có tài liệu",
       tức chính chế độ hỏng im lặng mà `MetadataFilter` sinh ra để chặn. Và câu
       trả lời không phụ thuộc vào việc tenant kia có tồn tại hay không, nên nó
       không thành một oracle đếm tenant.
    3. Mọi field khác của người gọi giữ nguyên — chúng chỉ **thu hẹp** thêm.

    `MetadataFilter` là `frozen`, nên bản trả về không nới ra được sau khi kiểm.
    Đó chính là lý do `W2-06` đặt `frozen` ở đó.
    """
    if requested is None:
        return MetadataFilter(tenant_id=principal.tenant_id)
    if requested.tenant_id is not None and requested.tenant_id != principal.tenant_id:
        raise CrossTenantError(
            f"token thuộc tenant {principal.tenant_id!r} nhưng request xin dữ liệu của tenant khác"
        )
    return requested.model_copy(update={"tenant_id": principal.tenant_id})


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI mỏng
    parser = argparse.ArgumentParser(description="Cấp API key cho Serving Plane.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("mint", help="Sinh key mới. In ra ĐÚNG MỘT LẦN.")
    new.add_argument("--tenant", required=True)
    new.add_argument("--file", type=Path, default=Path(_DEFAULT_STORE))
    new.add_argument("--scope", action="append", default=[], help=f"vd. --scope {ADMIN_SCOPE}")
    new.add_argument("--rpm", type=int, default=60)
    new.add_argument(
        "--token-file",
        type=Path,
        default=None,
        help=(
            "Ghi key thô ra file này thay vì chỉ in ra màn hình. Dành cho tiến "
            "trình hạ tầng đọc key từ đĩa (Prometheus `credentials_file`)."
        ),
    )
    ls = sub.add_parser("list", help="Liệt kê khoá đang có. KHÔNG in được key thô.")
    ls.add_argument("--file", type=Path, default=Path(_DEFAULT_STORE))

    rm = sub.add_parser("revoke", help="Xoá một khoá theo key_id. Cần KHỞI ĐỘNG LẠI server.")
    rm.add_argument("--key-id", required=True)
    rm.add_argument("--file", type=Path, default=Path(_DEFAULT_STORE))

    args = parser.parse_args(argv)

    if args.cmd == "list":
        entries = load_entries(args.file)
        if not entries:
            print(f"kho {args.file} rỗng hoặc chưa tồn tại")
            return 0
        print(f"{len(entries)} khoá trong {args.file}\n")
        print(f"{'key_id':<20} {'tenant':<12} {'rpm':>5}  scopes")
        for digest, spec in entries.items():
            key_id = spec.get("key_id", digest[:8])
            scopes = ",".join(spec.get("scopes", ())) or "-"
            rpm = spec.get("rate_limit_per_minute", 60)
            print(f"{key_id:<20} {spec['tenant_id']:<12} {rpm:>5}  {scopes}")
        return 0

    if args.cmd == "revoke":
        removed = revoke(args.file, key_id=args.key_id)
        if removed == 0:
            print(f"không có khoá nào mang key_id {args.key_id!r} trong {args.file}")
            return 1
        print(f"đã xoá {removed} mục khỏi {args.file}")
        # ⚠️ In ra, không chỉ ghi docstring: xem `revoke`.
        print(
            "\n⚠️  CHƯA CÓ HIỆU LỰC. Kho khoá chỉ được nạp lúc khởi động, nên khoá vừa\n"
            "    xoá VẪN dùng được cho tới khi tiến trình API khởi động lại."
        )
        return 0

    raw_key = mint(
        args.file, tenant_id=args.tenant, scopes=args.scope, rate_limit_per_minute=args.rpm
    )
    print(f"tenant   : {args.tenant}")
    print(f"scopes   : {args.scope or '(không)'}")
    print(f"kho      : {args.file}")
    if args.token_file is not None:
        # ⚠️⚠️ Ghi key **thô** ra đĩa. Không tránh được: Prometheus đọc bearer
        # token từ một file (`credentials_file`), và một scraper không gõ được
        # mật khẩu. Điều kiện đi kèm là ai gọi cờ này phải biết ba điều — file
        # nằm trong thư mục **không commit**, quyền đọc nó là quyền gọi API, và
        # thu hồi nghĩa là xoá dòng tương ứng trong kho khoá **rồi khởi động
        # lại server** (kho chỉ nạp lúc khởi động, `W4-04`).
        args.token_file.parent.mkdir(parents=True, exist_ok=True)
        args.token_file.write_text(raw_key, encoding="utf-8")
        print(f"token    : đã ghi ra {args.token_file} (KHÔNG commit file này)")
    else:
        print(f"\nAPI key (chỉ hiện MỘT lần, không đọc lại được):\n\n    {raw_key}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
