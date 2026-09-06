"""Nhận dạng credential trong văn bản — **một** bảng, hai người dùng — `W6-06`.

## ⭐⭐ Vì sao bảng này rời `pipeline/indexing/job_bundle.py` xuống đây

Nó ra đời ở `W3` cho đúng một việc: quét gói job trước khi gói ấy rời máy lên
GPU thuê. `W6-06` đo ra người dùng thứ hai, và người thứ hai cần **cùng** tri
thức ấy ở một tầng thấp hơn hẳn — bộ che log của `rag_core.generation`.

Để bảng ở `pipeline` rồi cho `rag_core` import ngược lên là đảo chiều phụ thuộc.
Chép sang một bản thứ hai thì hai bản lệch nhau vào ngày ai đó thêm một nhà cung
cấp mới vào đúng **một** bên — chính họ lỗi `AU-12` (một sự thật, hai bản sao),
và ở đây bản im lặng hơn: bản không được cập nhật vẫn chạy, vẫn xanh, chỉ là
không bắt được thứ nó sinh ra để bắt.

## ⭐⭐ Hai người dùng, hai hợp đồng, và chỗ chúng khác nhau

* **Quét gói** (`scan_text`) đi **từng dòng** để báo được số dòng. Nên mọi luật
  trong `SECRET_PATTERNS` phải khớp **trong phạm vi một dòng** — một luật trải
  nhiều dòng sẽ không bao giờ khớp ở đó, và nó hỏng theo hướng *báo an toàn*.
* **Che log** (`scrub_credentials`) thay giá trị bằng placeholder, và nó muốn
  giữ lại phần **nhãn**: một dòng log `Authorization: [credential]` còn đọc
  được, còn `[credential]` trơ trọi thì mất luôn thông tin header nào đã bị che.
  Đó là lý do vài luật có nhóm `(?P<keep>…)`.

⚠️ Khối PEM là chỗ hai hợp đồng **không** gộp được: luật chung chỉ khớp dòng
`-----BEGIN …-----`, đủ để *báo* nhưng không đủ để *che* (thân khoá nằm ở những
dòng sau). Nên có thêm đúng một luật chỉ dành cho bộ che — khai riêng, không
nhét vào bảng chung, vì nhét vào là làm `scan_text` mang một luật không bao giờ
khớp.
"""

from __future__ import annotations

import re

__all__ = [
    "CREDENTIAL_PLACEHOLDER",
    "SECRET_PATTERNS",
    "scrub_credentials",
]

CREDENTIAL_PLACEHOLDER = "[credential]"

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    # ⚠️⚠️ **Thứ tự có nghĩa, và một phép tiêm lỗi đã dạy tôi điều đó.**
    #
    # Hai luật mang nhóm `keep` (giữ lại nhãn) phải chạy **trước** các luật nhận
    # dạng token trần. Bản đầu xếp ngược: trên `Authorization: Bearer rag_…`,
    # `platform_api_key` khớp phần `rag_…` trước và trả về
    # `Authorization: Bearer [credential]` — đúng kết quả mong muốn, nhưng
    # **không phải nhờ `keep`**. Cơ chế `keep` khi ấy là mã chết cho ca phổ
    # biến nhất của nó, và bài test "nhãn còn nguyên" đi qua vì một lý do khác
    # hẳn lý do nó tin. Phép tiêm xoá `keep` **sống sót** chính vì thế.
    #
    # Xếp lại thì một dòng log chỉ có đúng một cách được che, và cách ấy là
    # cách đã viết ra.
    "bearer_token": re.compile(r"(?i)(?P<keep>\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}"),
    # ⭐⭐ `AU-09`. Audit đề xuất chà `Bearer` ở **một** chỗ gọi
    # (`openai_compat._raw_events`). Số đo của `W6-06` nói lỗ ở tầng thấp hơn:
    # bộ che log không có luật nào cho bí mật cả, nên vá một chỗ gọi là đóng
    # một cửa trong nhiều cửa cùng mở. Luật ở đây phủ mọi dòng log, kể cả của
    # thư viện bên thứ ba — cùng lý lẽ đã đặt `RedactingFilter` lên handler.
    #
    # Gán tường minh với giá trị đủ dài để không dính vào `KEY=` rỗng của
    # `.env.example` hay `api_key=api_key` trong code.
    "assigned_secret": re.compile(
        r"(?i)(?P<keep>\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key"
        r"|password)\s*[:=]\s*[\"']?)[A-Za-z0-9/_+=-]{20,}"
    ),
    # DeepSeek + OpenAI + OpenRouter đều dùng tiền tố `sk-`.
    "openai_style_key": re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{16,}"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "github_pat": re.compile(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    # ⭐ `W6-06`: khoá của **chính hệ thống này**. `serving.core.auth` cấp key
    # dạng `rag_` + `secrets.token_urlsafe(32)` (43 ký tự). Bảng cũ không có nó
    # — nó chỉ biết khoá của người khác, không biết khoá của mình.
    "platform_api_key": re.compile(r"\brag_[A-Za-z0-9_-]{40,}"),
}

_PEM_BODY = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----"
)
"""Chỉ dành cho `scrub_credentials` — xem ⚠️ ở docstring module."""


def _replace(match: re.Match[str]) -> str:
    """Giữ nhãn (nếu luật khai `keep`), thay phần giá trị."""
    keep = match.groupdict().get("keep") or ""
    return keep + CREDENTIAL_PLACEHOLDER


def scrub_credentials(text: str) -> str:
    """Thay mọi credential nhận ra được bằng `[credential]`.

    ⚠️ Đây là lớp **cuối**, không phải lớp đầu. Nó bắt được cái nó biết mặt; một
    nhà cung cấp có định dạng key lạ đi qua nguyên vẹn. Lớp đầu vẫn là "đừng ghi
    bí mật vào log" — chỉ là lớp đầu ấy phụ thuộc vào việc nhớ, và mọi thứ phụ
    thuộc vào việc nhớ đều hỏng vào ngày người ta quên.
    """
    out = _PEM_BODY.sub(CREDENTIAL_PLACEHOLDER, text)
    for pattern in SECRET_PATTERNS.values():
        out = pattern.sub(_replace, out)
    return out
