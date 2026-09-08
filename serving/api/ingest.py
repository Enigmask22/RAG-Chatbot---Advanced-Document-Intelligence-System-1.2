"""Cầu nối tới API ingestion (`W3-08`) cho giao diện — `W6-01`.

## ⭐⭐ DoD của `W6-01` viết "upload progress", và hệ thống này **không có upload**

`pipeline/ingest/schemas.py:IngestRequest` nhận một **tên config**, và docstring
ở đó nói thẳng vì sao nó không nhận đường dẫn: nhận đường dẫn thì `../../.env`
hay một YAML bất kỳ trên đĩa đều đi qua được. Không có endpoint nào nhận file.

Và đó không phải một thiếu sót cần vá ở đây. Quy tắc cứng của dự án là **corpus
phải công khai, license cho phép redistribute** (`pipeline/corpus/` cưỡng chế
bằng manifest + license + DVC). Một nút "tải tài liệu lên" mở đúng con đường mà
luật ấy sinh ra để đóng: tài liệu không rõ nguồn đi thẳng vào index, rồi vào
prompt, rồi vào một câu trả lời có trích dẫn.

Nên cái dựng được **một cách trung thực** là tiến độ của *job ingest*: chạy lại
một config đã có, và xem `documents_done / documents_total` chạy. Đó là phần
"progress" thật; phần "upload" cần một quyết định về license mà không hạng mục
nào của `W6` đã lấy.

## ⭐ Vì sao **proxy**, không để trình duyệt gọi thẳng cổng 8001

Cho UI gọi thẳng nó sẽ (a) buộc mở CORS trên một dịch vụ điều khiển pipeline,
(b) buộc nó rời khỏi loopback. Đi vòng qua `/admin/ingest` thì tầng auth của
`W4-04` áp dụng nguyên vẹn — `ADMIN_PREFIX` che theo **tiền tố đường dẫn**, nên
route này được bảo vệ vì nó *ở trong* `/admin`, không vì ai đó nhớ.

⚠️ **`W6-06` cập nhật `AU-10`.** Dòng cũ ở đây viết "API ingestion không có
auth" — đúng lúc viết, sai từ `W6-06`: dịch vụ ấy giờ đòi `INGEST_API_TOKEN`
khi token được cấu hình, và **chỉ nhận loopback** khi không. Proxy này gửi
token đi (`_auth_headers`). Hai tầng vẫn cần cả hai: tầng ngoài quyết định *ai
là người dùng*, tầng trong quyết định *dịch vụ nào được gọi tôi*.

⚠️ Mặc định **tắt** (`INGEST_API_URL` rỗng ⇒ 503 kèm lời giải thích). Một bề
mặt điều khiển pipeline mở sẵn ở mọi lần deploy là thứ không ai xin.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_core.schemas import LICENSE_ALLOWLIST, DocType, Language

if TYPE_CHECKING:
    from rag_core.settings import Settings

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/ingest", tags=["ingest"])

#: Hạn giờ cho lời gọi sang dịch vụ ingest. Ngắn: nó chỉ xếp job vào hàng đợi
#: hoặc đọc một hàng Redis — cả hai đều dưới một giây khi khoẻ. Dài hơn nghĩa là
#: một dịch vụ ingest treo sẽ giữ luôn worker của API chat.
TIMEOUT_S = 5.0


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated["Settings", Depends(get_settings)]


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str = Field(min_length=1, max_length=63)
    """Tên config, **không** phải đường dẫn — `pipeline.ingest.schemas` từ chối
    mọi thứ có dấu phân cách, và giới hạn ấy phải giữ nguyên khi đi qua proxy."""

    doc_ids: tuple[Annotated[str, Field(min_length=1, max_length=200)], ...] = Field(
        default=(), max_length=1000
    )
    """⚠️ `W6-06`: có trần. Bản đầu không giới hạn gì — một thân request vài chục
    MB toàn `doc_ids` đi thẳng sang dịch vụ ingest, và dịch vụ ấy không có auth
    để tự bảo vệ (`AU-10`). 1000 là trên mức mọi lần dùng thật (`W3-07` re-index
    theo lô vài chục tài liệu) và dưới mức gây hại."""

    recreate: bool = False


MAX_UPLOAD_BYTES = 512 * 1024
"""Trần nội dung upload, byte UTF-8. ⚠️ Phải bằng đúng
`pipeline.ingest.upload.MAX_UPLOAD_BYTES` — hai plane không import được nhau
nên hằng số tồn tại hai lần, và `tests/unit/test_upload.py` ghim chúng bằng
nhau (họ quan-hệ của `NEW-13`: phép kiểm là quan hệ, không phải bản chép tay
thứ ba). Lệch chiều nào cũng tệ: proxy rộng hơn thì upload hợp lệ chết ở dịch
vụ trong với thông điệp không tới được người dùng; proxy hẹp hơn thì trần thật
không bao giờ chạm tới."""


class UploadRequest(BaseModel):
    """Bản kiểm SỚM của `pipeline.ingest.upload.UploadRequest` — `NEW-11`.

    Kiểm ở proxy để lỗi trả về là một 422 nói tiếng người (giấy phép nào được
    nhận, trần bao nhiêu byte) thay vì một 4xx đã bị `_call` che thân (`AU-03`).
    `uploaded_by` cố ý KHÔNG có ở đây: nó là danh tính đã xác thực, proxy tự
    điền từ principal — client khai nó là một trường thừa và `extra="forbid"`
    từ chối."""

    model_config = ConfigDict(extra="forbid")

    config: str = Field(min_length=1, max_length=63)
    title: str = Field(min_length=3, max_length=300)
    content: str = Field(min_length=1, max_length=MAX_UPLOAD_BYTES)
    license: str = Field(min_length=1, max_length=100)
    source_url: str = Field(min_length=1, max_length=1000)
    license_url: str = Field(default="", max_length=1000)
    lang: Language = Language.UNKNOWN
    doc_type: DocType = DocType.OTHER
    notes: str = Field(default="", max_length=500)

    @field_validator("license")
    @classmethod
    def _license_must_be_allowed(cls, value: str) -> str:
        if value not in LICENSE_ALLOWLIST:
            raise ValueError(
                f"giấy phép {value!r} không nằm trong danh sách cho phép "
                f"redistribute + phái sinh. Chỉ nhận: {sorted(LICENSE_ALLOWLIST)}"
            )
        return value

    @field_validator("content")
    @classmethod
    def _content_within_byte_budget(cls, value: str) -> str:
        size = len(value.encode("utf-8"))
        if size > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"nội dung {size} byte UTF-8, trần {MAX_UPLOAD_BYTES} byte "
                "(tiếng Việt có dấu là 2–3 byte mỗi ký tự)"
            )
        return value


def _base_url(settings: Settings) -> str:
    url = (settings.ingest_api_url or "").strip().rstrip("/")
    if not url:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "chưa cấu hình INGEST_API_URL — bảng tiến độ ingest đang tắt. "
            "Đây là mặc định có chủ đích: xem docstring serving/api/ingest.py.",
        )
    return url


def _auth_headers(settings: Settings) -> dict[str, str]:
    """Token dịch vụ, nếu có — `AU-10`. Không có thì dịch vụ đích tự giới hạn ở
    loopback, xem `pipeline.ingest.app.guard`."""
    token = settings.ingest_api_token
    return {"Authorization": f"Bearer {token.get_secret_value()}"} if token else {}


async def _call(
    method: str, url: str, payload: dict[str, Any] | None = None, *, headers: dict[str, str]
) -> Any:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.request(method, url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        # ⚠️ **Không** dội nguyên văn lỗi ra client: `AU-03` (`NEW-08`) đã vá
        # đúng chế độ này ở đường chat — thân lỗi của một dịch vụ nội bộ mang
        # host, cổng, và đôi khi cả header đi kèm.
        logger.warning("gọi dịch vụ ingest thất bại: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "không gọi được dịch vụ ingest") from exc
    if response.status_code >= 400:
        logger.warning("dịch vụ ingest trả %d: %s", response.status_code, response.text[:300])
        raise HTTPException(response.status_code, "dịch vụ ingest từ chối yêu cầu")
    return response.json()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start(body: StartRequest, settings: SettingsDep) -> Any:
    """Xếp một job re-index vào hàng đợi. Trả về `job_id` để hỏi tiến độ."""
    return await _call(
        "POST",
        f"{_base_url(settings)}/ingest",
        {"config": body.config, "doc_ids": list(body.doc_ids), "recreate": body.recreate},
        headers=_auth_headers(settings),
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload(body: UploadRequest, request: Request, settings: SettingsDep) -> Any:
    """Đăng ký một tài liệu công khai vào corpus, qua dịch vụ ingest — `NEW-11`.

    Trả biên nhận có `doc_id`; index nó bằng `POST /admin/ingest` với
    `doc_ids=[doc_id]`. Danh tính người tải đi vào manifest — sổ đăng ký corpus
    nói được *ai* đưa tài liệu này vào, đó là nửa "ai chịu trách nhiệm" của
    quyết định `NEW-11`.
    """
    principal = getattr(request.state, "principal", None)
    uploaded_by = f"{principal.tenant_id}:{principal.key_id}" if principal else ""
    payload = body.model_dump(mode="json") | {"uploaded_by": uploaded_by}
    return await _call(
        "POST",
        f"{_base_url(settings)}/upload",
        payload,
        headers=_auth_headers(settings),
    )


JOB_ID = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")
"""⭐⭐ `W6-06`: `job_id` đi thẳng vào một URL, nên nó là **đầu vào của một lời
gọi mạng**, không phải một chuỗi hiển thị.

⚠️⚠️ **Giả thuyết đầu của tôi ở đây SAI, và phép đo bác bỏ nó.** Tôi viết rằng
`%2e%2e%2f%2e%2e%2f…` cho phép đi tới đường dẫn tuỳ ý trên dịch vụ ingest. Đo
trên router thật: **không**. Mọi thứ mang `%2f` bị chặn ở tầng định tuyến và
handler không bao giờ thấy — tức traversal nhiều đoạn không tới được.

Cái **thật sự** tới được handler, và URL nó tạo ra (base `http://ingest:8001`):

| `job_id` | path đi ra | query đi ra |
|---|---|---|
| `%2e%2e` → `..` | `/` | — |
| `abc%3Fx%3D1` → `abc?x=1` | `/ingest/abc` | **`x=1`** |
| `x%00y` | *ném `httpx.InvalidURL`* | — |

Nên mức độ đúng của nó là **vừa phải, không nghiêm trọng**: lùi được **một**
đoạn đường dẫn, và tiêm được query string tuỳ ý vào một lời gọi tới dịch vụ nội
bộ. Hôm nay `GET /ingest/{job_id}` không đọc query nào nên tác hại gần 0 — nhưng
đó là một tính chất của *dịch vụ kia*, không phải một hàng rào ở đây, và nó đổi
được bất cứ lúc nào mà file này không biết.

⭐ Ca `\\x00` là một lỗi thứ hai, khác loại: `httpx.InvalidURL` **không** kế thừa
`httpx.HTTPError`, nên nó xuyên qua `except` của `_call` và thành 500. Một phép
kiểm ở đầu route đóng cả hai bằng một dòng.

⚠️ Tập ký tự là **URL-safe**, không phải hex, dù id thật hôm nay là `uuid4().hex`.
Hex chặt hơn mà **không an toàn hơn** — cả hai đều loại sạch `/ . ? # %` — đổi
lại nó ghim proxy vào định dạng id của một dịch vụ khác. Chặn đúng thứ nguy
hiểm, không chặn thêm cho có.

⚠️ Kiểm ở **proxy**, không chỉ ở dịch vụ đích: đây là chỗ duy nhất còn biết rằng
chuỗi này sắp thành URL. Dịch vụ đích nhìn thấy một đường dẫn đã bị viết lại và
không còn cách nào biết nó từng là cái gì.
"""


@router.get("/{job_id}")
async def progress(job_id: str, settings: SettingsDep) -> Any:
    """Tiến độ một job: `documents_done / documents_total`, `chunks_embedded`."""
    if not JOB_ID.match(job_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "job_id không hợp lệ")
    return await _call(
        "GET", f"{_base_url(settings)}/ingest/{job_id}", headers=_auth_headers(settings)
    )
