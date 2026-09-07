"""Kết nối Postgres, phiên có tenant, và phép thử sẵn sàng — `W4-05`.

## ⭐⭐ "DB sẵn sàng" nghĩa là **migration đã chạy**, không phải "cắm được"

DoD của `W4-03` viết `/ready` chỉ 200 khi *"bundle + Qdrant + DB sẵn sàng"*, và
hạng mục đó để trống nhánh DB vì lúc ấy chưa có DB. Giờ có, và câu hỏi thật là
**hỏi cái gì**.

`SELECT 1` trả lời "cắm được". Đó không phải cách hệ thống này hỏng. Cách nó
hỏng là: image mới deploy xong, `alembic upgrade head` **chưa** chạy (hoặc chạy
lỗi và bị bỏ qua), pod báo sẵn sàng, nhận traffic, rồi mọi request chết bằng
`column "bundle_version" does not exist`. Một `SELECT 1` xanh suốt.

Nên phép thử so `alembic_version` trong DB với **head của thư mục migration
trong chính image này**. Lệch ⇒ chưa sẵn sàng, kèm cả hai số.

Đọc head từ `ScriptDirectory` chứ không ghim một hằng số trong mã: một hằng số
phải sửa tay mỗi lần thêm migration, và lần quên đầu tiên biến phép thử này
thành phép thử luôn xanh.

⚠️ **Nhiều head là lỗi, không phải là hai lựa chọn.** Hai nhánh migration chưa
merge nghĩa là không có câu trả lời cho "schema đúng là gì" — và nếu chọn bừa một
head thì deploy thành công trên nửa số pod.

## `SET LOCAL`, không `SET SESSION`

Policy RLS đọc `app.tenant_id` từ tham số phiên. Connection pool **tái dùng** kết
nối, nên một `SET SESSION` sót lại làm request kế tiếp mang tenant của request
trước — tức rò dữ liệu chéo tenant do một dòng cấu hình, và nó chỉ xảy ra khi có
tải (lúc pool thật sự tái dùng kết nối), tức không bao giờ xảy ra ở máy dev.
`SET LOCAL` chết cùng transaction.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from rag_core.settings import get_settings
from serving.db.models import TENANT_SETTING

__all__ = [
    "MigrationStateError",
    "async_session_factory",
    "atenant_session",
    "expected_revision",
    "make_async_engine",
    "make_engine",
    "postgres_check",
    "tenant_session",
]

logger = logging.getLogger(__name__)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"


class MigrationStateError(RuntimeError):
    """Schema trong DB không phải schema mà mã này giả định."""


@lru_cache(maxsize=1)
def expected_revision() -> str:
    """Head của thư mục migration đi kèm image này."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) != 1:
        raise MigrationStateError(
            f"thư mục migration có {len(heads)} head ({sorted(heads)}) — hai nhánh "
            "chưa merge nghĩa là không có câu trả lời cho 'schema đúng là gì'. "
            "Chạy `alembic merge` trước khi deploy."
        )
    return heads[0]


def make_engine(dsn: str | None = None, **kwargs: Any) -> Engine:
    """Engine của **ứng dụng**: role không superuser, nên RLS áp lên nó.

    ⭐ Mặc định là `postgres_app_dsn`, không phải `postgres_dsn`. Đó không phải
    một chi tiết: `POSTGRES_USER` của image postgres là superuser, và superuser
    bỏ qua RLS **hoàn toàn** — kể cả `FORCE ROW LEVEL SECURITY`. Nối bằng DSN
    kia thì năm policy của `0001_initial` trở thành trang trí và mọi phép kiểm
    cấu hình vẫn xanh.
    """
    return create_engine(dsn or get_settings().postgres_app_dsn, pool_pre_ping=True, **kwargs)


def postgres_check(engine: Engine) -> None:
    """Phép thử cho `/ready`. Ném = chưa sẵn sàng, lời của lỗi thành `detail`."""
    with engine.connect() as connection:
        found = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one_or_none()
    want = expected_revision()
    if found is None:
        raise MigrationStateError(
            f"DB chưa có migration nào; mã này cần {want}. Chạy `alembic upgrade head`."
        )
    if found != want:
        raise MigrationStateError(
            f"DB đang ở migration {found} nhưng mã này cần {want} — {_skew_hint(found, want)}"
        )


def _skew_hint(found: str, want: str) -> str:
    """⭐⭐ `found != want` phủ **hai** chiều lệch, và chúng cần hai hành động ngược nhau.

    Bản đầu của thông điệp này chỉ nói *"deploy đã lên trước khi migration
    chạy"* — tức DB **sau** mã — và khuyên `alembic upgrade head`. Nhưng chiều
    kia có thật và không hề hiếm: DB **trước** mã, vì image cũ hơn schema. Nó là
    hệ quả **bình thường** của việc rollback ứng dụng mà không rollback schema
    (`registry.py` có rollback), và của một `docker compose up` thiếu `--build`.
    Gặp cảnh ấy, "chạy `alembic upgrade head`" gửi người đọc đi đúng hướng sai:
    migration đã chạy rồi, thứ cũ là image.

    ⚠️ Tìm ra vì gặp thật (07/09/2026): DB ở `0005_message_user_link`, container
    cần `0003_message_query_plan`, và `/ready` khuyên chạy migration.

    Hỏi đồ thị revision của alembic chứ không so chuỗi: tên revision không mang
    thứ tự, nên `0005 > 0003` chỉ đúng nhờ một quy ước đặt tên mà không gì bắt
    buộc. Không xác định được chiều thì nói thẳng là không xác định được.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    script = ScriptDirectory.from_config(config)

    # ⚠️ Ba cảnh, không hai. `iterate_revisions(want, "base")` **không** nổ khi
    # `found` là revision lạ — nó chỉ duyệt tổ tiên của `want` — nên gộp "lạ" vào
    # "đi trước" là đoán bừa dưới lớp áo của một phép kiểm.
    if not any(rev.revision == found for rev in script.walk_revisions()):
        return (
            f"image này không biết revision {found} nào cả — DB đến từ một image "
            "khác (mới hơn, hoặc một nhánh migration khác). Đối chiếu phiên bản "
            "image trước, đừng chạy migration."
        )
    if found in {rev.revision for rev in script.iterate_revisions(want, "base")}:
        return "DB đi SAU mã: deploy đã lên trước khi migration chạy. Chạy `alembic upgrade head`."
    return (
        "DB đi TRƯỚC mã: image cũ hơn schema (rollback ứng dụng mà không rollback "
        "schema, hoặc `up` thiếu `--build`). Deploy lại image đúng phiên bản — "
        "`alembic upgrade head` KHÔNG sửa được cảnh này."
    )


@contextmanager
def tenant_session(factory: sessionmaker[Session], tenant_id: str) -> Iterator[Session]:
    """Phiên đã đặt tenant, nên mọi truy vấn trong đó bị RLS thu hẹp sẵn.

    ⭐ Đây là lý do lỗ tenant thứ ba (Postgres) đóng theo cùng một hướng với hai
    lỗ trước: người viết truy vấn **không cần nhớ** thêm `AND tenant_id = …`, và
    nếu họ quên thì kết quả là **rỗng**, không phải là dữ liệu của người khác.
    """
    with factory() as session:
        session.execute(
            # Tham số hoá qua `set_config` chứ không nối chuỗi vào `SET LOCAL`:
            # `SET` không nhận placeholder, nên nối chuỗi là con đường duy nhất
            # còn lại — và `tenant_id` đến từ token, nhưng "đến từ token" không
            # phải một lý do để bỏ phép tham số hoá.
            text(f"SELECT set_config('{TENANT_SETTING}', :tenant, true)"),
            {"tenant": tenant_id},
        )
        yield session


# --------------------------------------------------------------------- async
#
# `W4-06` là hạng mục đầu tiên chạm DB **trên đường request**, và đường đó là
# async. `psycopg` v3 nói được cả hai chế độ với **cùng một DSN** — đó chính là
# lý do `W4-05` chọn nó thay `asyncpg` (xem docstring `pyproject`).


def make_async_engine(dsn: str | None = None, **kwargs: Any) -> AsyncEngine:
    """Engine async của **ứng dụng** — cùng role `rag_app`, cùng lý do."""
    return create_async_engine(dsn or get_settings().postgres_app_dsn, pool_pre_ping=True, **kwargs)


def async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """⚠️ `expire_on_commit=False` là bắt buộc, không phải để cho tiện.

    Mặc định của SQLAlchemy là hết hạn mọi thuộc tính sau `commit()`, nên lần
    đọc kế tiếp là một lần `SELECT` **lười** — và ở async, lazy load ngoài phiên
    ném `MissingGreenlet`, một thông báo không nói gì về nguyên nhân. Ở đây nó
    còn tệ hơn thế: câu `SELECT` lười ấy chạy trong một transaction **mới**, tức
    một transaction **không có** `app.tenant_id`, nên RLS trả rỗng.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def atenant_session(
    factory: async_sessionmaker[AsyncSession], tenant_id: str
) -> AsyncIterator[AsyncSession]:
    """Bản async của `tenant_session`.

    ## ⭐⭐ Phiên này có giá trị đúng **một** transaction

    `set_config(..., true)` là `SET LOCAL`: nó chết khi transaction kết thúc.
    Nên `commit()` **ở giữa** block này không phải là "lưu tiến độ" — nó là
    **tháo tenant ra**. Mọi câu sau đó chạy trong một transaction mới mà
    `app.tenant_id` rỗng, và policy `tenant_id = current_setting(...)` khớp
    không hàng nào.

    Hướng hỏng lại tốt (rỗng, không phải dữ liệu người khác — đúng lý lẽ của
    `W2-06`), nhưng triệu chứng thì đánh lừa: `SELECT` trả rỗng và `INSERT` bị
    từ chối bởi `WITH CHECK`, cả hai **sau khi** một lệnh ghi đã thành công.
    Trông y hệt một bug logic ở tầng trên.

    Vì thế `W4-06` mở **hai** block ngắn quanh một stream dài thay vì một block
    dài ôm cả stream — và điều đó hoá ra cũng đúng vì một lý do thứ hai, độc
    lập: giữ một transaction mở suốt 30 giây sinh token là ghim một kết nối
    Postgres cho mỗi người dùng đang gõ chuyện.

    Có test ghim (`test_committing_midway_silently_drops_the_tenant`).
    """
    async with factory() as session:
        await session.execute(
            text(f"SELECT set_config('{TENANT_SETTING}', :tenant, true)"),
            {"tenant": tenant_id},
        )
        yield session
