"""Cấu hình đọc từ biến môi trường / `.env`.

Nguyên tắc: **fail-fast và nói rõ thiếu biến nào.** Cấu hình sai mà chương trình
vẫn chạy được rồi hỏng ở giữa job index 2 tiếng là kiểu lỗi đắt nhất.

Secret không có giá trị mặc định và không bao giờ lọt vào `repr` — dùng
`SecretStr` để lỡ log nguyên object settings cũng không rò key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ LLM API
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    glm_api_key: SecretStr | None = None
    glm_base_url: str = "https://api.z.ai/api/paas/v4"
    """Endpoint quoc te cua Z.ai. Ban dai luc la `https://open.bigmodel.cn/api/paas/v4`."""
    hf_token: SecretStr | None = None

    # ------------------------------------------------------------ Hạ tầng
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rag_chunks"

    postgres_user: str = "rag"
    postgres_password: SecretStr = SecretStr("rag_local_dev_only")
    postgres_db: str = "rag"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    postgres_app_user: str = "rag_app"
    postgres_app_password: SecretStr = SecretStr("rag_app_local_dev_only")
    """⭐⭐ Role mà **ứng dụng** dùng, tách khỏi role chạy migration (`W4-05`).

    Không phải để cho gọn. `POSTGRES_USER` của image postgres là **superuser**, và
    superuser **bỏ qua Row-Level Security hoàn toàn** — kể cả `FORCE ROW LEVEL
    SECURITY`. Nên nếu ứng dụng dùng chung role ấy thì mọi policy trở thành trang
    trí, trong khi `pg_class.relforcerowsecurity` vẫn báo `true` và mọi phép kiểm
    cấu hình vẫn xanh.

    Phát hiện được vì test **hành vi** đỏ trong khi test **cấu hình** xanh —
    xem `tests/integration/test_migrations.py` §2.
    """

    redis_url: str = "redis://127.0.0.1:6379/0"

    ingest_queue: str = "arq:queue"
    """Tên hàng đợi arq cho job ingest.

    Cấu hình được vì hai môi trường dùng chung một Redis sẽ **nhặt job của nhau**:
    worker của staging nhận job của production, thất bại vì thiếu config, và job
    biến mất khỏi hàng đợi. Đo được ở chính test của `W3-08` — các test dùng chung
    một hàng đợi đã làm đúng chuyện đó với nhau.
    """

    ingest_config_dir: Path = Path("configs/indexing")
    """Thư mục mà `POST /ingest` được phép đọc config từ đó (`W3-08`).

    Là **cận trên của quyền đọc**, không phải một tiện ích: `IngestRequest.config`
    nhận một *tên*, và `pipeline.ingest.schemas.resolve_config` ghép nó vào đúng
    thư mục này rồi kiểm lại rằng kết quả không thoát ra ngoài. Đổi được bằng biến
    môi trường vì container có thể mount config ở chỗ khác — và vì test cần một
    thư mục tạm mà không phải nới lỏng hàng rào.
    """

    # ------------------------------------------------------------ Serving
    bundle_root: Path = Path("bundles")
    """Thư mục chứa các `rag-bundle-v*/` mà Serving Plane được nạp từ đó (`W4-03`).

    Là **toàn bộ** đường nối giữa hai plane: pipeline ghi vào đây, serving đọc từ
    đây, và không có kênh nào khác. Cấu hình được vì trong container nó là một
    volume mount, còn trong test nó là `tmp_path`.
    """

    bundle_version: str | None = None
    """Bundle kích hoạt lúc khởi động. `None` = bản semver cao nhất tìm thấy.

    Ghim tường minh là cách duy nhất để một lần rollback **sống sót qua restart**:
    `BundleRegistry.rollback()` chỉ đổi trạng thái trong bộ nhớ, nên nếu deploy
    vẫn để `None` thì container khởi động lại sẽ lặng lẽ quay về đúng bản vừa bị
    lùi khỏi.
    """

    ingest_api_url: str | None = None
    """`W6-01`: URL của API ingestion (`W3-08`) để bảng tiến độ trong UI hỏi.

    **Mặc định tắt.** `serving/api/ingest.py` đi vòng qua `/admin/ingest` để tầng
    auth của `W4-04` áp dụng được. Một bề mặt điều khiển pipeline mở sẵn ở mọi
    lần deploy là thứ không ai xin — nên phải bật tường minh.
    """

    ingest_api_token: SecretStr | None = None
    """`W6-06` / `AU-10`: bí mật dùng chung giữa proxy `/admin/ingest` và dịch vụ
    ingestion.

    ⭐ Một **service token**, không phải một kho khoá đa tenant: dịch vụ ấy có
    đúng một client hợp lệ (proxy của Serving Plane), và dựng một `ApiKeyStore`
    thứ hai cho một client là thêm một chỗ để xoay vòng, một chỗ để quên.

    ⚠️ Không đặt ⇒ dịch vụ ingestion **chỉ nhận lời gọi loopback** (xem
    `pipeline.ingest.app.guard`). Đó là hành vi mặc định an toàn cho máy dev; một
    triển khai có hai container thì phải đặt token, vì khi đó client không còn
    là loopback nữa.
    """

    bundle_warmup: bool = True
    """Chạy một lượt truy hồi giả ngay sau khi kích hoạt bundle (`TD-72`).

    `W6-05` đo trên hệ đang phục vụ: request đầu tiên sau khi tiến trình lên tốn
    **13.386 ms** so với 4.272 ms của lượt nóng (TTFT 10.469 vs 1.389 ms), và
    toàn bộ phần phạt nằm ở `prepare` — kernel CUDA của cross-encoder chỉ khởi
    tạo ở lời gọi `score()` đầu tiên, sau khi `/ready` đã xanh.

    Bật (mặc định) chuyển 9 giây ấy từ **người dùng đầu tiên** sang **quy trình
    deploy**. Tắt khi thời gian container lên là ràng buộc chặt hơn độ trễ của
    lượt đầu. Xem `BundleRegistry._warm`.
    """

    bundle_allow_runtime_drift: bool = False
    """Cho chạy bundle mà runtime không khớp `components.retriever_name` (`TD-38`).

    Chỉ dành cho máy dev không có GPU muốn chạy thử bundle đã eval trên
    `cuda:float16`. Bật ở production nghĩa là mọi số trong manifest có thể đang
    nói về một hệ thống khác hệ thống đang phục vụ.
    """

    api_keys_file: Path | None = Path("secrets/api-keys.json")
    """Kho API key của Serving Plane (`W4-04`) — chứa **digest**, không chứa key.

    Thiếu file = **không có key nào**, tức mọi route cần xác thực trả 401. Cố ý
    không có key mặc định: một key mặc định cho tiện lúc dev là đường ngắn nhất
    để credential của môi trường test đi thẳng vào production, và nó không để
    lại dấu vết nào trong diff.
    """

    # ------------------------------------------------------------ Chat (W4-06)
    chat_provider: Literal["deepseek", "glm", "none"] = "deepseek"
    """Nguồn sinh text của `POST /chat`. `none` = tắt hẳn đường sinh.

    ⚠️ `none` **không** làm `/ready` đỏ. Một phép thử sẵn sàng gọi API trả tiền
    là một phép thử tự sinh hoá đơn, nhân với số replica nhân với tần suất poll.
    Thiếu key biểu hiện ở `POST /chat` bằng `503` kèm lý do, và ở một dòng cảnh
    báo lúc khởi động. `W4-08` sẽ có circuit breaker để nói được nhiều hơn.
    """

    chat_model: str | None = None
    """`None` = mặc định của provider (`DEFAULT_DEEPSEEK_MODEL` / `DEFAULT_GLM_MODEL`)."""

    chat_top_k: int = Field(default=5, ge=1, le=50)
    chat_max_tokens: int = Field(default=1024, ge=1)
    # `W4-10` — semantic cache. Ngưỡng 0,96 đo từ `probes/w4-10-cosine-threshold.json`:
    # bẫy "gần giống nhưng đổi đáp án" cao nhất là 0,9410, paraphrase thật vượt
    # 0,96 chỉ ~2/10 — cache này bảo thủ có chủ đích.
    chat_cache: bool = True
    chat_cache_threshold: float = Field(default=0.96, ge=0.0, le=1.0)
    chat_cache_ttl_s: int = Field(default=86_400, ge=1)
    chat_cache_max_entries: int = Field(default=128, ge=1)

    # `NEW-10` — gộp request TRÙNG NHAU đang bay. `AU-11` đo: 8 câu giống hệt
    # gửi đồng thời ⇒ 8 lời gọi trả tiền, 0 cache hit.
    chat_single_flight: bool = True
    chat_single_flight_wait_s: float = Field(default=15.0, gt=0.0)
    """Hạn giờ người theo sau chờ leader. Lớn hơn p99 end-to-end (`exp-003`:
    11.142 ms) một biên.

    ⚠️ Đây **không** phải một van an toàn rẻ tiền: quá hạn nghĩa là follower đã
    tiêu ngần này giây **rồi mới** bắt đầu tự làm. Nó là hàng rào cho ca leader
    biến mất mà không chạy `finally` (tiến trình bị giết) — ca duy nhất mà
    đường `resolve()` không phủ."""

    # ------------------------------------------------------- LLM Router (W4-08)
    chat_fallback_provider: Literal["openrouter", "glm", "none"] = "none"
    """Nhánh dự phòng khi nhà cung cấp chính hỏng. `none` = chỉ một nhánh.

    ⚠️ Mặc định là `none` **có chủ đích**: một fallback chưa từng được gọi lần
    nào là một fallback chưa biết có chạy không, và bật nó im lặng nghĩa là lần
    đầu nó chạy sẽ là lúc nhà chính đang chết — chỗ tệ nhất để phát hiện ra sai
    key hoặc sai slug. Bật nó là một quyết định, và `GET /admin/llm` cho biết
    nó đã phục vụ bao nhiêu request.
    """

    chat_fallback_model: str | None = None
    """⚠️ Với OpenRouter phải là **slug tường minh** (`deepseek/deepseek-chat`).

    Quy tắc cứng #1 được ép bằng mã ở `OpenAICompatProvider.__init__`: một model
    bắt đầu bằng `@preset/` bị từ chối ngay lúc dựng, không phải lúc gọi.
    """

    chat_daily_budget_usd: float = Field(default=1.0, ge=0.0)
    """Trần chi phí sinh text mỗi ngày (UTC). `0` = **không trần**, phải khai rõ.

    ⚠️ Đếm **trong tiến trình** — cùng giới hạn với hạn mức nhịp của `W4-04`
    (`TD-39`): N replica ⇒ trần thật là N×, và restart đưa bộ đếm về 0. Nó chặn
    được ca nó sinh ra để chặn (một vòng lặp hỏng đốt sạch ngân sách trong mười
    phút) nhưng không phải một trần đúng nghĩa cho nhiều tiến trình.
    """

    chat_breaker_threshold: int = Field(default=3, ge=1)
    chat_breaker_cooldown_s: float = Field(default=30.0, gt=0.0)

    chat_max_retries: int = Field(default=1, ge=0)
    """Số lần thử lại **bên trong một** lời gọi, cho đường serving.

    Mặc định của `OpenAICompatProvider` là 4 với backoff tới 30 s mỗi lần — đúng
    cho một job offline nghìn lời gọi, **sai** cho một request có người đang
    ngồi đợi: một lời gọi hỏng có thể mất hơn một phút trước khi router kịp
    chuyển sang nhánh dự phòng. Ở đây thử lại nhanh rồi nhường cho fallback.
    """

    # ------------------------------------------------- Query understanding (W4-07)
    chat_rewrite: bool = True
    """Bật/tắt **viết lại câu hỏi đa lượt** — và chỉ nó.

    Đây là việc duy nhất trong `W4-07` tốn tiền và cộng vào TTFB, nên nó là việc
    duy nhất có công tắc. Định tuyến (`NO_RETRIEVAL`/`CLARIFY`) và phát hiện ngôn
    ngữ là luật thuần, tất định, miễn phí — cho chúng một công tắc chỉ tạo thêm
    một cấu hình mà ở đó hệ thống chạy tệ hơn không vì lý do nào.

    Tắt nó **không** làm hỏng lượt follow-up: câu gốc vẫn đi truy hồi, đúng hành
    vi của `W4-06`.
    """

    chat_rewrite_timeout_s: float = Field(default=6.0, gt=0.0, le=30.0)
    """Bước viết lại nằm **trước** truy hồi, nên nó cộng thẳng vào TTFB. Quá hạn
    thì lượt đi tiếp bằng câu gốc — mất một cải thiện, không mất câu trả lời."""

    # ------------------------------------------------------- Quan sát (W5-06)
    langfuse_host: str = "http://127.0.0.1:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr | None = None
    """Quan sát bật **chỉ** khi có đủ host + cả hai khoá. Không có cờ bật/tắt
    riêng: một cờ bên cạnh một cặp khoá cho hai trạng thái mâu thuẫn, và cái
    người ta gặp thật là "bật nhưng chưa có khoá" — mọi thứ trông như đã chạy,
    không trace nào tới nơi. Xem `serving.core.langfuse.build_sink`."""

    langfuse_queue_size: int = Field(default=256, ge=1)
    """Trần hàng đợi đẩy trace. Đầy thì trace mới bị **vứt** kèm bộ đếm, chứ
    không xếp hàng vô hạn: một endpoint chết vì bộ đếm span của chính nó là chế
    độ hỏng khó chấp nhận nhất của một hạng mục quan sát."""

    # ------------------------------------------------------------ Model
    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(default=32, ge=1)

    # ------------------------------------------------------------ Khác
    log_level: str = "INFO"
    cache_dir: Path = Path(".cache")

    @field_validator("embedding_device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"EMBEDDING_DEVICE phải thuộc {sorted(allowed)}, nhận được {v!r}")
        return v

    @property
    def postgres_dsn(self) -> str:
        """DSN đồng bộ, dùng cho Alembic và cho phép thử `/ready`.

        ⚠️ Ghi rõ driver `+psycopg` (v3) chứ không để `postgresql://` trần:
        SQLAlchemy 2.0 mặc định `postgresql://` thành **psycopg2**, thứ không có
        trong dự án này — nên DSN trần chết bằng `ModuleNotFoundError` ở lần
        migration đầu, chứ không phải bằng một thông báo về cấu hình.
        """
        pwd = self.postgres_password.get_secret_value()
        return self._dsn(self.postgres_user, pwd)

    @property
    def postgres_app_dsn(self) -> str:
        """DSN của **ứng dụng** — role không superuser, nên RLS áp lên nó.

        Mọi thứ chạm dữ liệu khách hàng phải đi qua DSN này. `postgres_dsn` chỉ
        dành cho migration và cho việc quản trị.
        """
        return self._dsn(self.postgres_app_user, self.postgres_app_password.get_secret_value())

    def _dsn(self, user: str, password: str) -> str:
        # ⚠️ `quote_plus` cho cả user lẫn password (`NEW-08`/`AU-04`): một mật
        # khẩu chứa `@` trong f-string trần làm SQLAlchemy parse phần sau `@`
        # thành **host** — kết nối đi sang máy khác mà không có lỗi nào cảnh
        # báo, và DSN migration còn mang quyền superuser.
        return (
            f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def require(self, *names: str) -> None:
        """Khẳng định các secret cần thiết đã có, nếu thiếu thì báo đủ tên một lần.

        Gọi ở đầu mỗi entrypoint cần secret, thay vì để `None` chui xuống tầng
        HTTP client rồi báo 401 khó hiểu.
        """
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            env_names = ", ".join(n.upper() for n in missing)
            raise RuntimeError(
                f"Thiếu biến môi trường bắt buộc: {env_names}. "
                f"Sao chép `.env.example` thành `.env` rồi điền giá trị."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings dạng singleton. Gọi `get_settings.cache_clear()` trong test."""
    return Settings()
