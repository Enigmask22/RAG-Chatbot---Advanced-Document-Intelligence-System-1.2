"""Ứng dụng FastAPI của Serving Plane — `W4-03`.

    uv run uvicorn serving.api.app:app --port 8000

## ⭐ Khởi động **không** được chết vì bundle nạp lỗi

Phản xạ đầu tiên là fail-fast: nạp bundle trong `lifespan`, nạp không được thì
ném, tiến trình chết. Ở một job offline điều đó đúng (`rag_core.settings` làm
đúng thế). Ở một container serving nó sai, vì ba lý do cộng lại:

1. Container chết → orchestrator khởi động lại → chết → **crashloop**. Log của
   pod đã biến mất khi ta kịp gõ lệnh xem.
2. Không có tiến trình nào sống thì không có `/ready` nào để **hỏi vì sao**.
3. Không có tiến trình nào sống thì `POST /admin/bundle/reload` cũng không có —
   tức cách chữa duy nhất là deploy lại, kể cả khi nguyên nhân chỉ là gõ nhầm
   một số phiên bản.

Khởi động lên nhưng **chưa sẵn sàng** giữ nguyên phần an toàn — load balancer đọc
`/ready` nên không có traffic nào vào — mà đổi lại được cả ba điều trên.

⚠️ Điều này chỉ đúng khi `/ready` thực sự được cấu hình làm readiness probe. Nếu
deploy quên khai nó, mô hình này biến một pod hỏng thành một pod **nhận traffic
và trả 500**. Đó là đánh đổi thật, và nó phải nằm trong checklist deploy của
`W5`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from rag_core.bundle import POINTER_NAME, latest_bundle, read_pointer
from rag_core.generation import default_registry
from rag_core.llm import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GLM_MODEL,
    MIN_REASONING,
    CircuitBreaker,
    DailyBudget,
    LLMRouter,
    OpenAICompatProvider,
    Route,
    build_deepseek_provider,
    build_glm_provider,
)
from rag_core.settings import Settings, get_settings
from serving.api import admin, chat, feedback, health, ingest, ui
from serving.api.middleware import RequestContextMiddleware
from serving.api.security import AuthMiddleware
from serving.core.auth import ApiKeyStore
from serving.core.chat import ChatService
from serving.core.instrument import instrument_retriever
from serving.core.langfuse import build_sink
from serving.core.logging import configure_logging
from serving.core.metrics import MetricsSink, RagMetrics
from serving.core.probes import Check, ReadinessProbes
from serving.core.ratelimit import RateLimiter
from serving.core.registry import BundleRegistry, RuntimeBuilder
from serving.core.runtime import QdrantRuntimeBuilder
from serving.core.tracing import FanoutSink, TraceSink
from serving.core.understanding import QueryUnderstanding

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

__all__ = ["create_app"]  # `app` do `__getattr__` ở cuối file cấp

logger = logging.getLogger(__name__)


def _startup_version(settings: Settings) -> str | None:
    """Bundle nào được kích hoạt lúc khởi động. Ba tầng, giảm dần độ tường minh.

    1. `BUNDLE_VERSION` — người vận hành ghim tay, thắng tất cả.
    2. `bundles/CURRENT` — con trỏ phát hành (`W5-10`). Đây là đường mà
       production đi: promote ghi vào nó, rollback ghi ngược lại nó, và cả hai
       lần ghi ấy là commit đọc được.
    3. Bản semver cao nhất — ⚠️ **đoán**, và nó đoán sai theo đúng ba cách đã
       ghi ở docstring của `POINTER_NAME`. Giữ lại vì test và môi trường dev
       chưa có con trỏ, nhưng nó phải **nói ra là mình đang đoán**: một hệ
       thống chọn bundle bằng suy luận mà không để lại dòng log nào là một hệ
       thống không ai trả lời được câu "vì sao nó đang chạy bản này".
    """
    if settings.bundle_version is not None:
        return settings.bundle_version
    try:
        pointed = read_pointer(settings.bundle_root)
    except Exception:
        logger.exception("con trỏ %s hỏng", settings.bundle_root / POINTER_NAME)
        pointed = None
    if pointed is not None:
        logger.info("bundle theo con trỏ %s: %s", POINTER_NAME, pointed)
        return pointed
    try:
        newest = latest_bundle(settings.bundle_root)
    except Exception:
        # Một manifest hỏng trong thư mục **không** được ngăn tiến trình lên:
        # `/ready` 503 kèm lý do vẫn gỡ được, một crashloop thì không.
        logger.exception("không quét được %s", settings.bundle_root)
        return None
    if newest is None:
        return None
    logger.warning(
        "không có con trỏ %s trong %s — chọn bản semver cao nhất (%s) bằng SUY LUẬN. "
        "Trên production hãy ghim `BUNDLE_VERSION` hoặc tạo con trỏ, nếu không thì "
        "một lần đúc bundle thử nghiệm là một lần deploy.",
        POINTER_NAME,
        settings.bundle_root,
        newest.bundle_version,
    )
    return newest.bundle_version


def primary_generator(settings: Settings) -> str:
    """Slug model của nhánh sinh **chính** — thứ đi vào namespace cache. `W5-11`.

    Đọc từ `Settings`, không từ bundle, vì đó chính là chỗ nhánh sinh được chọn
    (`build_llm`). Trả `""` khi chưa cấu hình được: `ChatService` hiểu chuỗi
    rỗng là **tắt cache**, thay vì gộp mọi model vào một ô.

    Ghép cả `provider` lẫn `model` chứ không chỉ model: hai nhà cung cấp có thể
    phục vụ cùng một slug qua hai endpoint khác nhau, và khi đó câu trả lời vẫn
    là của hai hệ thống khác nhau.
    """
    # `"none"` không cần nhánh riêng: nó không có trong bảng, nên `.get` trả
    # `""` và dòng cuối trả `""`. Phép tiêm `M8` xoá nhánh ấy mà không test nào
    # đỏ — đúng dấu hiệu của một `if` chỉ để trấn an người đọc.
    model = settings.chat_model or {
        "deepseek": DEFAULT_DEEPSEEK_MODEL,
        "glm": DEFAULT_GLM_MODEL,
    }.get(settings.chat_provider, "")
    return f"{settings.chat_provider}:{model}" if model else ""


def primary_endpoint(settings: Settings) -> str:
    """Máy chủ nào đang phục vụ nhánh sinh chính — trục thứ tư của namespace cache.

    ## ⭐⭐ Bắt được trên hệ đang chạy, `W6-01`

    Trong lúc chụp ảnh màn hình cho giao diện: server trỏ vào DeepSeek **thật**
    phát lại nguyên văn câu trả lời do stub của `W6-05` sinh ra, và khung `done`
    khai `model: "deepseek-v4-flash"` — gọi tên một model chưa từng viết đoạn
    text ấy. `primary_generator` mang `provider:model`, mà `DEEPSEEK_BASE_URL`
    không nằm trong cả hai.

    Cùng một cặp provider+slug trỏ vào hai máy chủ khác nhau là **hai bộ sinh
    khác nhau** — và với một vLLM tự dựng thì slug còn do người dựng tự đặt.

    ⚠️ Trả URL **thô**, không băm: namespace là thứ người vận hành đọc bằng mắt
    khi đi tìm xem một câu trả lời cũ đến từ đâu, và một chuỗi hex ở đó biến một
    câu hỏi trả lời được trong mười giây thành một buổi chiều.
    """
    return {
        "deepseek": settings.deepseek_base_url,
        "glm": settings.glm_base_url,
        "openrouter": settings.openrouter_base_url,
    }.get(settings.chat_provider, "")


def _qdrant_check(registry: BundleRegistry) -> Check:
    """Phép thử Qdrant đi **qua chính retriever đang phục vụ**, không qua một
    client riêng.

    Một client riêng có thể trả lời "Qdrant sống" trong khi collection mà bundle
    này dùng đã bị xoá — tức `/ready` xanh còn mọi `/chat` trả rỗng. Hỏi đúng
    collection đang dùng là phép thử duy nhất đo được thứ mà request thật sự phụ
    thuộc vào.
    """

    def check() -> None:
        _store_of(registry.active.retriever).count()

    return check


def _store_of(retriever: Any) -> Any:
    """Đi xuống tận đáy `reranked[hybrid[dense]]` để tới `QdrantDenseRetriever`.

    ⚠️ Dùng `getattr` chứ không `isinstance` là có chủ đích ngược lại thói quen:
    `isinstance` bắt module này import `rag_core.retrieval.reranked` và
    `.hybrid` ở tầng module, tức kéo `qdrant-client` vào **mọi** lần import
    `serving.api.app` — kể cả trong test dùng builder giả. Đổi lại, `getattr`
    im lặng nếu quy ước tên thuộc tính đổi; đó là lý do có
    `test_the_probe_reaches_the_qdrant_store_through_every_wrapper`.
    """
    seen = retriever
    while True:
        nxt = getattr(seen, "base", None) or getattr(seen, "store", None)
        if nxt is None:
            return seen
        seen = nxt


def _postgres_check() -> Check:
    """⭐ "DB sẵn sàng" nghĩa là **migration đã chạy**, không phải "cắm được".

    `SELECT 1` trả lời câu thứ hai, và đó không phải cách hệ thống này hỏng. Cách
    nó hỏng là: image mới lên, `alembic upgrade head` chưa chạy, pod báo sẵn sàng,
    nhận traffic, rồi mọi request chết bằng `column … does not exist` — với một
    `SELECT 1` xanh suốt. Chi tiết ở `serving/db/engine.py`.

    Engine dựng **một lần** ở đây và giữ trong closure: `create_engine` không mở
    kết nối nào cho tới lượt probe đầu, nhưng dựng lại nó mỗi lượt probe thì mỗi
    lượt là một pool mới và một kết nối mới — tức phép thử sức khoẻ tự trở thành
    tải, đúng thứ mà TTL của `ReadinessProbes` sinh ra để tránh.
    """
    from serving.db.engine import make_engine, postgres_check

    engine = make_engine(pool_size=1, max_overflow=0)

    def check() -> None:
        postgres_check(engine)

    return check


def build_provider(
    settings: Settings, provider: str, model: str | None
) -> OpenAICompatProvider | None:
    """Một nhánh của bộ định tuyến. `None` = chưa cấu hình được (thiếu key).

    ⭐ Trả `None` thay vì ném. Cùng lý lẽ với việc bundle nạp lỗi không giết
    tiến trình (§đầu module): một container không lên được thì không có
    `/ready` nào để hỏi vì sao, và ở đây còn tệ hơn vì `/health`, `/ready`,
    `/admin/bundle` đều còn dùng được bình thường mà không cần LLM.

    ⭐ Kiểu trả về là lớp **cụ thể**, không phải Protocol, và đó là chỗ đúng để
    hẹp lại: `W4-07` cần cùng client này ở giao diện *không* stream
    (`LLMProvider.complete`). Một client, một pool kết nối, một model.

    ⚠️ `max_retries` lấy từ `chat_max_retries` (mặc định **1**), không phải mặc
    định 4 của provider: xem docstring của trường ấy — 4 lần với backoff tới
    30 s là đúng cho job offline và sai cho một request có người đang đợi.
    """
    if provider == "none":
        return None
    key = {
        "deepseek": settings.deepseek_api_key,
        "glm": settings.glm_api_key,
        "openrouter": settings.openrouter_api_key,
    }.get(provider)
    if key is None:
        logger.warning(
            "nhánh LLM %r chưa có API key — bỏ qua nhánh này",
            provider,
        )
        return None
    if provider == "deepseek":
        return build_deepseek_provider(
            model or DEFAULT_DEEPSEEK_MODEL,
            api_key=key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            max_retries=settings.chat_max_retries,
        )
    if provider == "glm":
        return build_glm_provider(
            model or DEFAULT_GLM_MODEL,
            api_key=key.get_secret_value(),
            base_url=settings.glm_base_url,
            max_retries=settings.chat_max_retries,
        )
    if model is None:
        # ⚠️ OpenRouter **không** có mặc định ở đây, và đó là quy tắc cứng #1:
        # slug phải tường minh. Một mặc định ở chỗ này là một con trỏ do người
        # khác nắm, đúng thứ mà quy tắc ấy cấm.
        logger.warning("nhánh openrouter cần `chat_fallback_model` là slug tường minh — bỏ qua")
        return None
    return OpenAICompatProvider(
        model,
        api_key=key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        max_retries=settings.chat_max_retries,
    )


def build_llm(settings: Settings) -> LLMRouter | None:
    """Bộ định tuyến LLM của `POST /chat` — `W4-08`.

    Một nhánh cũng là một router: đường đi qua cầu dao, trần chi phí và dòng log
    "model thực tế đã phục vụ" giống hệt nhau dù có fallback hay không. Cấu hình
    một-nhánh và cấu hình hai-nhánh vì thế **không** phải hai đường mã khác nhau
    — chỗ mà một fallback chưa từng chạy sẽ lặng lẽ hỏng.

    ⭐ `extra_body` gắn vào **từng** `Route`, không vào `ChatService`: xem
    docstring của `Route.extra_body`. Hai nhà cung cấp cần hai tham số tắt suy
    luận khác nhau, và một bảng dùng chung hỏng im lặng ở nhà này.
    """
    primary = build_provider(settings, settings.chat_provider, settings.chat_model)
    if primary is None:
        logger.warning(
            "chưa cấu hình được nhánh LLM chính (chat_provider=%s) — POST /chat sẽ trả 503, "
            "phần còn lại của API vẫn chạy",
            settings.chat_provider,
        )
        return None

    routes = [
        Route(
            provider=primary,
            label=settings.chat_provider,
            extra_body=MIN_REASONING.get(settings.chat_provider),
            breaker=CircuitBreaker(
                failure_threshold=settings.chat_breaker_threshold,
                cooldown_s=settings.chat_breaker_cooldown_s,
            ),
        )
    ]
    fallback = build_provider(
        settings, settings.chat_fallback_provider, settings.chat_fallback_model
    )
    if fallback is not None:
        routes.append(
            Route(
                provider=fallback,
                label=settings.chat_fallback_provider,
                extra_body=MIN_REASONING.get(settings.chat_fallback_provider),
                breaker=CircuitBreaker(
                    failure_threshold=settings.chat_breaker_threshold,
                    cooldown_s=settings.chat_breaker_cooldown_s,
                ),
            )
        )
    budget = (
        DailyBudget(settings.chat_daily_budget_usd, name="chat")
        if settings.chat_daily_budget_usd > 0
        else None
    )
    if budget is None:
        logger.warning("chat_daily_budget_usd=0 — KHÔNG có trần chi phí ngày cho POST /chat")
    return LLMRouter(routes=routes, budget=budget)


def build_understanding(settings: Settings, llm: LLMRouter | None) -> QueryUnderstanding:
    """Bước hiểu câu hỏi của `W4-07`, dùng **chung** bộ định tuyến với đường sinh.

    ⚠️ `chat_rewrite=false` tắt **đúng một** trong ba việc: viết lại đa lượt, thứ
    duy nhất tốn tiền và tốn TTFB. Định tuyến và phát hiện ngôn ngữ là luật
    thuần, không có lý do nào để tắt và không có công tắc nào tắt chúng.

    ⭐ **Không** truyền `extra_body` nữa: từ `W4-08` mỗi `Route` mang bảng của
    nhà cung cấp mình, và một giá trị ở tầng lời gọi sẽ **ghi đè** bảng ấy cho
    *mọi* nhánh — tức áp tham số của DeepSeek lên nhánh GLM, thứ trả HTTP 400.
    Cùng lý do `ChatService.extra_body` giờ để trống.
    """
    return QueryUnderstanding(
        llm=llm if settings.chat_rewrite else None,
        timeout_s=settings.chat_rewrite_timeout_s,
    )


def build_cache(settings: Settings) -> Any | None:
    """`W4-10` — semantic cache trên Redis. `None` = tắt.

    Client Redis async không kết nối lúc dựng (lazy), và mọi lỗi lúc chạy đều
    suy giảm thành cache miss bên trong `SemanticCache` — nên ở đây không có
    fail-fast: Redis chết không được phép chặn server phục vụ đường đầy đủ.
    """
    if not settings.chat_cache:
        return None
    try:
        import redis.asyncio as aioredis
    except ImportError:  # pragma: no cover - redis đi kèm arq trong extra serving
        logger.warning("chat_cache bật nhưng thiếu gói redis — cache tắt")
        return None
    from serving.core.semantic_cache import SemanticCache

    return SemanticCache(
        aioredis.from_url(settings.redis_url),  # type: ignore[no-untyped-call]
        threshold=settings.chat_cache_threshold,
        ttl_s=settings.chat_cache_ttl_s,
        max_entries=settings.chat_cache_max_entries,
    )


def build_sessions() -> async_sessionmaker[AsyncSession] | None:
    """Factory phiên async cho đường request (`W4-06`).

    Engine dựng ở đây và sống suốt vòng đời tiến trình — `create_async_engine`
    không mở kết nối nào cho tới lượt dùng đầu, nên nó rẻ lúc khởi động và
    không kéo dài thời gian tới lúc `/health` trả lời.

    Tách khỏi engine **đồng bộ** của `_postgres_check`: probe chạy trong
    threadpool với `pool_size=1`, đường request chạy trên vòng lặp sự kiện. Dùng
    chung một pool thì một `/ready` chậm giữ mất kết nối của một `/chat`.
    """
    try:
        from serving.db.engine import async_session_factory, make_async_engine

        return async_session_factory(make_async_engine())
    except Exception:
        logger.exception("không dựng được engine async — POST /chat sẽ trả 503")
        return None


def build_probes(registry: BundleRegistry) -> ReadinessProbes:
    """Tập phép thử của `/ready` — bundle + Qdrant + Postgres, đủ DoD `W4-03`."""
    return ReadinessProbes(
        checks={"qdrant": _qdrant_check(registry), "postgres": _postgres_check()}
    )


def _traced_runtime(builder: RuntimeBuilder) -> RuntimeBuilder:
    """Bọc chuỗi truy hồi mỗi lần một bundle được nạp — `W5-06`.

    Đặt ở đây chứ không trong `BundleRegistry`: registry là chỗ giữ *cái gì
    đang phục vụ*, và nhét quan sát vào đó buộc mọi test của `W4-02` phải biết
    về span. Bọc ở tầng builder thì `create_app(build_runtime=…)` của test vẫn
    tiêm được một retriever giả, và nó cũng được bọc — nên đường span được test
    bằng đúng cơ chế mà production dùng, không bằng một nhánh riêng.
    """

    def build(bundle: Any) -> tuple[Any, Any]:
        retriever, reranker = builder(bundle)
        return instrument_retriever(retriever), reranker

    return build


def create_app(
    *,
    settings: Settings | None = None,
    build_runtime: RuntimeBuilder | None = None,
    probe_factory: Callable[[BundleRegistry], ReadinessProbes] = build_probes,
) -> FastAPI:
    """`build_runtime` tiêm được từ ngoài để test chạy không cần Qdrant/GPU —
    cùng lý lẽ đã làm cho `BundleRegistry` nhận Protocol thay vì tự dựng
    (`W4-02` §1)."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    # `W4-11`: mỗi lần khởi động khai rõ prompt nào đang phục vụ — DoD "runtime
    # log rõ prompt version". Prompt hỏng đã chết từ lúc import `chat.py` (fail
    # fast có chủ đích, xem docstring `_PROMPTS` bên đó), nên tới đây chỉ còn
    # việc nói to.
    for prompt in default_registry().all():
        logger.info("prompt registry: %s (sha256 %s…)", prompt.spec, prompt.sha256[:12])

    registry = BundleRegistry(
        root=resolved.bundle_root,
        warmup=resolved.bundle_warmup,
        build_runtime=_traced_runtime(
            build_runtime
            or QdrantRuntimeBuilder(
                url=resolved.qdrant_url,
                api_key=(
                    resolved.qdrant_api_key.get_secret_value() if resolved.qdrant_api_key else None
                ),
                device=resolved.embedding_device,
                batch_size=resolved.embedding_batch_size,
                allow_runtime_drift=resolved.bundle_allow_runtime_drift,
            )
        ),
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        version = _startup_version(resolved)
        if version is None:
            logger.error(
                "không tìm thấy bundle nào trong %s — API lên nhưng /ready sẽ trả 503",
                resolved.bundle_root,
            )
        else:
            try:
                # Chặn vòng lặp sự kiện vài chục giây (nạp trọng số). Chấp nhận
                # được **chỉ ở đây**: chưa có traffic, và uvicorn chưa mở cổng
                # cho tới khi lifespan xong. Ở route thì không — xem `admin.py`.
                registry.activate(version)
            except Exception:
                logger.exception(
                    "nạp bundle %s lúc khởi động thất bại — API vẫn lên, /ready trả 503 "
                    "và POST /admin/bundle/reload sửa được mà không cần deploy lại",
                    version,
                )
        yield
        # Đẩy nốt hàng đợi trace trước khi tiến trình đi. Có trần thời gian —
        # xem `LangfuseSink.close`: một Langfuse chết không được phép giữ
        # container không tắt được.
        if langfuse_sink is not None:
            langfuse_sink.close()

    # ⭐⭐ `MetricsSink` đứng **trước** `LangfuseSink` — xem `FanoutSink`.
    # Langfuse vứt trace khi hàng đợi đầy, và hàng đợi đầy đúng lúc tải cao;
    # đảo thứ tự thì bảng RED mất đúng phần cần nhìn nhất.
    metrics = RagMetrics()
    langfuse_sink = build_sink(resolved)
    sinks: list[TraceSink] = [MetricsSink(metrics)]
    if langfuse_sink is not None:
        sinks.append(langfuse_sink)
    trace_sink: TraceSink = FanoutSink(tuple(sinks))
    api = FastAPI(
        title="RAG serving",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
    )
    api.state.settings = resolved
    api.state.registry = registry
    api.state.probes = probe_factory(registry)
    api.state.metrics = metrics
    # ⚠️ `state.trace_sink` là cái **Langfuse**, không phải cái fanout: nó tồn
    # tại để `/admin/tracing` và `/metrics` hỏi `status()` (hàng đợi đầy chưa,
    # vứt bao nhiêu). Một `FanoutSink` không có hàng đợi nào để khai.
    api.state.trace_sink = langfuse_sink
    llm = build_llm(resolved)
    api.state.chat = ChatService(
        registry=registry,
        sessions=build_sessions(),
        llm=llm,
        top_k=resolved.chat_top_k,
        max_tokens=resolved.chat_max_tokens,
        generator=primary_generator(resolved),
        endpoint=primary_endpoint(resolved),
        # ⭐ `extra_body` để trống: từ `W4-08` mỗi `Route` mang bảng của nhà
        # cung cấp mình, và một giá trị ở đây sẽ ghi đè bảng ấy cho MỌI nhánh.
        understanding=build_understanding(resolved, llm),
        cache=build_cache(resolved),
        sink=trace_sink,
    )
    # ⚠️ **Thứ tự quan trọng và nó ngược trực giác.** `add_middleware` *chèn lên
    # đầu*, nên cái thêm **sau** nằm **ngoài**. Auth phải thêm trước để
    # `RequestContextMiddleware` bọc ngoài nó — nếu ngược lại thì mọi phản hồi
    # 401/403/429 do auth gửi sẽ không đi qua chỗ gắn `X-Request-ID`, tức đúng
    # những phản hồi mà người vận hành cần truy vết lại là những phản hồi không
    # truy được. Có test ghim (`test_a_401_still_carries_a_request_id`).
    api.state.keys = ApiKeyStore.load(resolved.api_keys_file)
    api.state.limiter = RateLimiter()
    api.add_middleware(AuthMiddleware, keys=api.state.keys, limiter=api.state.limiter)
    # ASGI thuần, không `BaseHTTPMiddleware` — lý do ở docstring của
    # `middleware.py`, nó liên quan trực tiếp tới SSE của `W4-06`.
    api.add_middleware(RequestContextMiddleware, metrics=metrics)
    api.include_router(health.router)
    api.include_router(admin.router)
    api.include_router(chat.router)
    api.include_router(feedback.router)
    api.include_router(ingest.router)
    api.include_router(ui.router)
    return api


def __getattr__(name: str) -> FastAPI:
    """`app` dựng **khi được lấy**, không phải khi module được import.

    `uvicorn serving.api.app:app` vẫn chạy đúng (uvicorn import module rồi
    `getattr`). Nhưng `import serving.api.app` trong một test thì không còn kéo
    theo `get_settings()` đọc `.env` thật và `configure_logging()` gỡ sạch handler
    của pytest — hai tác dụng phụ mà một dòng `import` không nên có.
    """
    if name == "app":
        return create_app()
    raise AttributeError(name)
