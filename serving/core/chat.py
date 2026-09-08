"""Một lượt hỏi–đáp: truy hồi → sinh theo dòng → ghi lại. `W4-06`.

## ⭐⭐ Đường phân giới: chỗ nào còn trả được HTTP status, chỗ nào thì không

Đây là quyết định kiến trúc của cả hạng mục, và nó là một **đường thật trong
mã** chứ không phải một lời dặn.

Sau khi byte đầu tiên của `200 OK` đã ra khỏi socket, không còn cách nào biến
một lỗi thành `503`. Nếu LLM chết ở token thứ 50, thứ client nhận được là một
dòng SSE dừng lại — và điều đó **trông y hệt** một câu trả lời ngắn đã kết thúc
bình thường. Không có mã lỗi, không có exception, và người dùng đọc nửa câu như
thể đó là toàn bộ câu trả lời.

Nên module này tách làm hai nửa:

| | `prepare()` | `stream_turn()` |
|---|---|---|
| chạy khi | chưa gửi byte nào | đang gửi |
| lỗi biểu hiện thành | **HTTP status thật** (404/403/503) | khung SSE `event: error` |
| gồm | kiểm tenant, mở hội thoại, **truy hồi** | gọi LLM, ghi message trợ lý |

⭐ **Truy hồi nằm ở nửa trên** là lựa chọn có chủ đích, và nó tốn ~600 ms TTFB.
Đổi lại: Qdrant chết trả `503` với `Retry-After` đọc được bằng máy, thay vì một
khung `event: error` mà mọi client phải tự viết mã xử lý. Nửa dưới càng mỏng thì
càng ít thứ chỉ hỏng được theo kiểu không nói ra được.

Hệ quả cho `W4-09`: xác minh citation cần **toàn bộ** câu trả lời, nên nó nằm ở
nửa dưới, và nó phải chấp nhận rằng một citation bịa chỉ báo được bằng một khung
SSE. Đó là lý do khung ở đây tên là `sources` (cái đã đưa cho model) chứ không
phải `citations` (cái đã kiểm) — hai thứ khác nhau, và gộp tên chúng lại là cách
chắc chắn để `W4-09` trở nên vô hình với client.

## ⭐ Ngắt kết nối giữa chừng: token đã trả tiền rồi

Người dùng đóng tab ở giây thứ 3 của một câu trả lời 10 giây. Tiền đã tiêu, và
phần đã sinh vẫn là dữ liệu thật. Nhưng lúc đó Starlette **huỷ** task đang chạy
generator này, nên mọi `await` trong `finally` bị huỷ ngay lập tức — tức đoạn mã
"lưu lại trước khi thoát" viết theo cách hiển nhiên nhất sẽ **không bao giờ
chạy**, và nó không bao giờ chạy đúng ở chỗ khó nhìn thấy nhất.

Cách chặn là `asyncio.create_task` — hàm **đồng bộ**, không treo, nên nó chạy
xong kể cả trong lúc bị huỷ, và task nó tạo ra sống độc lập với request. Ba chi
tiết đi kèm:

* Giữ **tham chiếu mạnh** tới task (`_PENDING`), nếu không GC thu nó giữa chừng
  và việc ghi biến mất — im lặng, ngẫu nhiên, chỉ dưới tải.
* Task tự mở phiên DB **của riêng nó**. Phiên của request đã bị đóng lúc đó.
* ⚠️ Tiến trình tắt trong lúc còn task chờ ⇒ mất bản ghi. Giới hạn thật, và cách
  chữa đúng là outbox — `W5`, không phải một `sleep` ở chỗ shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_core.generation import (
    CitationHoldback,
    context_nonce,
    default_registry,
    scan_injection,
    split_citation_block,
    verify_citations,
    wrap_context,
)
from rag_core.llm import BudgetExceeded, ChatMessage, LLMError, StreamingLLM
from rag_core.retrieval.filters import MetadataFilter
from rag_core.retrieval.hybrid import QdrantHybridRetriever
from rag_core.schemas import RetrievedChunk
from serving.core.auth import Principal, tenant_filter
from serving.core.quota import RedisDailySpend
from serving.core.registry import ActiveBundle, BundleRegistry, NoBundleLoadedError
from serving.core.semantic_cache import CachedAnswer, SemanticCache, embedder_of
from serving.core.single_flight import Flight, SingleFlight, flight_key
from serving.core.tracing import Trace, TraceSink, Usage, trace_scope
from serving.core.understanding import QueryPlan, QueryUnderstanding, detect_language
from serving.db.engine import atenant_session
from serving.db.models import Conversation, Message

__all__ = [
    "CHAT_NO_RETRIEVAL",
    "CHAT_SYSTEM",
    "HISTORY_PAGE",
    "MAX_HISTORY_MESSAGES",
    "NO_RETRIEVAL_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "ChatEvent",
    "ChatService",
    "ChatTurn",
    "ConversationNotFound",
    "GenerationUnavailable",
    "HistoryCursorNotFound",
    "prepare_ms_of",
]

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 10
"""Số message cũ đưa vào prompt.

Không phải một con số tuỳ tiện: lịch sử **không giới hạn** là một quả bom chi
phí có ngòi nổ chậm — hội thoại thứ 200 của một khách hàng gửi lại toàn bộ 199
lượt trước ở *mỗi* lượt, nên giá một câu hỏi tăng tuyến tính theo số câu đã hỏi,
cho tới khi nó vượt cửa sổ ngữ cảnh và request bắt đầu trả 400.

⚠️ Cắt theo **số message** chứ không theo token là một xấp xỉ thô, và nó sai
theo hướng nguy hiểm khi 10 message ấy đều dài. `HFTokenCounter` (`W1-10`) chờ
sẵn để thay bằng ngân sách token thật — `TD-51` (bản đầu của docstring này hứa
"W4-07/W4-11" nhưng việc ấy không thuộc DoD hạng mục nào).
"""

_PROMPTS = default_registry()
"""`W4-11`: prompt nạp từ registry YAML (`rag_core/generation/prompts/`), có
version + hash. Đổi nội dung mà không qua `scripts/prompt_stamp.py` thì import
này NÉM và server không lên — cố ý fail-fast, khác với bundle nạp lỗi: prompt
là package data đóng trong image, một file hỏng là một bản build hỏng, không
phải một trạng thái runtime chữa được bằng `/admin/bundle/reload`."""

CHAT_SYSTEM = _PROMPTS.get("chat-system")
"""Prompt nhánh RETRIEVE. Nội dung giữ nguyên byte so với hằng số cũ — các phép
đo `W4-07` (chỉ thị ngôn ngữ) và `W4-09` (block CITATIONS) gắn với đúng nội
dung này, đổi byte nào là các con số ấy thôi so được.

Câu cuối của template là hàng rào injection hạng nhẹ, ghi ra để nó không bị
nhầm là đã xong: một dòng chỉ dẫn không chặn được tài liệu cố tình chiếm quyền.
`W4-12` mới là chỗ đó, với bộ payload để đo.

⭐ Luật 4 ("trả lời bằng đúng ngôn ngữ của câu hỏi") không hoạt động một mình:
`W4-07` đo được 8/8 câu tiếng Anh nhận trả lời tiếng Việt khi thiếu dòng chỉ
thị cuối lượt người dùng — xem `QueryPlan.directive`."""

CHAT_NO_RETRIEVAL = _PROMPTS.get("chat-no-retrieval")
"""⭐ Prompt RIÊNG cho nhánh `NO_RETRIEVAL`, không phải `SYSTEM_PROMPT` với ngữ
cảnh rỗng: đưa `"hello"` vào đó cùng một khối ngữ cảnh trống thì model làm ĐÚNG
điều được bảo — nó trả lời rằng không đủ thông tin để chào lại. Luật đúng, ngữ
cảnh đúng, kết quả vô lý — và không có gì trong log nói ra."""

SYSTEM_PROMPT = CHAT_SYSTEM.text
"""Tên cũ giữ dạng `str` cho mọi chỗ đã dùng; nguồn sự thật là `CHAT_SYSTEM`."""

NO_RETRIEVAL_SYSTEM_PROMPT = CHAT_NO_RETRIEVAL.text


def cache_namespace(bundle_version: str, top_k: int, generator: str, endpoint: str) -> str:
    """`W4-11`: version prompt phải nằm trong namespace cache như bundle_version.

    Registry vừa biến prompt thành một biến số có version thì semantic cache
    phải invalidate theo nó: một câu trả lời sinh dưới `chat-system@v1` KHÔNG
    phải câu trả lời của `chat-system@v2`, và phát lại nó là phát lại kết quả
    của một hệ thống đã không còn tồn tại — đúng kiểu hỏng mà namespace-theo-
    bundle của `W4-10` sinh ra để chặn, chỉ là ở một trục khác.

    ## ⚠️ `NEW-08`/`AU-02`: `top_k` cũng là một biến số của câu trả lời

    Cùng một câu hỏi với `top_k=5` và `top_k=20` là hai lượt sinh trên hai bộ
    ngữ cảnh khác nhau — trả lời của lượt này cho lượt kia là vi phạm hợp đồng
    của chính tham số API. `top_k` đi vào **namespace** (không phải điều kiện
    loại): một client dùng `top_k` khác mặc định một cách nhất quán vẫn giữ
    được cache của riêng nó.

    ## ⭐⭐ `W5-11`: model sinh cũng vậy — và nó KHÔNG nằm trong `bundle_version`

    Tìm ra bởi chính lượt đo của `W5-11`: đổi nhánh sinh sang GLM thì lượt chạy
    thứ hai nhận lại nguyên văn câu trả lời của DeepSeek, và bảng ablation sẽ so
    một model với **chính nó** mà mọi con số trông vẫn bình thường.

    Cám dỗ là bảo `bundle_version` đã phủ rồi. Nó không phủ: `app.py` dựng nhánh
    sinh từ **`Settings`** (`chat_provider`/`chat_model`), không từ bundle. Nên
    một lần đổi biến môi trường — nâng model, đổi nhà cung cấp, sửa cấu hình
    failover — vẫn phát lại câu trả lời của model cũ tới hết TTL 24 giờ, với
    một `bundle_version` không đổi và không có gì kêu.

    Cùng họ với `AU-02`, ở trục thứ ba: **một khoá cache phải chứa mọi đầu vào
    làm đổi câu trả lời**, và danh tính bộ sinh là đầu vào lớn nhất trong số đó.

    ## ⭐⭐ `W6-01`: và ĐIỂM CUỐI cũng là một phần của danh tính ấy

    Bắt được trên hệ đang chạy, trong lúc chụp ảnh màn hình cho `W6-01`: server
    trỏ vào DeepSeek **thật** phát lại nguyên văn câu trả lời do stub của
    `W6-05` sinh ra — và khung `done` khai `model: "deepseek-v4-flash"`, tức nó
    gọi tên một model chưa từng viết ra đoạn text ấy.

    Lý do: `primary_generator` dựng danh tính từ `chat_provider` + `chat_model`,
    và `DEEPSEEK_BASE_URL` không có trong đó. Cùng một cặp provider+slug trỏ
    vào hai máy chủ khác nhau (stub, vLLM tự dựng, một proxy, một region khác)
    là **hai bộ sinh khác nhau** — và với vLLM thì slug thậm chí do người dựng
    tự đặt.

    Hẹp hơn `W5-11` ở production (base URL ít khi đổi), nhưng nó là đúng lỗi đã
    cắn trong lúc phát triển, và luật thì không đổi: một khoá cache phải chứa
    mọi đầu vào làm đổi câu trả lời. Endpoint là một trong số đó.

    ⚠️ Để **ngoài** `generator` chứ không nhét vào nó: `generator` còn là tín
    hiệu failover (`requested_model == generator.split(":", 1)[-1]`), và URL có
    dấu `:` sẽ làm phép tách ấy trả về nhầm chuỗi. Hai thứ khác nhau thì hai
    tham số khác nhau, không phải một chuỗi khéo léo.
    """
    return f"{bundle_version}+{CHAT_SYSTEM.spec}+k{top_k}+g{generator}+e{endpoint}"


def cache_eligible(
    plan: QueryPlan, history: Sequence[ChatMessage], filters: MetadataFilter | None
) -> bool:
    """Lượt nào được phép chạm cache. `W4-10`.

    Chỉ lượt ĐẦU hội thoại, tự đủ nghĩa, có truy hồi: câu hỏi giữa hội thoại
    mang nghĩa từ lịch sử, và hai người dùng có cùng một câu chữ giữa hai hội
    thoại khác nhau thì KHÔNG có cùng một câu hỏi. Bản viết lại cũng loại —
    nó phụ thuộc lịch sử theo định nghĩa.

    ⚠️ `NEW-08`/`AU-02`: request mang `filters` của client cũng loại. Một câu
    hỏi bó trong `doc_type=circular` KHÔNG phải câu hỏi ấy trên toàn corpus —
    trả câu trả lời cache của lượt không filter là vi phạm phạm vi dữ liệu
    được yêu cầu. Loại thay vì băm filter vào khoá: cache này bảo thủ có chủ
    đích (xem docstring `semantic_cache.py`), giá trị của nó nằm ở câu hỏi
    lặp nguyên văn không filter — ca có filter hiếm tới mức một khoá riêng
    chỉ nuôi entry chết. (`filters` ở đây là của CLIENT, trước khi
    `tenant_filter()` bọc — hàng rào tenant giống nhau cho mọi lượt của một
    tenant nên không cần vào khoá.)
    """
    return plan.retrieves and not history and not plan.rewritten and filters is None


def _unwrap_traced(candidate: Any) -> Any:
    """Bóc lớp `TracedRetriever` của `W5-06` (giữ `_inner`), nếu có."""
    inner = getattr(candidate, "_inner", None)
    return candidate if inner is None else inner


def wants_precomputed(retriever: Any, embedder: Any) -> bool:
    """Lượt này có dùng chung MỘT forward pass cho cache lẫn truy hồi không.

    `NEW-08`/`AU-06`. Hai điều kiện, cả hai đều cần:

    * nhánh nền (xuyên qua lớp wrap của reranker **và** lớp trace của
      `W5-06`) là `QdrantHybridRetriever` — retriever khác không nhận kwarg
      `precomputed`;
    * embedder có `embed_query_hybrid` — provider base trả `None` là chuyện
      của người gọi (rơi về `embed_query`), nhưng không có method thì thôi.

    `isinstance` chứ không duck-typing: truyền một cặp vector vào một retriever
    hiểu sai nó là loại lỗi *trông vẫn chạy* — thứ đắt nhất để tìm lại.

    ⚠️ Phải bóc `TracedRetriever` ở CẢ HAI tầng: production bọc cả chuỗi
    (`instrument_retriever` bọc reranked lẫn nhánh nền của nó), nên
    `retriever.base` của server thật là một `TracedRetriever(hybrid)` chứ
    không phải `QdrantHybridRetriever`. Bản đầu thiếu bước bóc này: mọi unit
    test trên class trần xanh, còn trên server thật `isinstance` fail lặng lẽ
    và đường embed-một-lần không bao giờ chạy — probe đo sống
    (`probes/new08-au06-single-embed.json`) là thứ bắt được nó, không phải
    test.
    """
    outer = _unwrap_traced(retriever)
    target = _unwrap_traced(getattr(outer, "base", outer))
    return isinstance(target, QdrantHybridRetriever) and callable(
        getattr(embedder, "embed_query_hybrid", None)
    )


_PENDING: set[asyncio.Task[None]] = set()


async def _ghi_chi_phi(spend: RedisDailySpend, tenant: str, cost_usd: float) -> None:
    """Bọc `charge` để nó trả `None`.

    ⚠️ Không nới `_PENDING` thành `Task[Any]`: kiểu ấy đang nói đúng một điều —
    *"mọi task nền ở đây là việc phụ, không ai đọc kết quả"* — và nới nó ra là
    mở đường cho một task **có** kết quả bị bỏ quên ở đây. Rẻ hơn là vứt kết
    quả ở đúng chỗ nó bị vứt.
    """
    await spend.charge(tenant, cost_usd)


"""Tham chiếu mạnh tới các task ghi đang chạy — xem §"Ngắt kết nối" ở docstring.

`asyncio` chỉ giữ tham chiếu **yếu** tới task, nên một task không ai giữ có thể
bị GC dọn giữa chừng. Tài liệu chuẩn nói đúng điều này, và triệu chứng của việc
bỏ qua nó là những lần ghi biến mất ngẫu nhiên dưới tải.
"""


class HistoryCursorNotFound(LookupError):
    """Con trỏ phân trang trỏ vào một message không có trong hội thoại này.

    ⭐ `AU-08`: tách khỏi `ConversationNotFound` vì hai lỗi này bảo người gọi
    làm hai việc khác nhau — một cái là "hội thoại không tồn tại, đừng hỏi
    nữa", cái kia là "con trỏ của bạn cũ rồi, đọc lại từ đầu". Gộp làm một
    404 chung thì client vòng lặp phân trang không phân biệt được.
    """


class ConversationNotFound(LookupError):
    """Không có hội thoại ấy — **hoặc** nó thuộc tenant khác.

    ⭐ Cố ý không phân biệt hai ca. RLS làm cho câu `SELECT` trả rỗng ở cả hai,
    và giữ nguyên sự mập mờ ấy trong phản hồi là điều đúng: một `403` cho hội
    thoại của người khác và `404` cho hội thoại không tồn tại biến endpoint này
    thành máy dò xem một `conversation_id` có tồn tại hay không.
    """


class GenerationUnavailable(RuntimeError):
    """Chưa cấu hình được nguồn sinh text, hoặc bundle chưa nạp."""


@dataclass(frozen=True)
class ChatEvent:
    """Một khung SSE. `event` là tên khung, `data` được JSON hoá nguyên vẹn."""

    event: str
    data: dict[str, Any]


@dataclass
class ChatTurn:
    """Mọi thứ đã giải quyết xong **trước** khi byte đầu tiên rời đi."""

    principal: Principal
    conversation_id: str
    user_message_id: str
    plan: QueryPlan
    history: list[ChatMessage]
    contexts: list[RetrievedChunk]
    bundle_version: str
    answer_message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Id của hàng `message` trợ lý — sinh **trước** khi có câu trả lời.

    ## ⭐⭐ Không có trường này thì feedback không có gì để trỏ vào

    `W5-08` cần một khoá cho "câu trả lời này sai". Khung `meta` tới giờ chỉ
    mang `message_id` của **người dùng**, còn hàng trợ lý ra đời trong một task
    nền sau khi stream kết thúc — tức client không bao giờ nhìn thấy id của
    đúng cái nó muốn chấm. Cách hiển nhiên (gắn feedback vào message người
    dùng) làm hỏng đúng chỗ nó cần đúng: một câu hỏi hỏi lại lần hai cho hai
    câu trả lời khác nhau, và cả hai chấm vào cùng một hàng.

    Sinh trước, phát ra trong `meta`, rồi `_save()` dùng lại — nên client cầm
    khoá từ khung đầu tiên chứ không phải sau khung cuối.

    ⭐ `TD-78`: id này là một **sự thật**, không còn là một lời hứa — `_open_turn`
    ghi hàng placeholder (`content=""`, `finish_reason="pending"`) trong cùng
    transaction với câu hỏi, và `_save` chỉ còn **điền vào**. Feedback tới trước
    khi task nền chạy, hay tới cho một lượt model im lặng, đều có hàng để trỏ.

    Default factory chỉ còn cho chế độ không trạng thái (`sessions=None`) và cho
    test dựng `ChatTurn` bằng tay; đường thật luôn nhận id từ `_open_turn`.
    """

    cached: CachedAnswer | None = None
    """`W4-10`: lượt này được trả từ cache — `stream_turn` phát lại thay vì gọi model."""
    cache_vector: Any | None = None
    """Vector câu hỏi đã embed cho lần tra cache TRƯỢT — giữ lại để ghi cache
    sau khi stream thành công, khỏi embed lần thứ ba."""
    resolved_top_k: int = 5
    """`top_k` đã chốt cho lượt này (`NEW-08`/`AU-02`) — thành phần của
    `cache_namespace`, và phải là CÙNG một giá trị ở đầu tra lẫn đầu ghi."""
    cached_from_flight: bool = False
    """`NEW-10`: `cached` của lượt này đến từ **single-flight**, không từ Redis.

    ⚠️ Tồn tại vì một phép đo, không vì gọn gàng: probe `new10-singleflight-after`
    cho thấy 7 người theo sau đi qua đúng nhánh phát lại của `W4-10`, nên span
    `cache.replay` — thứ `MetricsSink` dùng để đếm *"phục vụ mà không gọi
    provider"* — quy **cả** công của single-flight cho semantic cache. Con số
    không sai, nhưng cái tên thì nói dối về **cơ chế**, và bảng RAG Health sẽ
    báo tỉ lệ trúng cache tăng vọt sau một bản vá không đụng gì tới cache. Cùng
    lý lẽ với `refusals_suspected` của `W5-07`: đặt tên đúng thứ đang đếm."""
    shared_answer: CachedAnswer | None = None
    """`NEW-10`: thứ leader chia cho người theo sau — đặt **bên trong** đúng
    khối điều kiện đã cho phép ghi cache, không dựng lại bộ điều kiện ấy ở chỗ
    giải phóng. Hai bản sao của một luật là `AU-12`, và ở đây bản sao lệch
    nghĩa là chia câu trả lời của nhánh failover."""
    flight: Flight | None = None
    """`NEW-10`: vé single-flight của lượt này, và **chỉ khi lượt này là leader**.

    Follower không giữ vé: nó đã `await` xong ở `_prepare` và không có gì để
    giải phóng. Đặt vé của follower vào đây sẽ khiến `finally` của
    `stream_turn` gọi `resolve()` trên future của **người khác** — `resolve()`
    tự chặn bằng `is_leader`, nhưng dựa vào một phép kiểm ở xa để giữ đúng một
    bất biến gần là cách bất biến ấy chết ở lần sửa sau."""
    nonce: str = field(default_factory=context_nonce)
    """`W4-12`: mã phiên bọc mỗi khối ngữ cảnh. MỖI LƯỢT một mã mới — một mã cố
    định là một mã cuối cùng sẽ nằm trong một tài liệu nào đó, và từ giây ấy nó
    thôi là bí mật. Sinh bằng `secrets`, không phải `random`."""
    max_tokens: int = 1024
    started: float = field(default_factory=time.perf_counter)

    trace: Trace = field(default_factory=Trace)
    """`W5-06`. **Luôn** có một trace, kể cả khi không ai thu — khi ấy nó là một
    `Trace` không sink và cây span rơi vào GC lúc lượt kết thúc.

    ⭐ Không khai `Trace | None`, và lý do là số chỗ phải kiểm. Một `Optional`
    ở đây sinh ra khoảng một tá `if trace is not None` rải khắp hai nửa của
    lượt, và mỗi cái là một chỗ để một span biến mất khi ai đó quên. Dựng
    thừa vài dataclass cho một request 2,5 giây là cái giá không đo được;
    một span thiếu trong đúng lượt cần đọc thì đo được."""

    @property
    def question(self) -> str:
        """Chuỗi **đã đưa vào truy hồi** — viết lại rồi nếu `W4-07` có viết lại.

        ⚠️ Không phải chuỗi người dùng gõ; cái đó là `plan.original`, và nó mới
        là cái được ghi vào cột `content` của message. Gộp hai thứ này lại là
        cách chắc chắn để lịch sử hội thoại hiện ra một câu hỏi không ai hỏi.
        """
        return self.plan.question

    def prompt_spec(self) -> str | None:
        """Prompt nào đã đứng sau lượt này — `chat-system@v1`, hoặc `None` cho
        nhánh CLARIFY (không gọi model). Đi vào khung `meta`: mỗi lượt tự khai
        biến số prompt của mình, để một con số eval sau này truy được nó sinh
        dưới prompt nào mà không phải đoán từ ngày giờ."""
        if self.plan.route == "clarify":
            return None
        return CHAT_SYSTEM.spec if self.plan.retrieves else CHAT_NO_RETRIEVAL.spec

    def sources(self) -> list[dict[str, Any]]:
        """Cái đã **đưa cho model**, đánh số khớp với `[n]` trong prompt.

        `W4-12`: mỗi nguồn mang thêm `flags` — tên các luật tiêm đã khớp trong
        nội dung chunk. Cờ đi ra tới **client** chứ không chỉ vào log: người đọc
        câu trả lời là người duy nhất biết nó có bất thường hay không, và giấu
        cờ ở log server là bắt họ tin mà không đưa dữ liệu.

        ⚠️ Cờ **không** loại chunk khỏi ngữ cảnh. Bộ luật có dương tính giả
        (đo được: 2/20.424 chunk corpus thật), và một dương tính giả ở đường
        loại bỏ sẽ xoá lặng lẽ một tài liệu thật khỏi câu trả lời — đổi một
        kiểu hỏng ồn ào lấy một kiểu hỏng câm.
        """
        out: list[dict[str, Any]] = []
        for n, hit in enumerate(self.contexts, start=1):
            meta = hit.chunk.metadata
            out.append(
                {
                    "n": n,
                    "flags": list(scan_injection(hit.chunk.content)),
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_id": hit.chunk.doc_id,
                    "title": meta.title if meta else None,
                    "source_url": meta.source_url if meta else None,
                    "section_path": hit.chunk.section_path,
                    "score": round(hit.score, 6),
                    # ⭐⭐ `W6-01`: nội dung chunk đi ra client, và đó là điều
                    # kiện để "bấm citation → nhảy tới chỗ được trích" tồn tại.
                    # Không có nó thì UI chỉ hiện được **tiêu đề** nguồn, tức
                    # người đọc vẫn phải tin lời model rằng quote có thật —
                    # đúng thứ `W4-09` sinh ra để không phải tin.
                    #
                    # Không phải một khoản lộ mới: chunk này **đã** đi tới model
                    # trong prompt của chính người dùng đang hỏi, và bộ lọc
                    # tenant đã chạy trước đó (`tenant_filter()`). Thứ thêm vào
                    # là ai *nhìn thấy* nó — client hỏi, thay vì chỉ nhà cung
                    # cấp LLM.
                    #
                    # ⚠️ Đi kèm một nghĩa vụ ở phía client: nội dung này là dữ
                    # liệu corpus **không tin được** (`flags` ngay trên đây nói
                    # thẳng thế). UI phải dựng nó bằng `textContent`, không bao
                    # giờ `innerHTML` — xem `serving/ui/index.html`.
                    "content": hit.chunk.content,
                }
            )
        return out

    def persisted_sources(self) -> list[dict[str, Any]]:
        """Nguồn để **ghi xuống Postgres** — khác `sources()` ở đúng một nhánh.

        ## ⭐⭐ Lỗi thật, và nó chế ra đúng triệu chứng mà công cụ này đi tìm

        Một lượt trúng cache không chạy truy hồi, nên `contexts` rỗng và
        `sources()` trả về `[]`. Khung SSE thì vẫn đầy đủ — nó phát
        `cached.sources`. Nên trước `W5-08`, cứ mỗi lần trúng cache là hàng
        Postgres mất sạch nguồn trong khi client nhìn thấy chúng.

        Vô hình cho tới hạng mục này. Rồi file ứng viên đọc hàng ấy và in ra:
        **3 citation, 0 chunk được truy hồi** — tức một citation trỏ vào tài
        liệu chưa từng được đưa cho model, thứ trông y hệt một citation bịa.
        Người review sẽ mở đúng câu ấy ra tìm một lỗi bộ sinh không tồn tại.

        ⚠️ Đo được bằng lượt chạy thật, không bằng test: bộ test đơn vị tắt
        cache, và `W4-10` chỉ kiểm khung SSE — đúng chỗ dữ liệu vẫn đúng.

        ## ⭐⭐ `W6-01`: và **không** mang `content` xuống Postgres

        Khung SSE giờ chở nguyên văn chunk để UI nhảy tới chỗ được trích. Hàng
        Postgres thì không được: nó là **bản sao thứ hai của index**, phình mỗi
        hàng lịch sử từ ~1 KB lên ~8 KB, và nó sẽ đi tiếp vào file ứng viên
        golden set của `W5-08`. Cùng một danh sách nguồn phục vụ hai mục đích
        khác nhau, nên nó phải là hai payload khác nhau — và sự khác nhau ấy
        phải viết ra ở một chỗ, chứ không để mỗi chỗ đọc tự nhớ.
        """
        raw = list(self.cached.sources) if self.cached is not None else self.sources()
        return [{k: v for k, v in source.items() if k != "content"} for source in raw]

    def prompt(self) -> list[ChatMessage]:
        directive = self.plan.directive()
        if not self.plan.retrieves:
            # Nhánh `NO_RETRIEVAL`: không ngữ cảnh, không luật trích nguồn, và
            # dùng chuỗi **gốc** — không có gì để viết lại trong một lời chào.
            return [
                ChatMessage(role="system", content=NO_RETRIEVAL_SYSTEM_PROMPT),
                *self.history,
                ChatMessage(role="user", content=f"{self.plan.original}{directive}"),
            ]
        # ⭐⭐ **Cả hai** chuỗi, và thứ tự này là kết quả của một lần chạy thật.
        #
        # Bản đầu chỉ đưa câu **gốc**, với lý lẽ: truy hồi cần một chuỗi tự đủ
        # nghĩa để so vector, còn model đã có lịch sử ở ngay trên và nên thấy
        # đúng thứ người dùng vừa gõ. Lý lẽ nghe đúng và **sai trong thực tế**:
        # với `"cái đó thì sao?"`, `deepseek-v4-flash` truy hồi ra đúng 5 chunk
        # về di cư lao động rồi trả lời *"tôi không đủ thông tin để trả lời câu
        # hỏi 'cái đó thì sao?' vì câu hỏi không nêu rõ 'cái đó' là gì"*. Lịch
        # sử có trong prompt; model vẫn áp luật 3 lên chuỗi mơ hồ trước mắt nó.
        #
        # Đưa **mỗi** bản viết lại thì câu trả lời lại nói về một câu hỏi người
        # dùng không gõ. Đưa cả hai giữ được cả hai: người dùng thấy chữ của
        # mình, model có bản đã giải nghĩa, và một bản viết lại lệch chủ đề nằm
        # ngay cạnh bản gốc để model tự thấy.
        return [
            # `.replace` chứ không `.format`: template chứa `{"n": 1, …}` của
            # mẫu CITATIONS, và `.format` sẽ nổ trên đúng những dấu ngoặc ấy.
            ChatMessage(role="system", content=SYSTEM_PROMPT.replace("{{nonce}}", self.nonce)),
            *self.history,
            ChatMessage(role="user", content="\n\n".join(self.user_content_parts())),
        ]

    def context_blocks(self) -> list[str]:
        """⭐⭐ `W4-12`: khối ngữ cảnh bọc trong mốc mang nonce, thay cho `[n] …`
        trần. Đây là chỗ DUY NHẤT của hàng rào là một cơ chế thật: nội dung
        tài liệu viết được mọi thứ, trừ 16 ký tự hex sinh ra SAU khi nó đã
        nằm trong index — nên nó không đóng được khối dữ liệu để mở một
        khối chỉ thị giả. Số `[n]` giữ trong mốc vì luật 2 và `W4-09` đánh
        số nguồn theo nó."""
        return [
            wrap_context(n, hit.chunk.content, self.nonce)
            for n, hit in enumerate(self.contexts, start=1)
        ]

    def _user_tail(self) -> str:
        question = f"CÂU HỎI: {self.plan.original}"
        if self.plan.rewritten:
            question += f'\n(Hiểu đầy đủ theo hội thoại: "{self.plan.question}")'
        return f"{question}{self.plan.directive()}"

    def user_content_parts(self) -> list[str]:
        """Message user cuối, dưới dạng CÁC MẢNH — `TD-74`.

        ⭐⭐ Bất biến: `"\n\n".join(parts)` phải bằng ĐÚNG chuỗi đã gửi cho model
        (có test ghim). Vì thế header "NGỮ CẢNH:" dán vào khối đầu bằng `\n`
        (đúng như chuỗi ghép cũ) chứ không đứng thành mảnh riêng.

        Lý do tồn tại: `redact()` cắt MỖI CHUỖI ở `_MAX_TEXT` (4.000), mà một
        khối 5 chunk là ~10k — cắt chuỗi đã ghép làm người gỡ lỗi mất nửa sau
        ngữ cảnh, thường là nửa chứa lỗi. Cắt theo mảnh thì mỗi chunk có ngân
        sách 4.000 của riêng nó, và khối nào bị cắt tự khai ngay trong mảnh ấy.
        """
        blocks = self.context_blocks() or ["(không tìm thấy tài liệu liên quan)"]
        return [f"NGỮ CẢNH:\n{blocks[0]}", *blocks[1:], self._user_tail()]

    def prompt_trace_view(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Cách span `prompt` nhìn `messages` — `TD-74`: message ngữ cảnh phát
        `content_parts` thay vì `content`, để `redact()` (vốn đệ quy qua list)
        cắt từng khối một. Các message khác giữ nguyên `content`.

        ⚠️ Đây là một CÁCH TRÌNH BÀY, không phải một nguồn thứ hai: các mảnh
        đến từ đúng các hàm mà `prompt()` dùng để ghép, và bất biến tái dựng
        ở `user_content_parts` giữ cho hai đường không trôi khỏi nhau.
        """
        out: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in messages]
        if self.plan.retrieves and out:
            out[-1] = {"role": "user", "content_parts": self.user_content_parts()}
        return out


def prepare_ms_of(turn: ChatTurn) -> float | None:
    """Bao nhiêu mili giây đã trôi **trước** khi `turn.started` được đặt.

    ⭐⭐ Nửa còn lại của `TD-55`. `total_ms` trong khung `done` đếm từ
    `ChatTurn.started`, thứ được gán ở *dòng cuối* của `prepare()` — tức sau
    embed, truy hồi và rerank. `W4-13` đo được 725 ms nằm ngoài khung ấy; trace
    của `W5-06` đo lại trên 5 lượt thật và ra **787 ms, 19,9%** của một request.

    Một hệ thống báo SLA màu hồng bằng cách bắt đầu bấm giờ sau phần chậm nhất
    của chính nó là một hệ thống nói dối theo đúng một hướng. Khung `done` giờ
    khai con số ấy ra, nên client cộng được `prepare_ms + total_ms` mà không
    phải có Langfuse.

    `None` = không đo được (trace bị người gọi thay bằng một trace khác giữa
    chừng). Không thay bằng `0.0`: xem `tracing.Usage`.
    """
    value = turn.trace.metadata.get("prepare_ms")
    return float(value) if isinstance(value, (int, float)) else None


@dataclass
class ChatService:
    """Người điều phối một lượt. Không biết gì về HTTP hay SSE — đó là `api/chat.py`.

    `llm` khai kiểu `StreamingLLM` (Protocol) chứ không phải một lớp cụ thể, nên
    `W4-08` cắm router vào đây mà không sửa một dòng nào trong file này.
    """

    registry: BundleRegistry

    sessions: async_sessionmaker[AsyncSession] | None
    """`None` = chạy **không trạng thái**: không lịch sử, không ghi gì xuống đâu.

    ⭐⭐ Cho tới `W6-02` kiểu này nói dối. Nó khai `| None` từ đầu, còn dòng thứ
    hai của `_prepare` thì `raise GenerationUnavailable("chưa cấu hình Postgres")`
    — nên ai đọc chữ ký cũng dựng được đúng một cấu hình mà mã từ chối phục vụ,
    và biết điều đó ở *runtime*. Tôi là người ấy: HF Space không có Postgres, và
    tôi lắp `sessions=None` vì chữ ký cho phép.

    Chế độ này có thật và có người dùng: một endpoint RAG không lưu vết. Cái giá
    khai ra ở đây — truyền `conversation_id` vào một service không trạng thái là
    `ConversationNotFound`, **không** phải im lặng bỏ qua: im lặng bỏ qua để
    client tin nó có mạch hội thoại trong khi mỗi lượt là một lượt độc lập.
    """

    llm: StreamingLLM | None
    top_k: int = 5
    max_tokens: int = 1024

    cache: SemanticCache | None = None
    """`W4-10`. `None` = tắt. Mọi lỗi cache đều suy giảm thành miss — cache
    không bao giờ được phép là lý do `/chat` trả lỗi."""

    daily_spend: RedisDailySpend | None = None
    """`TD-47`. `None` = chỉ còn trần toàn cục của router (hành vi trước
    08/09/2026): một con số cho cả tiến trình, không chia theo tenant, về 0
    sau mỗi restart, và N× khi có N replica."""

    single_flight: SingleFlight | None = None
    """`NEW-10`. `None` = tắt (mọi request trùng nhau tự đi đường riêng, tức
    hành vi trước 08/09/2026).

    ⚠️ Phụ thuộc `cache`: khoá gộp dùng chính `cache_namespace`, và người theo
    sau nhận một `CachedAnswer`. Bật single-flight khi `cache=None` không sai
    nhưng vô nghĩa — `_prepare` chỉ vào đường này bên trong nhánh cache."""

    sink: TraceSink | None = None
    """`W5-06`. `None` = cây span vẫn được dựng nhưng không đi đâu cả. Cùng hợp
    đồng với `cache`: quan sát hỏng làm mất một trace, không làm mất một câu
    trả lời."""

    understanding: QueryUnderstanding = field(default_factory=QueryUnderstanding)
    """`W4-07`. Mặc định là bản **không có LLM**: luật vẫn chạy đủ (định tuyến +
    ngôn ngữ), chỉ viết lại đa lượt là không có. Đó là mức suy giảm đúng — ba
    việc kia miễn phí và tất định, không có lý do gì để chúng phụ thuộc vào việc
    cấu hình được một provider."""

    generator: str = ""
    """Slug của model sinh **chính**, thứ đi vào namespace cache. `W5-11`.

    Rỗng = chưa khai ⇒ **cache tắt**, không phải "dùng chung một ô". Đó là mặc
    định fail-safe: một namespace thiếu danh tính bộ sinh là một namespace trộn
    câu trả lời của hai model, và hỏng theo kiểu im lặng nhất — `200 OK`, câu
    trả lời trôi chảy, sai hệ thống. Thà mất cache còn hơn phát lại lời của một
    model khác.
    """

    endpoint: str = ""
    """Máy chủ nào đã phục vụ `generator` — trục thứ tư của namespace. `W6-01`.

    Tách khỏi `generator` chứ không nhét vào: `generator` còn là tín hiệu
    failover và URL có dấu `:` sẽ làm phép tách ấy sai. Xem `cache_namespace`.

    Rỗng ở đây **không** tắt cache (khác `generator`): một triển khai chỉ có
    đúng một endpoint là bình thường, và bắt nó khai một chuỗi rỗng-nhưng-nhất-
    quán không mua thêm gì. Thứ nó chặn là hai endpoint **khác nhau** dùng chung
    một ô, và điều đó cần hai giá trị khác nhau chứ không cần một giá trị.
    """

    extra_body: Mapping[str, Any] | None = None
    """⭐⭐ Tham số ngoài chuẩn của provider — trong thực tế là `MIN_REASONING`.

    Lần chạy thật đầu tiên của endpoint này, với `max_tokens = 1024` và một
    prompt 1.613 token, trả về **0 ký tự**: toàn bộ 1024 token completion đi vào
    chuỗi suy luận của `deepseek-v4-flash`, thứ không xuất hiện trong `content`.
    `finish_reason` = `"empty"`, hoá đơn $0,0015, câu trả lời không tồn tại.

    Đó là **đúng** phát hiện của `W3-04` (83% token vào suy luận, 6/30 request
    trả rỗng), xuất hiện lại ở đường request. Khác biệt là ở đó nó tốn tiền, còn
    ở đây nó là một endpoint hỏng — và nó hỏng theo kiểu tệ nhất: `200 OK`, dòng
    SSE hợp lệ, không một khung `error` nào.

    Đặt ở `ChatService` chứ không nhét cứng vào provider: bảng đúng phụ thuộc
    **nhà cung cấp**, và `app.py` là chỗ duy nhất biết nhà nào đang được dùng.
    `W4-08` sẽ chuyển nó vào router, nơi mỗi nhánh fallback mang bảng của mình.
    """

    # ---------------------------------------------------------- nửa trên

    async def prepare(
        self,
        principal: Principal,
        *,
        question: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
        trace: Trace | None = None,
    ) -> ChatTurn:
        """Mọi thứ còn hỏng thành một HTTP status tử tế. Xem bảng ở docstring module.

        ⭐ `trace` do **người gọi** cấp, không do hàm này tạo, và đó là điều kiện
        để quan sát nhìn thấy phần đáng nhìn nhất. Nếu trace ra đời cùng
        `ChatTurn` — thứ chỉ tồn tại ở *dòng cuối* của hàm này — thì mọi lượt
        404/403/429/503 không có trace nào cả: hệ thống quan sát sẽ phủ đúng
        những request đã chạy trót lọt. `api/chat.py` mở trace trước khi gọi vào
        đây, và đóng nó ở `except` của chính nó.
        """
        trace = trace if trace is not None else Trace(sink=self.sink)
        with trace_scope(trace):
            return await self._prepare(
                principal,
                question=question,
                conversation_id=conversation_id,
                top_k=top_k,
                filters=filters,
                trace=trace,
            )

    async def _prepare(
        self,
        principal: Principal,
        *,
        question: str,
        conversation_id: str | None,
        top_k: int | None,
        filters: MetadataFilter | None,
        trace: Trace,
    ) -> ChatTurn:
        """Thân của `prepare()`, chạy bên trong `trace_scope`.

        ⚠️ `trace_scope` là một `ContextVar`, và một `ContextVar` đặt trong thân
        một **async generator** sẽ rò sang context của người gọi giữa hai lần
        `yield` (generator chạy trong context của người tiêu thụ nó). Đó là lý
        do nửa dưới — `stream_turn`, đúng là một async generator — **không** dùng
        `trace_scope` mà truyền `Span` cha tường minh. Ở đây thì an toàn: một
        coroutine thường không bị treo giữa chừng bởi người ngoài, nên khối
        `with` luôn reset trước khi hàm trả về.
        """
        if self.llm is None:
            raise GenerationUnavailable(
                "chưa cấu hình LLM cho serving — đặt `DEEPSEEK_API_KEY` rồi khởi động lại"
            )
        # ⭐ `W4-08`: hỏi trần chi phí **trước** khi tốn một lượt truy hồi, và
        # trước khi byte đầu tiên rời đi. Sau `200 OK` thì "hết ngân sách" chỉ
        # còn là một dòng SSE dừng lại — cùng đường phân giới ở docstring module.
        # `BudgetExceeded` bay thẳng lên `api/chat.py` và thành `429`.
        check = getattr(self.llm, "assert_within_budget", None)
        if callable(check):
            check()
        # ⭐⭐ `TD-47`: trần **theo tenant**, ngay cạnh trần toàn cục ở trên —
        # và hai cái này **không** thay nhau được.
        #
        # Trần của router là một con số cho cả tiến trình, nên một tenant đốt
        # hết sẽ làm *mọi* tenant còn lại nhận `429`. Trần ở đây chia phần theo
        # `tenant_id` và đếm trên Redis, nên nó vừa công bằng vừa không về 0
        # sau restart. Trần kia **ở lại** làm hàng rào thô thứ hai: nó là thứ
        # duy nhất còn chặn khi Redis hỏng và cả hai đường lui cùng bị nới ra.
        #
        # ⚠️ `peek`, không `charge`: chỗ này chỉ **hỏi**. Ghi nhận xảy ra sau
        # khi biết chi phí thật, ở cuối `_stream_turn` — trả tiền cho một ước
        # lượng rồi không sửa lại là cách để sổ chi tiêu trôi khỏi hoá đơn.
        if self.daily_spend is not None:
            quyet_dinh = await self.daily_spend.peek(principal.tenant_id)
            if not quyet_dinh.allowed:
                raise BudgetExceeded(
                    f"tenant {principal.tenant_id!r} đã tiêu "
                    f"${quyet_dinh.spent_usd:.4f}/${quyet_dinh.cap_usd:.2f} hôm nay"
                )

        try:
            snapshot: ActiveBundle = self.registry.active
        except NoBundleLoadedError as exc:
            # Ảnh chụp **một lần** cho cả lượt — luật 2 của `W4-02`. Đọc
            # `registry.active` lần thứ hai ở giữa lượt có thể trả về runtime
            # khác nếu có reload chen vào, và khi đó câu trả lời trích chunk của
            # index này bằng điểm số của index kia.
            raise GenerationUnavailable(str(exc)) from exc

        history = await self._history(principal, conversation_id)

        # ⭐ Lần gọi **đầu tiên** của `tenant_filter()` từ `W4-04`. Trước dòng
        # này nó chỉ có test; từ đây nó là thứ đứng giữa truy vấn của một khách
        # hàng và corpus của mọi khách hàng còn lại.
        #
        # ⚠️ Gọi **trước** khi rẽ nhánh theo `route`, và luôn gọi kể cả khi lượt
        # này không truy hồi: hàm này không chỉ *thu hẹp* filter, nó còn **từ
        # chối** filter trỏ sang tenant khác. Bỏ nó ở nhánh `no_retrieval` thì
        # cùng một request nhận `403` hay `200` tuỳ vào việc người dùng có chào
        # hỏi hay không — một chỗ dò danh sách tenant, và là một hành vi bảo mật
        # phụ thuộc vào bộ phân loại câu hỏi.
        scoped = tenant_filter(principal, filters)

        with trace.span("understand", input=question, n_history=len(history)) as span:
            # Span `rewrite` (nếu có) mở bên trong `QueryUnderstanding._rewrite`
            # và nối vào đây qua `ContextVar` — nó cần thời lượng và chi phí của
            # đúng lời gọi model, thứ chỉ hàm ấy nhìn thấy.
            plan = await self.understanding.plan(question, history)
            span.end(
                output={"question": plan.question, "reason": plan.reason},
                route=plan.route,
                language=plan.language,
                rewritten=plan.rewritten,
            )
        trace.metadata["route"] = plan.route
        trace.metadata["language"] = plan.language

        # ⭐ `W4-10`: tra cache TRƯỚC khi truy hồi — hit thì tiết kiệm cả lượt
        # embed+rerank (~800 ms) lẫn lượt model. Vector tra trượt được giữ lại
        # trên turn để ghi cache sau khi stream thành công. Mọi lỗi ở đây suy
        # giảm thành miss; đường đầy đủ không phụ thuộc cache sống hay chết.
        cached: CachedAnswer | None = None
        cache_vector: Any | None = None
        precomputed: Any | None = None
        flight: Flight | None = None
        cached_from_flight = False
        resolved_top_k = top_k or self.top_k
        # ⚠️ `and self.generator` ở đây là một **hàng rào hiệu năng**, không
        # phải hàng rào đúng đắn — phép tiêm `M5` (`W5-11`) xoá nó và **sống
        # sót** đúng như dự đoán, vì namespace nó đọc (`…+g`) là namespace mà
        # không đường ghi nào chạm tới được: đầu ghi đã bị điều kiện failover
        # chặn khi `generator` rỗng. Bỏ nó đi không sinh ra câu trả lời sai, chỉ
        # sinh ra một lần embed câu hỏi + một lượt Redis mỗi lượt chat, vĩnh
        # viễn miss. Giữ lại và nói rõ, thay vì thêm một bài test service-level
        # nặng để canh một thứ không thể sai.
        if self.cache is not None and self.generator and cache_eligible(plan, history, filters):
            embedder = embedder_of(snapshot.retriever)
            if embedder is not None:
                with trace.span("cache.lookup", input=plan.question) as span:
                    # ⭐ Span `embed.query` chỉ tồn tại ở **đường cache**, và đó
                    # không phải một thiếu sót ở đường kia. `prepare()` tự gọi
                    # embedder ở đây nên có mối nối để bọc; trong truy hồi thì
                    # `embed_query_hybrid` nằm trong thân
                    # `QdrantHybridRetriever.retrieve` và tách nó ra đòi sửa
                    # `rag_core`. Xem docstring `serving/core/instrument.py`.
                    with trace.span("embed.query") as embed_span:
                        # ⭐ `NEW-08`/`AU-06`: MỘT forward pass cho cả tra cache
                        # lẫn truy hồi. Bản cũ gọi `embed_query` ở đây rồi —
                        # khi miss — `retrieve()` tự gọi `embed_query_hybrid`
                        # trên đúng chuỗi ấy lần nữa: +12,6 ms và giữ khoá
                        # model (`TD-63`, thứ đã gây 503 dưới tải) HAI lần mỗi
                        # lượt. Với BGE-M3 phần dense của hai đường trùng nhau
                        # (prefix rỗng có chủ ý, cùng `_forward`, cùng L2).
                        # Chỉ đi đường này khi nhánh nền là hybrid — retriever
                        # khác không nhận kwarg `precomputed`.
                        if wants_precomputed(snapshot.retriever, embedder):
                            # `None` vẫn có thể trả về (default của provider
                            # base) — khi ấy rơi xuống `embed_query` như cũ.
                            precomputed = await asyncio.to_thread(
                                embedder.embed_query_hybrid, plan.question
                            )
                        if precomputed is not None:
                            cache_vector = precomputed[0]
                        else:
                            cache_vector = await asyncio.to_thread(
                                embedder.embed_query, plan.question
                            )
                        embed_span.end(
                            embedder=getattr(embedder, "name", None),
                            shared_with_retrieval=precomputed is not None,
                        )
                    assert cache_vector is not None
                    cached = await self.cache.lookup(
                        principal.tenant_id,
                        cache_namespace(
                            snapshot.version, resolved_top_k, self.generator, self.endpoint
                        ),
                        plan.question,
                        cache_vector,
                    )
                    span.end(
                        output={
                            "hit": cached is not None,
                            "similarity": cached.similarity if cached else None,
                            "matched_question": cached.question if cached else None,
                        },
                        hit=cached is not None,
                    )

                # ⭐⭐ `NEW-10`: cache TRƯỢT không có nghĩa là "chưa ai hỏi câu
                # này" — nó có nghĩa là "chưa ai hỏi xong". `AU-11` đo được 8
                # request trùng nhau đi lọt qua đúng khe giữa hai câu ấy: 8 lời
                # gọi trả tiền, 0 hit. Chỗ gộp phải nằm **ngay sau** lượt tra và
                # **trước** truy hồi, vì phần đắt (rerank + sinh) nằm ở dưới.
                if cached is None and self.single_flight is not None:
                    key = flight_key(
                        principal.tenant_id,
                        cache_namespace(
                            snapshot.version, resolved_top_k, self.generator, self.endpoint
                        ),
                        plan.question,
                    )
                    ticket = self.single_flight.join(key)
                    if ticket.is_leader:
                        flight = ticket
                    else:
                        with trace.span("cache.single_flight", input=plan.question) as sf:
                            cached = await ticket.wait()
                            sf.end(output={"joined": cached is not None}, joined=cached is not None)
                        cached_from_flight = cached is not None
                        # ⚠️ Follower trượt (leader hỏng / quá hạn) **không**
                        # trở thành leader mới: nó đi đường đầy đủ và không ghi
                        # sổ. Một đàn follower cùng tranh làm leader mới là đúng
                        # cái thundering herd mà cơ chế này sinh ra để chặn, chỉ
                        # dời đi một nhịp. Giá phải trả: N request đầy đủ — đúng
                        # bằng hành vi trước `NEW-10`, không tệ hơn.

        # ⚠️⚠️ Từ đây tới `return`, một ngoại lệ sẽ bỏ rơi vé: leader đã ghi
        # sổ ở trên nhưng `stream_turn` — chỗ duy nhất gọi `resolve()` — sẽ
        # không bao giờ chạy. Follower khi ấy chờ **hết hạn giờ 15 giây** rồi
        # mới tự đi, tức một lỗi truy hồi 200 ms bị khuếch đại thành 15 giây
        # cho mọi người đứng sau. Đúng chế độ hỏng mà quyết định 3 của
        # `single_flight.py` tồn tại để chặn, chỉ ở một khúc mã khác — nên
        # hàng rào phải đặt ở cả hai khúc, không chỉ ở khúc dễ thấy.
        try:
            return await self._retrieve_and_open(
                trace=trace,
                principal=principal,
                conversation_id=conversation_id,
                plan=plan,
                history=history,
                snapshot=snapshot,
                scoped=scoped,
                precomputed=precomputed,
                cached=cached,
                cached_from_flight=cached_from_flight,
                cache_vector=cache_vector,
                resolved_top_k=resolved_top_k,
                flight=flight,
            )
        except BaseException:
            if flight is not None:
                flight.resolve(None)
            raise

    async def _retrieve_and_open(
        self,
        *,
        trace: Trace,
        principal: Principal,
        conversation_id: str | None,
        plan: QueryPlan,
        history: list[ChatMessage],
        snapshot: Any,
        scoped: MetadataFilter | None,
        precomputed: Any | None,
        cached: CachedAnswer | None,
        cached_from_flight: bool,
        cache_vector: Any | None,
        resolved_top_k: int,
        flight: Flight | None,
    ) -> ChatTurn:
        """Nửa sau của `_prepare`, tách ra **chỉ** để có một biên bắt ngoại lệ.

        Không phải một phép tách theo trách nhiệm — nó vẫn là cùng một việc.
        Nó tồn tại vì `try/except BaseException` bọc trọn phần còn lại của một
        hàm 200 dòng thì cái `try` ấy che mất chỗ nào đang được bảo vệ, và
        `AU-11` vừa dạy rằng cửa sổ đua nào không nhìn thấy được thì không ai
        đóng."""
        contexts: list[RetrievedChunk] = []
        if plan.retrieves and cached is None:
            retrieve_kwargs: dict[str, Any] = {"filters": scoped}
            if precomputed is not None:
                retrieve_kwargs["precomputed"] = precomputed
            contexts = list(
                await asyncio.to_thread(
                    # ⚠️ `retrieve()` là **đồng bộ** và tốn hàng trăm mili giây
                    # (embed trên GPU + cross-encoder). Gọi thẳng trong
                    # `async def` thì suốt khoảng đó vòng lặp sự kiện không chạy
                    # gì khác — kể cả `/health`, và orchestrator đọc đúng điều
                    # đó là "tiến trình chết". Cùng lý lẽ đã làm cho ba handler
                    # của `admin.py` là `def` chứ không `async def`.
                    snapshot.retriever.retrieve,
                    plan.question,
                    resolved_top_k,
                    **retrieve_kwargs,
                )
            )

        resolved_id, user_message_id, answer_message_id = await self._open_turn(
            principal, conversation_id, plan, snapshot.version
        )
        trace.session_id = resolved_id
        trace.user_id = principal.tenant_id
        # ⭐⭐ `TD-55` trả ở đây. Khung `done` đếm `total_ms` từ `ChatTurn.started`
        # — thứ được đặt ở **dòng dưới**, tức sau embed + truy hồi + rerank. Đo
        # được: `W4-13` thấy 725 ms nằm ngoài khung `done`, ~18% của một lượt.
        # Trace bắt đầu ở handler HTTP nên nó đo được cả phần ấy, và `prepare_ms`
        # nói ra đúng khoảng chênh giữa hai đồng hồ thay vì để người đọc trừ tay.
        trace.metadata["prepare_ms"] = round(
            (datetime.now(UTC) - trace.start_time).total_seconds() * 1000.0, 2
        )
        return ChatTurn(
            trace=trace,
            principal=principal,
            conversation_id=resolved_id,
            user_message_id=user_message_id,
            answer_message_id=answer_message_id,
            plan=plan,
            history=history,
            contexts=contexts,
            bundle_version=snapshot.version,
            max_tokens=self.max_tokens,
            cached=cached,
            cached_from_flight=cached_from_flight,
            cache_vector=cache_vector if cached is None else None,
            resolved_top_k=resolved_top_k,
            # Bất biến: leader ⇒ `cached is None` (vé chỉ được nhận trong
            # nhánh `cached is None`, và chỉ follower mới gán lại `cached`).
            # Nên leader **luôn** đi đường sinh ở `stream_turn`, tức luôn chạy
            # qua `finally` — chỗ duy nhất giải phóng vé.
            flight=flight,
        )

    # ---------------------------------------------------------- nửa dưới

    async def stream_turn(self, turn: ChatTurn) -> AsyncGenerator[ChatEvent, None]:
        """Lớp bọc mỏng quanh `_stream_turn`, **chỉ** để giải phóng vé `NEW-10`.

        ⭐⭐ Nó tồn tại vì một bài test đỏ, không vì một linh cảm. `try` lớn của
        `_stream_turn` bắt đầu **sau** hai khung đầu (`meta`, `sources`), nên
        một client ngắt kết nối ở đúng khoảng ấy làm `finally` — chỗ duy nhất
        gọi `resolve()` — không bao giờ chạy, và mọi người theo sau chờ đủ 15
        giây cho một request đã chết từ mili giây thứ nhất.

        ⚠️ Đây là **cùng một hình dạng** với `AU-11`: một cửa sổ hẹp giữa hai
        bước, vô hình với mọi bài test đi hết đường vui. Khác biệt duy nhất là
        lần này có bài test đi tìm nó (`test_a_client_that_disconnects...`).

        `_schedule_save` và `trace.finish` **cố ý** không được kéo lên đây:
        chúng thuộc về một lượt đã sinh ra chữ, và một request chết ở khung
        `meta` thì chưa có gì để lưu. Vé thì khác — nó là thứ **người khác**
        đang chờ.
        """
        try:
            # ⚠️ `aclosing`, không phải `async for` trần. Đóng generator ngoài
            # **không** đóng generator trong một cách dứt khoát — nó chờ GC — nên
            # `finally` của `_stream_turn` (chỗ ghi Postgres và đóng trace) chạy
            # muộn hoặc không chạy. Hai bài huỷ có sẵn từ `W4-06` đỏ ngay khi
            # lớp bọc này được thêm vào, và đó là cách nó bị bắt.
            async with aclosing(self._stream_turn(turn)) as inner:
                async for event in inner:
                    yield event
        finally:
            # Chạy **sau** `finally` của generator trong, nên `shared_answer`
            # đã được đặt nếu lượt ấy đi tới nơi.
            if turn.flight is not None:
                turn.flight.resolve(turn.shared_answer)

    async def _stream_turn(self, turn: ChatTurn) -> AsyncGenerator[ChatEvent, None]:
        """Từ đây trở đi mọi lỗi chỉ còn là một khung SSE.

        ⚠️ Kiểu trả về là `AsyncGenerator`, **không** phải `AsyncIterator`, và đó
        là một phần của hợp đồng chứ không phải một chi tiết: người gọi phải
        đóng được nó (`aclose`/`athrow`), vì hai đường huỷ ở §"Ngắt kết nối"
        chính là hai cách generator này kết thúc trong thực tế.

        ## ⚠️ `W5-06`: span ở đây mở/đóng **tường minh**, không bằng `with`

        Một async generator chạy trong context của người *tiêu thụ* nó, không
        có context riêng (PEP 568 chưa bao giờ được nhận). Nên một
        `with trace.span(...)` bọc quanh một `yield` sẽ đặt `ContextVar` cha rồi
        **trả quyền điều khiển về cho người gọi trong khi biến ấy vẫn đang
        đặt** — mọi span mà người gọi mở giữa hai lần `yield` sẽ mọc nhầm dưới
        span của chúng ta, và với SSE thì "giữa hai lần yield" là toàn bộ thời
        gian sinh chữ.

        Cây ở nửa dưới nông (mọi span đều là con trực tiếp của trace) nên mất
        `with` không mất gì; đổi lại `finally` là chỗ duy nhất đóng trace, và
        nó chạy trên cả ba đường thoát kể cả huỷ.
        """
        assert self.llm is not None  # `prepare()` đã kiểm; giữ mypy yên tâm
        if turn.cached is not None:
            # ⭐ `W4-10`: phát lại nguyên bộ khung từ cache. `meta.cache` nói RÕ
            # đây là câu trả lời của câu hỏi NÀO và giống bao nhiêu — một cache
            # hit sai (hai câu gần nhau nhưng khác đáp án) phải truy được từ
            # client, không phải chỉ từ log server.
            cached = turn.cached
            yield ChatEvent(
                "meta",
                {
                    "conversation_id": turn.conversation_id,
                    "message_id": turn.user_message_id,
                    "answer_message_id": turn.answer_message_id,
                    "trace_id": turn.trace.id,
                    "bundle_version": turn.bundle_version,
                    "model": cached.model,
                    "prompt": turn.prompt_spec(),
                    **turn.plan.as_meta(),
                    "cache": {
                        "hit": True,
                        "similarity": cached.similarity,
                        "matched_question": cached.question,
                    },
                },
            )
            yield ChatEvent("sources", {"sources": cached.sources})
            yield ChatEvent("delta", {"text": cached.text})
            if cached.citations_frame is not None:
                yield ChatEvent("citations", cached.citations_frame)
            elapsed = round((time.perf_counter() - turn.started) * 1000.0, 2)
            yield ChatEvent(
                "done",
                {
                    "finish_reason": "cache",
                    "model": cached.model,
                    "usage": {},
                    "ttfb_ms": elapsed,
                    "total_ms": elapsed,
                    "prepare_ms": prepare_ms_of(turn),
                    "language_mismatch": False,
                },
            )
            self._schedule_save(
                turn,
                cached.text,
                f"cache:{cached.model}",
                "cache",
                citations=cached.citations_frame,
            )
            replay = turn.trace.span("cache.replay", input=cached.question)
            replay.end(
                via="single_flight" if turn.cached_from_flight else "cache",
                output=cached.text,
                model=cached.model,
                similarity=cached.similarity,
                # Cùng con số mà khung `done` vừa khai — SLO TTFT đọc từ đây
                # (`MetricsSink`), và hai nguồn lệch nhau là thứ không truy được.
                ttfb_ms=elapsed,
                # ⚠️ **Không** khai `usage` cho lượt này. Một cache hit không
                # tốn token, nhưng nó cũng không phải "$0 cho câu hỏi này" —
                # câu trả lời ấy đã được trả tiền một lần ở lượt trước. Ghi 0
                # vào đây là chia tiền của lượt kia cho một lượt không gọi
                # model, và bảng chi phí sẽ tụt theo tỉ lệ trúng cache thay vì
                # theo giá thật.
                billed_here=False,
            )
            turn.trace.finish(output=cached.text, status="cache")
            return

        yield ChatEvent(
            "meta",
            {
                "conversation_id": turn.conversation_id,
                "message_id": turn.user_message_id,
                # ⭐ `W5-08`: khoá mà client dùng để chấm 👍/👎, có mặt từ khung
                # ĐẦU tiên. `trace_id` đi kèm để một người dùng báo lỗi đọc
                # được cùng một trace mà người vận hành mở trong Langfuse —
                # nhưng nó KHÔNG phải tham số của endpoint feedback: xem
                # `Message.trace_id`.
                "answer_message_id": turn.answer_message_id,
                "trace_id": turn.trace.id,
                "bundle_version": turn.bundle_version,
                "model": self.llm.model,
                "prompt": turn.prompt_spec(),
                **turn.plan.as_meta(),
            },
        )
        # `W4-12`: cờ tiêm tính MỘT lần rồi dùng lại cho cả khung và log —
        # gọi `sources()` hai lần là quét regex hai lần cho cùng một dữ liệu.
        sources = turn.sources()
        flagged = {s["n"]: s["flags"] for s in sources if s["flags"]}
        if flagged:
            # WARNING chứ không ERROR: một chunk bị gắn cờ là một điều đáng
            # nhìn, không phải một lỗi — và dương tính giả tồn tại.
            logger.warning(
                "nội dung nghi tiêm trong ngữ cảnh: %s (conversation %s)",
                flagged,
                turn.conversation_id,
            )
        yield ChatEvent("sources", {"sources": sources})

        if turn.plan.route == "clarify":
            # ⭐ Nhánh duy nhất **không** gọi model. Text lấy từ bảng trong mã,
            # nên nó tất định, miễn phí, và không thể sai ngôn ngữ đã phát hiện.
            #
            # Vẫn đi qua đúng bộ khung SSE (`delta` rồi `done`) chứ không phải
            # một dạng phản hồi riêng: client đã viết mã cho bốn khung ấy, và
            # thêm khung thứ năm cho một nhánh nội bộ là bắt mọi người tiêu thụ
            # phải biết về bộ phân loại câu hỏi.
            text = turn.plan.clarify_text()
            yield ChatEvent("delta", {"text": text})
            yield ChatEvent(
                "done",
                {
                    "finish_reason": "clarify",
                    "model": None,
                    "usage": {},
                    "ttfb_ms": round((time.perf_counter() - turn.started) * 1000.0, 2),
                    "total_ms": round((time.perf_counter() - turn.started) * 1000.0, 2),
                    "prepare_ms": prepare_ms_of(turn),
                    "language_mismatch": False,
                },
            )
            self._schedule_save(turn, text, "rule:clarify", "clarify", citations=None)
            turn.trace.finish(output=text, status="clarify")
            return

        parts: list[str] = []
        emitted: list[str] = []
        verified_frame: dict[str, Any] | None = None
        """Khung `citations` của `W4-09`, để `finally` ghi được nó xuống DB.

        ⚠️ Khai ở NGOÀI `try` chứ không dựa vào biến `report` bên trong: nhánh
        xác minh có điều kiện, nên `report` không tồn tại trên mọi đường thoát
        — và đường thoát nó không tồn tại (huỷ giữa chừng) đúng là đường mà
        `finally` chạy."""

        holdback = CitationHoldback()
        served_model = self.llm.model
        # ⭐ `W5-11`: model **được yêu cầu**, tách khỏi model đã phục vụ. Đây là
        # tín hiệu failover đúng — provider phân giải bí danh (`deepseek-chat` →
        # `deepseek-v4-flash`) làm hai giá trị lệch nhau một cách hoàn toàn hợp
        # lệ, nên so `served_model` với cấu hình sẽ tắt cache oan.
        requested_model = self.llm.model
        # ⭐ `"unknown"` chứ không phải `"client_disconnect"`, và đó là một lựa
        # chọn có chủ đích sau một phép tiêm lỗi **không** đỏ: nếu khởi tạo bằng
        # `"client_disconnect"` thì khối `except` bên dưới chỉ gán lại đúng giá
        # trị nó đã có — tức nó là chú thích chứ không phải hành vi, và xoá nó đi
        # không test nào thấy.
        #
        # Với `"unknown"`, hai đường huỷ *phải* tự khai tên mình, và một đường
        # thoát thứ ba xuất hiện trong tương lai sẽ được ghi lại là **không
        # biết** thay vì bị gán nhầm cho người dùng.
        finish_reason = "unknown"
        usage: dict[str, Any] = {}
        ttfb_ms: float | None = None

        prompt_span = turn.trace.span("prompt", prompt_spec=turn.prompt_spec())
        messages = turn.prompt()
        prompt_span.end(
            # `redact()` che `nonce` ở đây — xem `tracing.NONCE_MASK`. Đây là
            # span DUY NHẤT mang nguyên văn thứ đã gửi cho model, nên nó cũng là
            # chỗ duy nhất mã phiên của `W4-12` có thể rời khỏi tiến trình.
            # `TD-74`: message ngữ cảnh đi vào dưới dạng `content_parts` để trần
            # 4.000 ký tự của `redact()` áp cho TỪNG khối, không cho chuỗi ghép.
            output=turn.prompt_trace_view(messages),
            n_messages=len(messages),
            n_context_chunks=len(turn.contexts),
            prompt_chars=sum(len(m.content) for m in messages),
        )
        completion = turn.trace.span(
            "completion",
            kind="generation",
            input={"n_messages": len(messages)},
            temperature=0.0,
            max_tokens=turn.max_tokens,
        )
        try:
            async for chunk in self.llm.astream(
                messages,
                temperature=0.0,
                max_tokens=turn.max_tokens,
                extra_body=self.extra_body,
            ):
                if chunk.delta:
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - turn.started) * 1000.0
                    parts.append(chunk.delta)
                    # ⭐ Block `CITATIONS:` không được rò vào khung `delta`, kể
                    # cả khi marker bị cắt đôi giữa hai delta. `parts` giữ bản
                    # thô để parse; `emitted` là đúng những gì người dùng thấy —
                    # và cũng là bản được ghi vào Postgres, vì hai thứ đó lệch
                    # nhau là một bug không ai truy được từ log.
                    visible = holdback.feed(chunk.delta)
                    if visible:
                        # `append` TRƯỚC `yield`: generator có thể bị huỷ đúng
                        # tại điểm yield — sau khi khung đã rời đi. Append sau
                        # thì bản lưu thiếu đúng mẩu cuối người dùng đã thấy
                        # (hai test cancellation của `W4-06` bắt được điều này
                        # khi thứ tự bị viết ngược trong lúc làm `W4-09`).
                        emitted.append(visible)
                        yield ChatEvent("delta", {"text": visible})
                if chunk.final is not None:
                    served_model = chunk.final.model
                    requested_model = chunk.final.model_requested
                    finish_reason = chunk.final.finish_reason or "stop"
                    usage = {
                        "prompt_tokens": chunk.final.usage.prompt_tokens,
                        "completion_tokens": chunk.final.usage.completion_tokens,
                        "cost_usd": round(chunk.final.usage.cost_usd, 6),
                    }
                    # ⭐ `TD-47`: ghi nhận **chi phí thật**, không phải ước
                    # lượng — và chạy **nền**, cùng lý do với đường ghi Postgres
                    # và đường ghi cache. Một lượt đi Redis trên đường nóng chỉ
                    # để cập nhật một con số mà lượt này **đã** được phép tiêu
                    # là trả độ trễ cho một quyết định đã xong.
                    if self.daily_spend is not None and chunk.final.usage.cost_usd > 0:
                        ghi_no = asyncio.get_running_loop().create_task(
                            _ghi_chi_phi(
                                self.daily_spend,
                                turn.principal.tenant_id,
                                chunk.final.usage.cost_usd,
                            )
                        )
                        _PENDING.add(ghi_no)
                        ghi_no.add_done_callback(_PENDING.discard)
            tail = holdback.flush()
            if tail:
                emitted.append(tail)
                yield ChatEvent("delta", {"text": tail})
        except (asyncio.CancelledError, GeneratorExit):
            # Client đóng kết nối. Ném tiếp là **bắt buộc**: nuốt một
            # `CancelledError` làm hỏng cơ chế huỷ của cả vòng lặp sự kiện, và
            # nuốt một `GeneratorExit` cho `RuntimeError: async generator
            # ignored GeneratorExit`.
            finish_reason = "client_disconnect"
            # ⭐⭐ Đóng span **trước** khi ném tiếp, và ghi cả phần đã sinh.
            #
            # Cùng bài học với `_schedule_save` ở §"Ngắt kết nối": token đã trả
            # tiền rồi. Một hệ quan sát chỉ ghi lại những lượt người dùng ở lại
            # tới cuối là một hệ quan sát mù đúng chỗ đắt nhất — request bị bỏ
            # giữa chừng vẫn có hoá đơn, và một tỉ lệ bỏ cao là một tín hiệu
            # sản phẩm chứ không phải một khoảng trống trong log.
            completion.end(
                output="".join(emitted),
                level="WARNING",
                status="client_disconnect",
                model=served_model,
                chars_emitted=len("".join(emitted)),
            )
            raise
        except BudgetExceeded as exc:
            # Trần cạn **giữa** stream: phép kiểm ở `prepare()` chỉ hỏi "đã cạn
            # chưa", còn `astream` giữ chỗ theo ước lượng của chính lời gọi này.
            # Từ đây trở đi không còn HTTP status nào nữa, nên nó phải là một
            # khung `error` có tên riêng chứ không lặng lẽ dừng dòng token.
            finish_reason = "budget"
            logger.warning("hết ngân sách giữa lượt: %s", exc)
            completion.end(
                output="".join(emitted),
                level="ERROR",
                status=f"BudgetExceeded: {exc}",
                model=served_model,
            )
            yield ChatEvent(
                "error",
                {"detail": f"BudgetExceeded: {exc}", "partial_chars": len("".join(emitted))},
            )
        except LLMError as exc:
            finish_reason = "error"
            logger.warning("stream hỏng giữa chừng: %s", exc)
            completion.end(
                output="".join(emitted),
                level="ERROR",
                status=f"{type(exc).__name__}: {exc}",
                model=served_model,
            )
            # ⚠️ `NEW-08`/`AU-03`: lời của `LLMError` mang tên route, lỗi HTTP
            # thô của provider, có khi cả mẩu body — tất cả là chuyện nội bộ.
            # Client chỉ cần biết: tầng sinh hỏng, đã nhận được bao nhiêu chữ,
            # và `trace_id` để đối chiếu. Chi tiết đầy đủ nằm ở log (dòng
            # warning trên) và ở trace (status của span `completion`).
            yield ChatEvent(
                "error",
                {
                    "detail": "tầng sinh gặp lỗi giữa chừng — thử lại sau",
                    "partial_chars": len("".join(emitted)),
                    "trace_id": turn.trace.id,
                },
            )
        else:
            finish_reason = finish_reason if parts else "empty"
            completion.end(
                output="".join(emitted),
                model=served_model,
                # ⭐ `Usage` nhận `None` khi provider không khai, **không** nhận
                # 0. `usage` rỗng xảy ra thật: `finish_reason="empty"` của
                # `W4-08` (toàn bộ completion đi vào chuỗi suy luận) đến kèm một
                # `usage` có số, còn một stream đứt trước khung cuối thì không.
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    cost_usd=usage.get("cost_usd"),
                ),
                level="WARNING" if finish_reason == "empty" else None,
                status="model trả 0 ký tự" if finish_reason == "empty" else None,
                finish_reason=finish_reason,
                ttfb_ms=round(ttfb_ms, 2) if ttfb_ms is not None else None,
            )
            # ⭐⭐ Khung `sources` (đã đưa gì cho model) đã phát từ đầu; đây là
            # khung `citations` — model TUYÊN BỐ đã dùng gì, sau khi đối chiếu
            # từng quote với đúng chunk nó chỉ vào. Một quote bịa không thể là
            # một HTTP status nữa (đường phân giới `W4-06`): nó là
            # `verified: false` trong khung này, to và rõ, không im lặng.
            parsed = split_citation_block("".join(parts))
            if turn.plan.retrieves or parsed.block != "absent":
                cite_span = turn.trace.span("citations", n_sources=len(turn.contexts))
                report = verify_citations(parsed, [hit.chunk for hit in turn.contexts])
                claimed = len(report.citations) + len(report.invalid_ns)
                if report.block != "ok" or report.verified_count < claimed:
                    logger.warning(
                        "citations: block=%s verified=%d/%d (conversation %s)",
                        report.block,
                        report.verified_count,
                        claimed,
                        turn.conversation_id,
                    )
                cite_span.end(
                    output=report.as_frame(),
                    block=report.block,
                    verified=report.verified_count,
                    claimed=claimed,
                    # ⭐ Span này ở trace vì *kết quả*, không vì đồng hồ (đối
                    # chiếu quote là công việc thuần chuỗi, dưới một mili giây).
                    # `W4-09` là phép kiểm duy nhất chỉ báo được bằng một khung
                    # SSE — không có nó ở đây thì một citation bịa hiện ra với
                    # client mà **không** hiện ra với người vận hành.
                    level="WARNING" if report.verified_count < claimed else None,
                )
                verified_frame = report.as_frame()
                yield ChatEvent("citations", verified_frame)
            if (
                self.cache is not None
                # ⚠️ KHÔNG lặp lại `and self.generator` ở đây: phép tiêm `M2`
                # chứng minh nó không bao giờ đổi được kết quả — điều kiện
                # failover ngay dưới đã bao nó (`requested_model` không bao giờ
                # rỗng, nên `generator=""` tự chặn). Một điều kiện không thể
                # thay đổi hành vi là một chú thích viết bằng cú pháp `if`, và
                # dự án này đã học đúng bài ấy một lần ở `served_model` bên trên.
                and turn.cache_vector is not None
                and finish_reason == "stop"
                and emitted
                # ⭐ Chỉ ghi khi nhánh CHÍNH đã phục vụ. Một câu trả lời do
                # failover sinh ra là câu trả lời của một model khác; ghi nó vào
                # namespace của nhánh chính là để một sự cố năm phút biến thành
                # 24 giờ phát lại lời của nhà cung cấp dự phòng. Cùng lý lẽ với
                # `filters` ở `AU-02`: ca hiếm, xử bằng ĐIỀU KIỆN LOẠI.
                and requested_model == self.generator.split(":", 1)[-1]
            ):
                # Ghi cache là việc phụ — chạy nền như đường ghi Postgres, và
                # cùng lý do phải giữ tham chiếu mạnh (xem `_PENDING`).
                turn.shared_answer = CachedAnswer(
                    question=turn.plan.question,
                    text="".join(emitted),
                    sources=turn.sources(),
                    citations_frame=report.as_frame() if turn.plan.retrieves else None,
                    model=served_model,
                    # ⭐ `1.0`, không phải ngưỡng cache: người theo sau khớp
                    # **nguyên văn** (`flight_key`), không khớp theo cosine. Con
                    # số này đi thẳng ra khung `meta.cache.similarity` của
                    # client, nên nó phải nói đúng loại khớp đã xảy ra.
                    similarity=1.0,
                )
                store_task = asyncio.get_running_loop().create_task(
                    self.cache.store(
                        turn.principal.tenant_id,
                        cache_namespace(
                            turn.bundle_version, turn.resolved_top_k, self.generator, self.endpoint
                        ),
                        turn.plan.question,
                        turn.cache_vector,
                        text="".join(emitted),
                        sources=turn.sources(),
                        citations_frame=report.as_frame() if turn.plan.retrieves else None,
                        model=served_model,
                    )
                )
                _PENDING.add(store_task)
                store_task.add_done_callback(_PENDING.discard)
            answer_language = detect_language(parsed.text)
            # ⭐⭐ Chỗ chỉ dẫn ngôn ngữ trở thành một **con số**.
            #
            # `W4-06` đo được rằng luật 4 của prompt bị model bỏ qua (hỏi tiếng
            # Anh, đáp tiếng Việt), và `W4-07` không sửa được điều đó — một dòng
            # chỉ dẫn vẫn chỉ là một dòng chỉ dẫn. Cái đổi là từ đây thất bại ấy
            # **đếm được**: cả câu hỏi lẫn câu trả lời đều đi qua cùng một bộ
            # phát hiện, nên chênh lệch xuất hiện trong khung `done` và trong
            # log thay vì chỉ xuất hiện với người dùng.
            #
            # `unknown` ở bất kỳ bên nào ⇒ **không** báo lệch: không biết không
            # phải là biết-khác.
            mismatch = (
                turn.plan.language != "unknown"
                and answer_language != "unknown"
                and answer_language != turn.plan.language
            )
            if mismatch:
                logger.warning(
                    "câu trả lời lệch ngôn ngữ: hỏi %s, đáp %s (conversation %s)",
                    turn.plan.language,
                    answer_language,
                    turn.conversation_id,
                )
            yield ChatEvent(
                "done",
                {
                    "finish_reason": finish_reason,
                    "model": served_model,
                    "usage": usage,
                    "ttfb_ms": round(ttfb_ms, 2) if ttfb_ms is not None else None,
                    "total_ms": round((time.perf_counter() - turn.started) * 1000.0, 2),
                    "prepare_ms": prepare_ms_of(turn),
                    "language_mismatch": mismatch,
                },
            )
        finally:
            # ⚠️ **Đồng bộ, không `await`.** Xem §"Ngắt kết nối" ở docstring
            # module: ở đây có thể đang bị huỷ, và một `await` trong lúc bị huỷ
            # không chạy tới nơi.
            #
            # Ghi `emitted` chứ không phải `parts`: block CITATIONS là giao thức
            # giữa model và mã, không phải nội dung — lịch sử đọc lại từ DB phải
            # là đúng những gì người dùng đã thấy trên màn hình.
            self._schedule_save(
                turn, "".join(emitted), served_model, finish_reason, citations=verified_frame
            )
            # ⭐ `finish()` là idempotent và tự đóng mọi span còn mở, nên đây là
            # chỗ **duy nhất** cần đóng trace ở nửa dưới: nó chạy trên cả ba
            # đường thoát (xong, lỗi, huỷ) và không cần biết đường nào đã chạy.
            turn.trace.metadata["finish_reason"] = finish_reason
            turn.trace.finish(
                output="".join(emitted),
                level="ERROR" if finish_reason in {"error", "budget"} else None,
                status=finish_reason,
            )

    # ---------------------------------------------------------------- DB

    async def _history(
        self, principal: Principal, conversation_id: str | None
    ) -> list[ChatMessage]:
        if conversation_id is None:
            return []
        if self.sessions is None:
            # Không có kho thì không có hội thoại nào để tiếp — và nói ra chứ
            # không trả `[]`. Xem docstring của `sessions`.
            raise ConversationNotFound(
                f"service chạy không trạng thái nên không có hội thoại {conversation_id!r}"
            )
        async with atenant_session(self.sessions, principal.tenant_id) as session:
            exists = await session.scalar(
                select(Conversation.id).where(Conversation.id == conversation_id)
            )
            if exists is None:
                raise ConversationNotFound(f"không có hội thoại {conversation_id!r}")
            rows = (
                await session.scalars(
                    select(Message)
                    # `TD-78`: lọc hàng trợ lý rỗng (placeholder đang bay, lượt
                    # `empty`, crash giữa chừng) TRONG SQL — lọc ở Python thì
                    # mỗi hàng rỗng vẫn ăn mất một trong `MAX_HISTORY_MESSAGES`
                    # slot của prompt.
                    .where(
                        Message.conversation_id == conversation_id,
                        or_(Message.role != "assistant", Message.content != ""),
                    )
                    # Lấy **mới nhất** rồi đảo lại, chứ không `LIMIT` từ đầu:
                    # `ORDER BY created_at ASC LIMIT 10` cho 10 message **đầu
                    # tiên** của hội thoại, tức prompt càng ngày càng lạc đề khi
                    # cuộc trò chuyện dài ra — và nó vẫn trông như đang hoạt động.
                    .order_by(Message.created_at.desc(), Message.id.desc())
                    .limit(MAX_HISTORY_MESSAGES)
                )
            ).all()
        return [
            ChatMessage(role=row.role, content=row.content)  # type: ignore[arg-type]
            for row in reversed(rows)
            if row.role in ("user", "assistant") and row.content
        ]

    async def _open_turn(
        self,
        principal: Principal,
        conversation_id: str | None,
        plan: QueryPlan,
        bundle_version: str,
    ) -> tuple[str, str, str]:
        """Tạo hội thoại nếu cần, rồi ghi câu hỏi **và chỗ trống cho câu trả lời**.

        Ghi câu hỏi ở cuối lượt thì một lần crash giữa stream làm chính câu hỏi
        biến mất — người dùng tải lại trang và thấy câu mình vừa gõ không còn ở
        đâu cả. Mất câu trả lời thì họ hỏi lại được; mất câu hỏi thì lịch sử nói
        dối về chuyện đã xảy ra.

        ## ⭐⭐ `TD-78`: hàng trợ lý ghi Ở ĐÂY, `_save` chỉ còn điền vào

        Trước đây hàng trợ lý ra đời trong task nền SAU khi stream xong — nên
        `answer_message_id` trong khung `meta` là một **lời hứa**: feedback tới
        trước khi task chạy nhận 404, và một lượt model im lặng (`_save` bỏ qua
        text rỗng) làm lời hứa ấy **không bao giờ** thành sự thật — đúng lượt
        đáng nhận 👎 nhất.

        Ghi placeholder (`content=""`, `finish_reason="pending"`) trong CÙNG
        transaction với câu hỏi đóng cả hai lỗ một lần: id là hàng thật từ trước
        khung SSE đầu tiên, và lượt rỗng giữ nguyên hàng với
        `finish_reason="empty"` do `_save` điền — 👎 có chỗ để trỏ vào. Đường
        đọc (`_history`, `load_history`) lọc `content == ""` nên quyết định của
        `W4-06` — *"một hàng rỗng trong lịch sử tệ hơn không có hàng nào"* —
        vẫn đúng ở phía người đọc; hàng tồn tại cho feedback, không cho hiển thị.

        ⚠️ Crash giữa stream để lại `finish_reason="pending"` vĩnh viễn — cố ý:
        đó là sự thật ("lượt này chưa từng kết thúc"), nó vô hình với người đọc
        nhờ bộ lọc trên, và một job dọn dẹp cho nó là máy móc cho một ca chỉ
        xảy ra khi tiến trình chết giữa chừng.
        """
        if self.sessions is None:
            # Id vẫn phát ra: khung `meta` và `answer_message_id` là hợp đồng
            # với client, không phải hệ quả của việc có Postgres. Chúng chỉ
            # không trỏ tới hàng nào.
            return conversation_id or str(uuid.uuid4()), str(uuid.uuid4()), uuid.uuid4().hex
        async with atenant_session(self.sessions, principal.tenant_id) as session:
            if conversation_id is None:
                conversation = Conversation(
                    tenant_id=principal.tenant_id,
                    title=plan.original[:120],
                    bundle_version=bundle_version,
                )
                session.add(conversation)
                await session.flush()
                conversation_id = conversation.id
            message = Message(
                tenant_id=principal.tenant_id,
                conversation_id=conversation_id,
                role="user",
                # Chuỗi người dùng **thật sự gõ**. Ghi bản viết lại vào đây thì
                # lượt sau đọc lịch sử ra một câu hỏi không ai hỏi, và bước viết
                # lại của lượt sau sẽ dựa trên đó — sai số cộng dồn qua từng lượt.
                content=plan.original,
                route=plan.route,
                rewritten_query=plan.question if plan.rewritten else None,
            )
            session.add(message)
            await session.flush()
            message_id = message.id
            answer_id = uuid.uuid4().hex
            session.add(
                Message(
                    id=answer_id,
                    tenant_id=principal.tenant_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content="",
                    # Khoá nối `NEW-08`/`AU-07` đặt ngay lúc sinh: ở đây nó là
                    # một sự thật hiển nhiên (câu hỏi vừa flush ở trên), còn ở
                    # `_save` nó phải đi vòng qua `ChatTurn`.
                    user_message_id=message_id,
                    finish_reason="pending",
                )
            )
            await session.commit()
        return conversation_id, message_id, answer_id

    def _schedule_save(
        self,
        turn: ChatTurn,
        text: str,
        model: str,
        finish_reason: str,
        *,
        citations: dict[str, Any] | None,
    ) -> None:
        task = asyncio.create_task(self._save(turn, text, model, finish_reason, citations))
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)

    async def _save(
        self,
        turn: ChatTurn,
        text: str,
        model: str,
        finish_reason: str,
        citations: dict[str, Any] | None,
    ) -> None:
        if self.sessions is None:
            return
        # ⭐⭐ `TD-78`: UPDATE, không INSERT — hàng đã tồn tại từ `_open_turn`,
        # trong cùng transaction với câu hỏi. Text rỗng KHÔNG return sớm nữa:
        # lượt model im lặng điền `finish_reason="empty"` vào placeholder, và
        # đó chính là hàng mà một cú 👎 cần để tồn tại. Đường đọc lọc
        # `content == ""` nên hàng rỗng vẫn không hiện ra trong lịch sử.
        try:
            async with atenant_session(self.sessions, turn.principal.tenant_id) as session:
                row = await session.get(Message, turn.answer_message_id)
                if row is None:
                    # Hội thoại bị xoá giữa lượt (script GDPR, CASCADE dọn cả
                    # placeholder) — không có gì để điền, và tái tạo hàng cho
                    # một hội thoại đã xoá là chống lại chính lệnh xoá ấy.
                    logger.warning(
                        "placeholder %s không còn (hội thoại %s đã xoá giữa lượt?)",
                        turn.answer_message_id,
                        turn.conversation_id,
                    )
                    return
                row.content = text
                # `W5-08`: hai cột, không một. `retrieved_sources` là cái đã
                # đưa vào; `citations_verified` là cái model nói nó đã dùng,
                # sau khi đối chiếu. Một câu 👎 chỉ phân loại được khi có cả hai.
                row.retrieved_sources = turn.persisted_sources()
                row.citations_verified = citations
                row.trace_id = turn.trace.id
                row.latency_ms = int((time.perf_counter() - turn.started) * 1000.0)
                row.model = model
                row.finish_reason = finish_reason
                # ⚠️ `created_at` đặt lại LÚC ĐIỀN, không giữ mốc placeholder:
                # hai hàng chèn cùng transaction ở `_open_turn` nhận CÙNG một
                # `now()` (Postgres: transaction start), nên thứ tự
                # `(created_at, id)` giữa câu hỏi và câu trả lời rơi xuống so
                # id ngẫu nhiên — `assistant` có thể đứng TRƯỚC `user`. Mốc
                # "lúc điền" cũng chính là ngữ nghĩa cũ (hàng INSERT ở task nền
                # sau khi stream xong), nên người đọc lịch sử không thấy khác.
                row.created_at = func.now()
                await session.commit()
        except Exception:
            # Task này chạy ngoài request; một exception ở đây không có ai bắt
            # và `asyncio` chỉ in nó lúc GC dọn task — tức nó lạc khỏi
            # `request_id` và khỏi mọi dòng access.
            logger.exception(
                "ghi message trợ lý thất bại (%s, %s)", turn.conversation_id, finish_reason
            )


HISTORY_PAGE = 50
"""Số message mặc định cho một trang lịch sử — `AU-08`, vá ở `W6-06`.

Không phải `MAX_HISTORY_MESSAGES` (10): đó là ngân sách **prompt**, còn đây là
ngân sách **payload**. Trộn hai con số ấy làm một là buộc người đọc lại hội thoại
chỉ thấy đúng phần mà model nhìn thấy.
"""


async def load_history(
    sessions: async_sessionmaker[AsyncSession],
    principal: Principal,
    conversation_id: str,
    *,
    limit: int = HISTORY_PAGE,
    after: str | None = None,
) -> list[dict[str, Any]]:
    """Đọc lại một hội thoại — thứ chứng minh câu DoD "sống sót qua restart".

    ## ⭐⭐ `AU-08`: con trỏ theo **id**, không theo offset

    `OFFSET n` phải đếm qua n hàng mỗi lần, và tệ hơn: một hàng mới chèn vào giữa
    hai lần gọi làm mọi trang sau đó **trượt đi một** — người đọc mất đúng một
    message và không có gì nói ra. Ở đây hàng mới *luôn* được chèn (task nền ghi
    câu trả lời sau khi stream xong), nên đó không phải một ca hiếm.

    `after` là `Message.id` của phần tử cuối trang trước; thứ tự sắp theo
    `(created_at, id)` nên khoá so sánh cũng phải là cặp ấy — so mỗi `created_at`
    thì hai message cùng mốc mili giây làm mất một cái.

    ⚠️ `after` trỏ vào một message **không tồn tại** (hoặc của hội thoại khác) là
    `ValueError`, không phải một trang rỗng: trang rỗng đọc y hệt "hết lịch sử".
    """
    async with atenant_session(sessions, principal.tenant_id) as session:
        exists = await session.scalar(
            select(Conversation.id).where(Conversation.id == conversation_id)
        )
        if exists is None:
            raise ConversationNotFound(f"không có hội thoại {conversation_id!r}")
        # `TD-78`: hàng trợ lý rỗng tồn tại CHO feedback, không cho hiển thị —
        # quyết định `W4-06` ("một hàng rỗng tệ hơn không có hàng nào") giữ
        # nguyên ở phía người đọc. Lọc trong SQL chứ không sau khi `limit`:
        # client đọc "trang ngắn hơn `limit`" là "hết lịch sử", nên một trang
        # bị lọc hụt ở Python sẽ cắt cụt lịch sử ngay giữa chừng, không dấu vết.
        query = select(Message).where(
            Message.conversation_id == conversation_id,
            or_(Message.role != "assistant", Message.content != ""),
        )
        if after is not None:
            anchor = (
                await session.execute(
                    select(Message.created_at, Message.id).where(
                        Message.id == after, Message.conversation_id == conversation_id
                    )
                )
            ).one_or_none()
            if anchor is None:
                raise HistoryCursorNotFound(
                    f"không có message {after!r} trong hội thoại {conversation_id!r}"
                )
            query = query.where(
                tuple_(Message.created_at, Message.id) > tuple_(anchor[0], anchor[1])
            )
        rows: Sequence[Message] = (
            await session.scalars(query.order_by(Message.created_at, Message.id).limit(limit))
        ).all()
    return [
        {
            "id": row.id,
            "role": row.role,
            "content": row.content,
            "created_at": row.created_at.isoformat(),
            "sources": row.retrieved_sources,
            "citations": row.citations_verified,
            "trace_id": row.trace_id,
            "model": row.model,
            "finish_reason": row.finish_reason,
            "latency_ms": row.latency_ms,
            "route": row.route,
            "rewritten_query": row.rewritten_query,
        }
        for row in rows
    ]
