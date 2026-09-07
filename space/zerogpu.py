"""Mối nối duy nhất giữa hệ thống này và ZeroGPU. `W6-02`.

## ⭐⭐ Vì sao chỉ có MỘT mối nối

Space không được phép cài lại truy hồi. Một bản cài thứ hai của cùng logic là
họ `AU-12`, và ở đây nó tệ hơn mọi lần trước: demo sẽ trình diễn một hệ thống
*khác* hệ thống có số đo, mà không có gì đỏ để nói ra điều đó. `qdrant-client`
local mode đo được là chạy nguyên đường code hiện tại
(`probes/w602-local-mode.json`, 9/9), nên `rag_core` đi lên Space **nguyên vẹn**,
cài từ `git+…@<sha>`.

Thứ duy nhất phải đổi là *chỗ* việc chạm GPU xảy ra. ZeroGPU chỉ gắn GPU thật
vào trong thân một hàm `@spaces.GPU`. Nên có đúng một lớp bọc, đặt ở đúng ranh
giới ấy: `Retriever.retrieve` — trong đó có embed truy vấn (GPU), tìm hybrid
(CPU), và cross-encoder (GPU).

## ⭐⭐ MỘT lần vào GPU mỗi câu hỏi, không phải hai

Bọc hẹp hơn (chỉ embedder và chỉ reranker) tiết kiệm được ~730 ms thời gian
GPU-đang-gắn mỗi câu — đó là phần quét sparse thuần Python của local mode
(`probes/w602-local-latency.json`: sparse 764 ms, dense 77 ms). Nhưng nó đổi
lấy **hai** lần vào hàng đợi cấp node mỗi câu, và lần thứ hai có thể **không
lấy được suất GPU sau khi lần thứ nhất đã lấy được** — tức một câu hỏi hỏng
giữa chừng, một chế độ hỏng mới không tồn tại ở bản một-lần-vào.

Phép tính hạn mức nói cách bọc rộng vẫn đủ: khách chưa đăng nhập có 2 phút
GPU/ngày, mỗi câu ~1,5 s ⇒ ~80 câu. Đổi 730 ms lấy việc không có câu hỏi nào
chết giữa chừng là đổi đúng chiều.

## ⭐⭐ `TD-72` bị đảo ngược ở đây

`TD-72` vá lượt lạnh bằng cách chạy **một lượt truy hồi thật** ngay sau khi dựng
runtime (13.386 → 4.427 ms). Trên ZeroGPU đó chính là thứ **cấm**: compute CUDA
ngoài `@spaces.GPU` không có GPU thật đằng sau. Nhưng nền tảng đòi nửa còn lại
của cùng một ý tưởng — tài liệu HF: *"models must be placed on `cuda` at the root
module level… Lazy-loading or moving models to CUDA inside `@spaces.GPU` is
discouraged"*.

Nên hai nửa tách ra: `materialise_weights()` **nạp trọng số lên cuda** ở module
scope (hợp lệ — ZeroGPU giả lập CUDA ngoài hàm và gói trọng số ra đĩa), còn lượt
làm nóng có compute thì **tắt** (`warmup=False`). Cùng một mục tiêu, nền tảng
đảo ngược nửa nào được phép.

⚠️ Và trọng số ở `rag_core` nạp **lười**: `HuggingFaceEmbeddingProvider.model` và
`CrossEncoderReranker.model` đều là property bọc `lru_cache`. Không có
`materialise_weights()` thì 2,2 GB + 1,5 GB trọng số nạp bên trong lời gọi
`@spaces.GPU` đầu tiên — tính vào hạn mức GPU của người dùng đầu tiên, và đúng
kiểu chậm mà tài liệu HF vừa dặn tránh.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import spaces

from rag_core.retrieval.base import Retriever

if TYPE_CHECKING:
    from rag_core.bundle import RagBundle
    from rag_core.reranking.base import Reranker
    from rag_core.retrieval.filters import FilterSpec
    from rag_core.schemas import RetrievedChunk

__all__ = ["ZeroGpuRetriever", "ZeroGpuRuntimeBuilder", "materialise_weights"]

logger = logging.getLogger(__name__)

#: Trần thời gian khai báo cho mỗi lời gọi GPU, giây.
#:
#: ⚠️⚠️ Con số này **không** chỉ là một timeout. Nền tảng đối chiếu nó với hạn
#: mức CÒN LẠI của khách trước khi cho chạy, nên khai thừa sẽ chặn người dùng
#: sớm hơn hẳn: khách chưa đăng nhập có 120 s, khai `duration=60` ⇒ chỉ **2**
#: câu là bị từ chối với 60 s hạn mức chưa dùng. Khai sát thực tế (đo được
#: ~1,5 s) cho ~80 câu. Nó cũng là thứ xếp hạng ưu tiên trong hàng đợi.
GPU_DURATION_S = int(os.environ.get("SPACE_GPU_DURATION_S", "20"))

_INNER: Retriever | None = None
"""Retriever thật, đặt một lần lúc khởi động.

Biến module chứ không tham số: đối số của hàm `@spaces.GPU` đi qua pickle giữa
hai tiến trình, và một `Retriever` mang theo model 2,2 GB thì không pickle được
— cũng không nên. Tiến trình worker fork từ tiến trình chính nên nó **đã có**
biến này; chỉ câu hỏi và kết quả đi qua ranh giới.
"""


@spaces.GPU(duration=GPU_DURATION_S)
def _retrieve_on_gpu(
    query: str, top_k: int, filters: Any, precomputed: Any = None
) -> list[RetrievedChunk]:
    """Thân duy nhất chạy với GPU thật gắn vào.

    Trả `list[RetrievedChunk]` — pydantic, pickle được, và **không** phải tensor
    CUDA: unpickle một tensor CUDA ở tiến trình chính kích `torch.cuda._lazy_init`
    mà ZeroGPU chặn. `RetrievedChunk` chỉ mang chữ và số float.

    `precomputed` chỉ truyền xuống **khi có**, cùng luật với `TracedRetriever`
    và `RerankedRetriever`: nhánh nền có thể là một retriever không nhận kwarg
    ấy, và truyền `None` tường minh vào đó là `TypeError`.
    """
    if _INNER is None:  # pragma: no cover - lỗi lắp ráp, không phải lỗi runtime
        raise RuntimeError("ZeroGpuRetriever chưa được nối với retriever thật")
    kwargs: dict[str, Any] = {"filters": filters}
    if precomputed is not None:
        kwargs["precomputed"] = precomputed
    return _INNER.retrieve(query, top_k, **kwargs)


class ZeroGpuRetriever(Retriever):
    """Chuyển `retrieve()` vào một suất GPU của ZeroGPU. Không đổi kết quả.

    ⭐ `name` giữ **nguyên** của retriever bên trong, có chủ ý. Quy ước của
    `TD-38` là: `name` gom đúng những cần điều khiển **làm đổi con số**. Lớp này
    đổi *chỗ chạy*, không đổi phép tính — cùng lý lẽ đã giữ `tenant_id` ra ngoài
    `name` ở `QdrantDenseRetriever`. Đổi tên ở đây sẽ làm mọi bundle đã ký hỏng
    chữ ký vì một lý do không liên quan tới chất lượng truy hồi.

    ## ⚠️⚠️ `_inner` + `__getattr__`, và bản đầu của tôi làm ngược lại

    Bản đầu đặt tên thuộc tính là `target` và **từ chối** `__getattr__`, với lý
    lẽ nghe hợp lý: một lớp bọc bắt-tất-cả *trông như* có mọi method của mọi
    retriever. Lượt chạy thử đầu tiên bác bỏ nó — `materialise_weights` trả về
    `{"reranker": True}`, **không có embedder**, vì `embedder_of()` đào chuỗi
    `retriever.base.store.embeddings` bằng duck-typing và dừng lại ở lớp bọc.
    Hệ quả trên ZeroGPU: 2,2 GB trọng số BGE-M3 nạp bên trong lời gọi
    `@spaces.GPU` **đầu tiên**, tính vào hạn mức của người dùng đầu tiên, đúng
    thứ tài liệu HF vừa dặn tránh — và không có gì đỏ, chỉ có một dict trông ổn.

    `serving/core/instrument.py` đã gặp đúng bài này và ghi lại kết luận:
    *"`__getattr__` uỷ quyền là bắt buộc chứ không phải tiện tay… một lớp bọc
    không uỷ quyền sẽ làm semantic cache tắt lặng lẽ"*. Tôi đảo một quyết định
    đã có mà không đọc nó. Nên lớp này theo đúng quy ước ấy: tên `_inner`, uỷ
    quyền qua `__getattr__`, và **nhận `precomputed`** như `TracedRetriever`.

    ⚠️ Cái giá đi kèm, viết ra thay vì né: tên `_inner` khiến `_unwrap_traced`
    bóc được lớp này ⇒ `wants_precomputed` trả True ⇒ `ChatService` sẽ embed câu
    hỏi **ở tiến trình chính**, ngoài `@spaces.GPU`. Hôm nay không xảy ra vì cả
    nhánh ấy nằm sau `cache is not None` và Space chạy `cache=None`. Ai bật
    cache cho Space phải đọc dòng này trước.
    """

    def __init__(self, inner: Retriever) -> None:
        global _INNER
        _INNER = inner
        self._inner = inner
        self.name = inner.name

    def __getattr__(self, item: str) -> Any:
        # `object.__getattribute__` chứ không `self._inner`: lúc unpickle/copy,
        # `_inner` chưa tồn tại và `self._inner` sẽ rơi lại vào `__getattr__` —
        # đệ quy vô hạn. Cùng cái bẫy `_delegate()` của `instrument.py` ghim.
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError:
            raise AttributeError(item) from None
        return getattr(inner, item)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: FilterSpec = None,
        precomputed: Any | None = None,
    ) -> list[RetrievedChunk]:
        return _retrieve_on_gpu(query, top_k, filters, precomputed)


def materialise_weights(embedder: Any, reranker: Reranker | None) -> dict[str, bool]:
    """Ép trọng số nạp lên `cuda` **ở module scope**, không chạy một forward nào.

    Xem docstring module: đây là nửa được phép của `TD-72` trên ZeroGPU. Chạm
    property là đủ — `lru_cache` bên dưới nạp và `.to(device)`.

    ⭐ **Embedder vắng mặt là lỗi lắp ráp, reranker vắng mặt thì không.** Một
    bundle có thể khai `rerank: null` một cách hợp lệ; không bundle nào chạy
    được mà thiếu embedder. Bản đầu gộp cả hai vào một `if holder is None:
    continue`, nên lần lắp sai đầu tiên trả về một dict *trông ổn*
    (`{"reranker": True}`) thay vì một tiếng nổ. Phân biệt hai ca chính là
    phần việc của hàm này.

    Nạp **hỏng** thì vẫn ghi log và đi tiếp, cùng lý lẽ với
    `BundleRegistry._warm`: một tối ưu độ trễ không được phép biến thành
    "Space không lên được". Không tìm *thấy* embedder thì khác — đó là lắp sai.
    """
    if embedder is None:
        raise RuntimeError(
            "không đào được embedder ra khỏi retriever — `embedder_of()` trả None. "
            "Gần như chắc chắn là một lớp bọc không uỷ quyền `__getattr__`; xem "
            "docstring `ZeroGpuRetriever`."
        )
    done: dict[str, bool] = {}
    for label, holder in (("embedder", embedder), ("reranker", reranker)):
        if holder is None:
            continue
        try:
            getattr(holder, "model", None)
            if label == "embedder":
                getattr(holder, "sparse_head", None)
            done[label] = True
        except Exception:
            logger.warning("không nạp trước được trọng số %s", label, exc_info=True)
            done[label] = False
    return done


class ZeroGpuRuntimeBuilder:
    """Bọc `QdrantRuntimeBuilder`, thêm mối nối GPU **sau** mọi phép kiểm.

    Thứ tự quan trọng: builder thật chạy `verify_schema`, `_check_size` và
    `_check_identity` trên retriever **chưa bọc**, nên `TD-38` vẫn so đúng chuỗi
    `…@cuda:L512:float16:n50` với thứ bundle đã ký. Bọc trước sẽ làm phép so ấy
    nhìn vào một object khác.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, bundle: RagBundle) -> tuple[Retriever, Reranker | None]:
        retriever, reranker = self.inner(bundle)
        return ZeroGpuRetriever(retriever), reranker
