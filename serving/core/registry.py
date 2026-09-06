"""`W4-02` — đổi bundle đang phục vụ mà không restart tiến trình, và lùi lại được.

DoD viết ba câu, và câu ở giữa là câu khó: *"request đang chạy không bị lỗi"*.
Hai câu kia là việc gọi hàm; câu ấy là một tính chất về **thời điểm**, và nó
hỏng theo những cách không lộ ra ở bài test một luồng.

## Ba luật, mỗi luật chặn một cách hỏng

### 1. Dựng xong rồi mới đổi — không bao giờ đổi rồi mới dựng

Nạp bundle mới là: đọc manifest → kiểm chữ ký → dựng retriever → nối Qdrant →
nạp reranker. Bất kỳ bước nào cũng hỏng được (manifest sai, collection không
tồn tại, GPU hết chỗ). Nếu gỡ bundle cũ ra trước thì một lần reload lỗi biến
`/chat` từ *"đang phục vụ bản cũ"* thành *"không phục vụ gì"* — tức một thao tác
nhằm cải thiện hệ thống lại là thao tác làm sập nó.

Nên `activate()` dựng **toàn bộ** runtime mới trước, và phép gán tham chiếu là
việc cuối cùng. Lỗi ở bất kỳ đâu trước đó thì bản cũ **không hề bị chạm tới**.

### 2. Request cầm một ảnh chụp, không cầm một tham chiếu tới "bản hiện tại"

`registry.active.retriever` gọi hai lần trong cùng một request có thể trả về hai
runtime khác nhau nếu có reload chen vào giữa — và khi đó câu trả lời trích dẫn
chunk của index này bằng điểm số của index kia. Không có gì đỏ.

Cách chặn không phải là khoá, mà là **kiểu dữ liệu**: `active` trả về một
`ActiveBundle` bất biến, và người gọi giữ nó suốt request. Phép đổi là một lần
gán một thuộc tính — nguyên tử dưới GIL — nên ảnh chụp đã phát ra không bao giờ
đổi dưới chân người cầm.

### 3. Không đóng runtime cũ khi đổi

Cám dỗ là gọi `close()` trên retriever cũ để thu hồi kết nối. Làm thế là **kéo
đổ đúng những request mà luật 2 vừa bảo vệ**. Runtime cũ sống tới khi không ai
tham chiếu nữa và GC dọn.

⚠️ Cái giá: giữ bản trước để rollback nghĩa là giữ **hai** runtime cùng lúc, và
một cross-encoder là 2,2 GB. Chỗ này được cứu bởi việc phần lớn hai bundle liên
tiếp **dùng chung model** — `RuntimeBuilder` chia sẻ instance theo danh tính
model, nên bản trước thường tốn gần như 0. Khi hai bundle **khác** model thì bộ
nhớ nhân đôi thật, và đó là lý do lịch sử chỉ giữ **một** bản.

## Vì sao rollback không phải là "reload bản cũ"

Reload-từ-đĩa có thể hỏng: đĩa đổi, mạng rớt, GPU hết chỗ. Một cơ chế rollback
chỉ có ích khi mọi thứ đang hỏng, nên **nó không được phép hỏng**. Vì thế
`rollback()` kích hoạt lại **chính object runtime** đã chạy trước đó, không dựng
lại gì cả — và đó là lý do luật 3 tồn tại.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from rag_core.bundle import RagBundle, load_bundle, parse_semver
from rag_core.reranking.base import Reranker
from rag_core.retrieval.base import Retriever

__all__ = [
    "WARMUP_QUERY",
    "ActiveBundle",
    "BundleRegistry",
    "NoBundleLoadedError",
    "NothingToRollBackError",
    "RuntimeBuilder",
]

logger = logging.getLogger(__name__)


class NoBundleLoadedError(RuntimeError):
    """Chưa có bundle nào được kích hoạt.

    Tách riêng khỏi mọi lỗi khác vì `W4-03` ánh xạ đúng lỗi này thành `/ready`
    **503** — "chưa sẵn sàng" là một trạng thái khởi động bình thường, không phải
    một sự cố.
    """


class NothingToRollBackError(RuntimeError):
    """`rollback()` khi chưa có bản nào trước đó.

    Cố ý **ném** thay vì trả về `None` hay lặng lẽ không làm gì: người gọi
    rollback đang tin rằng hệ thống vừa quay về trạng thái trước, và một no-op im
    lặng để họ tin nhầm đúng lúc đang xử lý sự cố.
    """


class RuntimeBuilder(Protocol):
    """Dựng phần chạy được từ một bundle.

    Là một Protocol để `BundleRegistry` không phải biết gì về Qdrant, torch hay
    sentence-transformers — nhờ đó toàn bộ logic đổi/lùi ở đây test được **không
    cần hạ tầng**, và những cách hỏng đáng sợ nhất (dựng lỗi, đổi giữa chừng)
    kiểm được bằng một builder giả năm dòng.
    """

    def __call__(self, bundle: RagBundle) -> tuple[Retriever, Reranker | None]: ...


@dataclass(frozen=True)
class ActiveBundle:
    """Ảnh chụp bất biến của một bundle **và** runtime của nó.

    Cả hai đi cùng nhau là điểm chính: một request cầm `retriever` mà đọc
    `bundle` từ chỗ khác thì hai thứ có thể lệch phiên bản. Đóng gói chung khiến
    việc đó không viết ra được.
    """

    bundle: RagBundle
    retriever: Retriever
    reranker: Reranker | None
    loaded_at: datetime

    @property
    def version(self) -> str:
        return self.bundle.bundle_version


#: Truy vấn dùng để làm nóng. Ngắn, không dấu chấm hỏi, không khớp gì đặc biệt —
#: nó chỉ cần đi hết đường embed → Qdrant → rerank một lần.
WARMUP_QUERY = "khoi dong"


@dataclass
class BundleRegistry:
    """Giữ bundle đang phục vụ, đổi được lúc chạy, lùi lại được một bước."""

    root: Path
    build_runtime: RuntimeBuilder
    warmup: bool = True
    """Chạy một lượt truy hồi giả ngay sau khi dựng runtime. Xem `_warm`."""
    _active: ActiveBundle | None = field(default=None, init=False, repr=False)
    _previous: ActiveBundle | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ------------------------------------------------------------------ đọc

    @property
    def active(self) -> ActiveBundle:
        """Ảnh chụp đang phục vụ. Người gọi **giữ nó suốt request**.

        Gọi lại trong cùng một request là tự mở ra khả năng lấy hai runtime khác
        nhau cho một câu trả lời.
        """
        snapshot = self._active
        if snapshot is None:
            raise NoBundleLoadedError(
                "chưa có bundle nào được kích hoạt. Gọi `activate()` lúc khởi "
                "động; trước đó `/ready` phải trả 503."
            )
        return snapshot

    @property
    def is_ready(self) -> bool:
        return self._active is not None

    @property
    def previous(self) -> ActiveBundle | None:
        return self._previous

    def status(self) -> dict[str, str | None]:
        """Đủ để một lệnh `curl` trả lời "đang chạy bản nào, lùi được về đâu"."""
        active, previous = self._active, self._previous
        return {
            "active": active.version if active else None,
            "active_since": active.loaded_at.isoformat() if active else None,
            "rollback_to": previous.version if previous else None,
        }

    # ------------------------------------------------------------------ ghi

    def activate(self, version: str) -> ActiveBundle:
        """Nạp bundle từ đĩa và đổi sang nó. Hỏng ở bất kỳ đâu ⇒ bản cũ nguyên vẹn.

        Khoá bao **cả** phần dựng chứ không chỉ phép gán: hai lần reload chồng
        nhau sẽ dựng hai runtime rồi ghi đè lẫn nhau, và bản thua cuộc trở thành
        một runtime không ai tham chiếu nhưng vẫn giữ GPU.
        """
        parse_semver(version)  # từ chối sớm, trước khi chạm đĩa
        with self._lock:
            bundle = load_bundle(self.root / f"rag-bundle-v{version}")
            # ⚠️ Mọi thứ có thể hỏng phải xảy ra TRƯỚC dòng gán bên dưới.
            retriever, reranker = self.build_runtime(bundle)
            if self.warmup:
                self._warm(retriever, version)
            snapshot = ActiveBundle(
                bundle=bundle,
                retriever=retriever,
                reranker=reranker,
                loaded_at=datetime.now(UTC),
            )
            return self._swap(snapshot)

    def _warm(self, retriever: Retriever, version: str) -> None:
        """Một lượt truy hồi giả trước khi bundle được công bố. `TD-72`.

        ## ⭐⭐ Chỗ đúng là `activate`, không phải `lifespan`

        `W5-06` phát hiện rerank lạnh tốn 10,7× rerank nóng, và `W6-05` đo lại
        trên hệ đang phục vụ: request đầu **13.386 ms** so với 4.272 ms
        (+9,1 s, 3,1×), TTFT **10.469 ms** so với 1.389 ms (7,5×). Toàn bộ phần
        phạt nằm ở `prepare` (9.620 ms so với 740 ms) — trọng số đã nạp lúc
        `build_runtime`, nhưng kernel CUDA chỉ khởi tạo ở `score()` đầu tiên.

        Phản xạ đầu là làm nóng trong `lifespan`. Nhưng chế độ hỏng không thuộc
        về *khởi động*, nó thuộc về **kích hoạt**: `POST /admin/bundle/reload`
        dựng một runtime mới với trọng số mới, và người dùng ngay sau lệnh
        reload nhận đúng 13 giây ấy — không có deploy nào để đổ lỗi. Đặt ở đây
        thì cả hai đường đi qua cùng một lượt làm nóng.

        Và nó tự động khiến `/ready` đúng: startup gọi `activate()` bên trong
        `lifespan`, mà uvicorn chưa mở cổng cho tới khi `lifespan` xong.

        ## ⭐ Làm nóng hỏng thì **không** được chặn kích hoạt

        Qdrant chưa lên khi container khởi động là chuyện thường. Một bản vá
        latency mà biến sự cố tạm thời của phụ thuộc thành "không deploy được"
        đã đổi một vấn đề nhỏ lấy một vấn đề lớn hơn. Hỏng ⇒ ghi log, đi tiếp,
        và request đầu tiên trả giá đúng như trước khi có bản vá này.

        ⚠️ Cái giá: thời gian khởi động dài thêm đúng một lượt truy hồi lạnh
        (~9 s trên máy đo). Đó là đánh đổi có chủ đích — nó chuyển 9 giây từ
        *người dùng đầu tiên* sang *quy trình deploy*, nơi không có ai đang đợi.
        `warmup=False` tắt được cho môi trường mà thời gian lên là ràng buộc.
        """
        started = time.perf_counter()
        try:
            retriever.retrieve(WARMUP_QUERY, top_k=1)
        except Exception:
            logger.warning(
                "làm nóng bundle %s thất bại — kích hoạt vẫn tiếp tục, "
                "request đầu tiên sẽ trả giá khởi tạo kernel (TD-72)",
                version,
                exc_info=True,
            )
            return
        logger.info(
            "làm nóng bundle %s xong sau %.0f ms", version, (time.perf_counter() - started) * 1000.0
        )

    def rollback(self) -> ActiveBundle:
        """Quay về bản trước. Không dựng lại gì, nên không hỏng được.

        Sau khi lùi, bản vừa bị bỏ trở thành "bản trước" — nên gọi hai lần là
        quay lại chỗ cũ. Đó là hành vi có chủ đích: lịch sử sâu một bậc, và một
        lệnh rollback luôn có nghĩa xác định thay vì phụ thuộc vào đã gọi bao
        nhiêu lần.
        """
        with self._lock:
            target = self._previous
            if target is None:
                raise NothingToRollBackError(
                    "không có bản nào để lùi về — mới chỉ kích hoạt một bundle "
                    "kể từ khi tiến trình khởi động."
                )
            logger.warning(
                "rollback bundle %s → %s",
                self._active.version if self._active else "(none)",
                target.version,
            )
            return self._swap(target)

    def _swap(self, snapshot: ActiveBundle) -> ActiveBundle:
        """Phép đổi thật. Một lần gán, và **không** đóng runtime cũ.

        Đóng nó ở đây sẽ kéo đổ đúng những request mà `ActiveBundle` bất biến vừa
        bảo vệ: chúng đang cầm ảnh chụp cũ và vẫn dùng chính retriever đó.
        Runtime cũ sống tới khi request cuối cầm nó kết thúc, rồi GC dọn.
        """
        outgoing = self._active
        # ⚠️ So **version**, không so danh tính object. Nạp lại cùng một version
        # (sau khi Qdrant rớt rồi lên lại) dựng ra một `ActiveBundle` MỚI, nên so
        # bằng `is` sẽ đẩy chính nó vào lịch sử — và `rollback()` sau đó "lùi" về
        # đúng bản đang chạy trong khi người vận hành tin là đã lùi.
        if outgoing is not None and outgoing.version != snapshot.version:
            self._previous = outgoing
        self._active = snapshot
        logger.info(
            "bundle đang phục vụ: %s (trước đó: %s)",
            snapshot.version,
            self._previous.version if self._previous else "(none)",
        )
        return snapshot
