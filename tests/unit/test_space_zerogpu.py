"""`W6-02` — mối nối ZeroGPU, và cái bẫy mà lượt chạy thử đầu tiên bắt được.

Bản đầu của `ZeroGpuRetriever` đặt tên thuộc tính là `target` và từ chối
`__getattr__`. Hậu quả: `embedder_of()` — thứ đào `retriever.base.store.embeddings`
bằng duck-typing — dừng lại ở lớp bọc và trả `None`, nên `materialise_weights`
báo `{"reranker": True}` và **2,2 GB trọng số embedder không được nạp trước**.
Trên ZeroGPU đó là 2,2 GB nạp bên trong lời gọi GPU đầu tiên, tính vào hạn mức
của người dùng đầu tiên. Không có gì đỏ; chỉ có một dict trông ổn.

Nên bộ test này không hỏi "lớp bọc có gọi được không". Nó hỏi **lớp bọc có tàng
hình trước những thứ đào xuyên qua nó không**, và **hàm nạp trước có phân biệt
được "không có reranker" với "lắp sai" không**.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from rag_core.retrieval.base import Retriever
from rag_core.schemas import Chunk, DocumentMetadata, RetrievalMode, RetrievedChunk
from serving.core.semantic_cache import embedder_of
from zerogpu import ZeroGpuRetriever, ZeroGpuRuntimeBuilder, materialise_weights

if TYPE_CHECKING:
    from rag_core.bundle import RagBundle

META = DocumentMetadata(source_url="https://example.org/x", license="CC BY 3.0 IGO")


def _hit(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"c{n}", doc_id="d", content="x", chunk_index=n, metadata=META),
        score=1.0 / n,
        rank=n,
        mode=RetrievalMode.HYBRID,
    )


class _Embedder:
    name = "gia-lap"

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _Store:
    embeddings = _Embedder()


class _Base(Retriever):
    """Đóng vai `QdrantHybridRetriever`: có `.store`, nhận `precomputed`."""

    name = "nhanh-nen"
    store = _Store()

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self, query: str, top_k: int = 10, *, filters: Any = None, precomputed: Any = None
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": filters, "precomputed": precomputed}
        )
        return [_hit(1), _hit(2)]


class _Reranked(Retriever):
    """Đóng vai `RerankedRetriever`: giữ nhánh nền ở `.base`, không có `.store`."""

    def __init__(self, base: Retriever) -> None:
        self.base = base
        self.name = f"reranked[{base.name}]:model@cuda:L512:float16:n50"
        self.verified = 0

    def retrieve(
        self, query: str, top_k: int = 10, *, filters: Any = None, precomputed: Any = None
    ) -> list[RetrievedChunk]:
        return self.base.retrieve(  # type: ignore[call-arg]
            query, top_k, filters=filters, precomputed=precomputed
        )

    def verify_schema(self) -> None:
        self.verified += 1


@pytest.fixture
def wrapped() -> tuple[ZeroGpuRetriever, _Base, _Reranked]:
    base = _Base()
    inner = _Reranked(base)
    return ZeroGpuRetriever(inner), base, inner


class TestLopBocPhaiTangHinhTruocThuDaoXuyenQuaNo:
    def test_embedder_of_van_tim_thay_embedder_qua_lop_boc(
        self, wrapped: tuple[ZeroGpuRetriever, _Base, _Reranked]
    ) -> None:
        """⭐⭐ Đây là bài bắt được lỗi thật. Xem docstring module."""
        retriever, base, _ = wrapped
        assert embedder_of(retriever) is base.store.embeddings

    def test_khong_boc_thi_embedder_of_cung_tim_thay_nen_bai_tren_do_dung_thu(
        self, wrapped: tuple[ZeroGpuRetriever, _Base, _Reranked]
    ) -> None:
        """Nhóm chứng: nếu `embedder_of` vốn đã trả None thì bài trên vô nghĩa."""
        _, base, inner = wrapped
        assert embedder_of(inner) is base.store.embeddings

    def test_verify_schema_di_toi_duoc_retriever_that(
        self, wrapped: tuple[ZeroGpuRetriever, _Base, _Reranked]
    ) -> None:
        retriever, _, inner = wrapped
        retriever.verify_schema()
        assert inner.verified == 1

    def test_thuoc_tinh_khong_ton_tai_van_la_AttributeError_chu_khong_de_quy(self) -> None:
        """⚠️ `self._inner` trong `__getattr__` là đệ quy vô hạn khi `_inner`
        chưa có — cái bẫy mà `_delegate()` của `instrument.py` đã ghim."""
        orphan = ZeroGpuRetriever.__new__(ZeroGpuRetriever)
        with pytest.raises(AttributeError):
            orphan.khong_co_thuoc_tinh_nay  # noqa: B018


class TestTenKhongDuocDoi:
    def test_name_giu_nguyen_cua_retriever_ben_trong(
        self, wrapped: tuple[ZeroGpuRetriever, _Base, _Reranked]
    ) -> None:
        """`TD-38`: `name` gom thứ **làm đổi con số**. Chạy ở đâu thì không."""
        retriever, _, inner = wrapped
        assert retriever.name == inner.name
        assert "zerogpu" not in retriever.name.lower()
        assert "space" not in retriever.name.lower()


class TestChuyenTiepThamSo:
    def test_query_top_k_filters_di_qua_nguyen_ven(
        self, wrapped: tuple[ZeroGpuRetriever, _Base, _Reranked]
    ) -> None:
        retriever, base, _ = wrapped
        retriever.retrieve("câu hỏi", 7, filters={"lang": "vi"})
        assert base.calls[-1]["query"] == "câu hỏi"
        assert base.calls[-1]["top_k"] == 7
        assert base.calls[-1]["filters"] == {"lang": "vi"}

    def test_precomputed_di_qua_khi_nguoi_goi_co_dua(
        self, wrapped: tuple[ZeroGpuRetriever, _Base, _Reranked]
    ) -> None:
        retriever, base, _ = wrapped
        retriever.retrieve("q", 3, precomputed=([0.1], "sparse"))
        assert base.calls[-1]["precomputed"] == ([0.1], "sparse")

    def test_KHONG_truyen_precomputed_xuong_nhanh_khong_nhan_kwarg_ay(self) -> None:
        """⭐⭐ Bài này thay một bài **không thể đỏ**.

        Bản đầu kiểm `base.calls[-1]["precomputed"] is None` khi người gọi
        không đưa gì — và giá trị ấy là `None` dù lớp bọc có truyền tường minh
        hay không, vì nhánh nền giả có sẵn `precomputed=None` trong chữ ký.
        Phép tiêm `M15` (luôn truyền `precomputed`) **sống sót**.

        Luật thật là: nhánh nền có thể **không có** kwarg ấy. Nên nhánh nền ở
        đây cũng không có, và bài test đỏ bằng `TypeError`.
        """

        class _KhongNhan(Retriever):
            name = "nhanh-nen-cu"
            store = _Store()

            def retrieve(
                self, query: str, top_k: int = 10, *, filters: Any = None
            ) -> list[RetrievedChunk]:
                return [_hit(1)]

        retriever = ZeroGpuRetriever(_KhongNhan())
        assert retriever.retrieve("q", 3), "không được truyền precomputed=None xuống đây"

    def test_khong_nhan_precomputed_thi_duong_cache_se_no_chu_khong_lang_le(self) -> None:
        """Nhóm chứng cho bài trên: chữ ký phải **có** `precomputed`.

        `TracedRetriever` bỏ sót đúng chỗ này một lần (`NEW-08`), và triệu chứng
        là đường embed-một-lần chết ở lớp ngoài cùng của production trong khi
        mọi unit test trên class trần vẫn xanh.
        """
        import inspect

        assert "precomputed" in inspect.signature(ZeroGpuRetriever.retrieve).parameters


class TestNapTrongSoPhanBietHaiCaVangMat:
    def test_khong_dao_ra_duoc_embedder_thi_NO(self) -> None:
        """⭐ Lắp sai, không phải cấu hình hợp lệ. Bản đầu trả về một dict."""
        with pytest.raises(RuntimeError, match="embedder_of"):
            materialise_weights(None, None)

    def test_bundle_khong_co_reranker_la_hop_le(self) -> None:
        done = materialise_weights(_Embedder(), None)
        assert done == {"embedder": True}

    def test_ca_hai_co_mat_thi_ca_hai_duoc_cham(self) -> None:
        class _Rr:
            model = "da-nap"

        done = materialise_weights(_Embedder(), _Rr())  # type: ignore[arg-type]
        assert done == {"embedder": True, "reranker": True}

    def test_nap_HONG_thi_ghi_log_va_di_tiep_chu_khong_giet_Space(self) -> None:
        """Cùng lý lẽ `BundleRegistry._warm`: một tối ưu độ trễ không được biến
        thành 'không lên được'."""

        class _Vo:
            @property
            def model(self) -> Any:
                raise OSError("đĩa hỏng")

        done = materialise_weights(_Embedder(), _Vo())  # type: ignore[arg-type]
        assert done == {"embedder": True, "reranker": False}


class TestThuTuBocQuanTrong:
    def test_builder_that_chay_TRUOC_roi_moi_boc(self) -> None:
        """⭐ `_check_identity` (`TD-38`) phải nhìn vào retriever **chưa bọc**."""
        seen: list[str] = []

        class _Inner:
            def __call__(self, bundle: Any) -> tuple[Retriever, Any]:
                inner = _Reranked(_Base())
                seen.append(type(inner).__name__)
                return inner, None

        retriever, reranker = ZeroGpuRuntimeBuilder(_Inner())(cast("RagBundle", object()))
        assert seen == ["_Reranked"], "builder thật phải thấy retriever trần"
        assert isinstance(retriever, ZeroGpuRetriever)
        assert reranker is None
