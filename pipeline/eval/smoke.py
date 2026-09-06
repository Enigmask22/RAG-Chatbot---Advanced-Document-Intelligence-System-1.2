"""Smoke eval truy hồi cho CI — chạy trên index đóng băng, không model, $0.

`W5-09`. Đây là phép đo duy nhất trong dự án **cố ý không dùng model nào**, và
lý do là một ràng buộc chứ không phải một tối ưu.

## ⭐⭐ Một cổng chặn PR phải tất định, miễn phí, và chạy được cho fork

Ba tính chất ấy loại hết mọi thiết kế hiển nhiên:

* **Gọi LLM mỗi PR** — `TD-41` đo được rằng `temp=0` ở DeepSeek **không tất
  định**, nên cổng sẽ đỏ vì lý do không nằm trong diff. Cộng thêm: tốn tiền mỗi
  lần đẩy commit, và PR từ fork **không** thấy secret nên cổng biến mất đúng lúc
  cần nó nhất.
* **Chạy BGE-M3 trên runner** — 2,2 GB trọng số, CPU, cho một phép đo mà kết quả
  đã biết trước là hằng số (cùng model, cùng text ⇒ cùng vector).
* **Dựng lại index 15.814 chunk** — corpus nằm ở DVC remote, và runner không có
  quyền.

Cái còn lại: **đóng băng vector**. Fixture mang sẵn vector dense + sparse của
300 chunk và của 30 câu hỏi, đo bằng chính model thật ở máy có GPU
(`smoke_fixture.py`). CI nạp chúng vào một Qdrant rỗng rồi chạy **đúng lớp
`QdrantHybridRetriever` của production** lên trên.

Cái được giữ lại là phần hay hỏng vì một diff: hợp nhất RRF, `candidate_k`,
trọng số nhánh, filter, schema collection, thứ tự payload. Cái bị bỏ đi là phần
không hỏng vì một diff: chất lượng của model.

## ⚠️ Con số ở đây KHÔNG so được với con số của eval đêm

300 chunk, không rerank. `recall@10` trên 300 chunk và `recall@10` trên 15.814
chunk là **hai đại lượng khác nhau mang cùng một cái tên** — đúng cái bẫy đã làm
gate của `W5-05` cho qua một hệ thống vượt ngân sách 34%. Nên mọi metric ở đây
mang tiền tố `smoke_`, và `SMOKE_PREFIX` có test ghim.

Cổng này trả lời *"diff có làm tụt truy hồi không"*, không trả lời *"hệ thống
truy hồi tốt đến đâu"*. Câu thứ hai thuộc `W5-10`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rag_core.bundle import MANIFEST_NAME, POINTER_NAME, bundle_dir_name, read_pointer
from rag_core.embedding.base import EmbeddingProvider, FloatArray, HybridVectors
from rag_core.embedding.sparse import SparseVector
from rag_core.schemas import Chunk, DocumentMetadata

from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BUNDLE_ROOT",
    "SMOKE_PREFIX",
    "FrozenEmbedder",
    "SmokeFixture",
    "SmokeQuery",
    "SmokeResult",
    "UnknownTextError",
    "default_bundle",
    "load_fixture",
    "retrieval_options",
    "run_smoke",
]

SMOKE_PREFIX = "smoke_"
"""Tiền tố bắt buộc cho mọi metric của module này.

⚠️ Không phải mỹ quan. Xem docstring module: cùng công thức, cùng tên, khác tập
— và `W5-05` đã trả giá một lần cho đúng lỗi ấy (`p95_latency_ms` vs
`p95_end_to_end_ms`). Tiền tố làm cho một con số smoke lọt vào bảng eval trở
thành thứ nhìn thấy được.
"""

FIXTURE_VERSION = 1
DEFAULT_TOP_K = 10
DEFAULT_BUNDLE_ROOT = Path("bundles")


def default_bundle(root: Path = DEFAULT_BUNDLE_ROOT) -> Path:
    """Manifest của bundle **đang được trỏ tới**, không phải một version gõ tay.

    ## ⭐⭐ `AU-12`: một cổng mang bản sao thứ hai của "đang chạy bản nào"

    Bản `W5-09` viết `DEFAULT_BUNDLE = Path("bundles/rag-bundle-v0.2.1/...")`,
    còn CI lẫn Makefile không truyền `--bundle`. Hằng số ấy là bản sao thứ hai
    của một sự thật đã có chỗ ở: bundle nào đang phục vụ. Bump lên `0.3.0` mà
    quên sửa nó thì cổng PR vẫn xanh — trong khi thứ nó gác là `retrieval.options`
    của **bản cũ**, tức đúng loại thay đổi mà `retrieval_options` sinh ra để bắt.

    Bug ở tầng meta: nó không làm hỏng một truy vấn nào, nó làm hỏng **niềm tin
    vào cổng**. Và nó không thể tự lộ ra, vì triệu chứng của nó là màu xanh.

    Sửa bằng cách bỏ bản sao đi: hỏi `bundles/CURRENT`, đúng con trỏ mà
    `serving.api.app._startup_version` hỏi. Hai bên lệch nhau không còn là một
    khả năng.
    """
    version = read_pointer(root)
    if version is None:
        raise FileNotFoundError(
            f"không có con trỏ {root / POINTER_NAME}. Smoke phải gác đúng bundle "
            "đang phục vụ, nên nó không đoán: tạo con trỏ, hoặc truyền `--bundle` "
            "tường minh nếu đang cố tình chấm một bản khác."
        )
    return root / bundle_dir_name(version) / MANIFEST_NAME


def retrieval_options(manifest_path: Path) -> dict[str, Any]:
    """Tham số nhánh hybrid, đọc từ **manifest bundle đang phục vụ**.

    ## ⭐ Cổng phải gác cả cấu hình, không chỉ mã

    Bản đầu ghim `k=1, candidate_k=20, weights=(1.0, 0.25)` thẳng trong file
    này. Nó gác được mọi thay đổi *mã* của tầng truy hồi — và mù hoàn toàn với
    một PR đổi `components.retrieval.options` trong bundle, tức đúng loại thay
    đổi dễ làm nhất và khó thấy nhất. Một cổng mang bản sao thứ hai của cấu
    hình production sẽ gác một hệ thống không tồn tại kể từ lần hai bản sao
    lệch nhau.

    `weights` về `tuple` chứ không giữ `list`: `QdrantHybridRetriever` đưa nó
    vào `name`, và một `list` cho ra chuỗi khác — tức baseline không khớp vì
    một kiểu dữ liệu, không vì một phép đo.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    options = dict(manifest["components"]["retrieval"]["options"])
    if isinstance(options.get("weights"), list):
        options["weights"] = tuple(options["weights"])
    return options


class UnknownTextError(KeyError):
    """`FrozenEmbedder` nhận một chuỗi không có trong fixture."""


@dataclass(frozen=True)
class SmokeQuery:
    query_id: str
    query: str
    category: str
    lang: str
    relevant_chunk_ids: tuple[str, ...]
    """Nhãn đã **phân giải từ span** lúc dựng fixture, trên index đầy đủ.

    ⭐ Phân giải ở lúc dựng chứ không lúc chạy: `resolve_queries` cần **mọi**
    chunk của tài liệu liên quan (~264 chunk/tài liệu ở corpus này), tức fixture
    sẽ phình từ 300 lên vài nghìn chunk chỉ để tính lại một thứ không đổi.
    """


@dataclass
class SmokeFixture:
    version: int
    built_at: str
    source_collection: str
    embedding_model: str
    dimension: int
    queries: list[SmokeQuery]
    chunks: list[Chunk]
    query_vectors: dict[str, dict[str, Any]]
    chunk_vectors: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if self.version != FIXTURE_VERSION:
            raise ValueError(
                f"fixture phiên bản {self.version}, mã này đọc {FIXTURE_VERSION} — "
                "dựng lại bằng `python -m pipeline.eval.smoke_fixture`"
            )


# --------------------------------------------------------------- embedder giả


class FrozenEmbedder(EmbeddingProvider):
    """Tra vector theo **nguyên văn** chuỗi. Không biết ⇒ ném.

    ## ⭐⭐ Ném chứ không trả vector 0

    Một embedder giả trả `zeros(1024)` cho chuỗi lạ vẫn chạy trót lọt: Qdrant
    nhận, tìm ra một thứ gì đó, metric ra một con số. Con số ấy đo **không gì
    cả**, và nó sẽ nằm trên bảng cạnh những con số thật.

    Ném là cách duy nhất để smoke không lặng lẽ đổi thành một phép đo khác khi
    ai đó thêm một câu hỏi, sửa một khoảng trắng, hoặc bật một bước viết lại
    truy vấn ở giữa đường.
    """

    def __init__(
        self,
        vectors: dict[str, dict[str, Any]],
        *,
        dimension: int,
        model_name: str,
        sparse_vocab_size: int = 250_002,
    ) -> None:
        self._vectors = vectors
        self._dimension = dimension
        self.name = f"frozen[{model_name}]"
        self._sparse_vocab_size = sparse_vocab_size
        self.seen: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def sparse_vocab_size(self) -> int | None:
        return self._sparse_vocab_size

    def _lookup(self, text: str) -> dict[str, Any]:
        stored = self._vectors.get(text)
        if stored is None:
            raise UnknownTextError(
                f"fixture không có vector cho {text[:60]!r}. Smoke chỉ chạy trên "
                "tập chuỗi đã đóng băng — dựng lại fixture nếu câu hỏi đổi."
            )
        self.seen.append(text)
        return stored

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        rows = [np.asarray(self._lookup(t)["dense"], dtype=np.float32) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self._dimension), dtype=np.float32)

    def embed_documents_hybrid(self, texts: Sequence[str]) -> HybridVectors:
        dense = self.embed_documents(texts)
        sparse = [_sparse_of(self._vectors[t]) for t in texts]
        return HybridVectors(dense=dense, sparse=sparse)

    def embed_query_hybrid(self, text: str) -> tuple[FloatArray, SparseVector]:
        stored = self._lookup(text)
        dense = np.asarray(stored["dense"], dtype=np.float32)
        return dense, _sparse_of(stored)


def _sparse_of(stored: dict[str, Any]) -> SparseVector:
    return SparseVector(
        indices=tuple(int(i) for i in stored["sparse"]["indices"]),
        values=tuple(float(v) for v in stored["sparse"]["values"]),
    )


# ------------------------------------------------------------------ đọc file


def load_fixture(path: Path) -> SmokeFixture:
    """Đọc `*.jsonl.gz` (text + nhãn) cộng `*.npz` (vector) đứng cạnh nó."""
    meta_path = Path(path)
    npz_path = meta_path.with_name(meta_path.name.replace(".jsonl.gz", ".npz"))
    if not npz_path.exists():
        raise FileNotFoundError(f"thiếu file vector {npz_path}")

    header: dict[str, Any] | None = None
    queries: list[SmokeQuery] = []
    chunks: list[Chunk] = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            kind = row.pop("kind")
            if kind == "header":
                header = row
            elif kind == "query":
                queries.append(
                    SmokeQuery(
                        query_id=row["query_id"],
                        query=row["query"],
                        category=row["category"],
                        lang=row["lang"],
                        relevant_chunk_ids=tuple(row["relevant_chunk_ids"]),
                    )
                )
            elif kind == "chunk":
                meta = row.pop("metadata", None)
                chunks.append(Chunk(**row, metadata=DocumentMetadata(**meta) if meta else None))
            else:  # pragma: no cover - file hỏng
                raise ValueError(f"dòng lạ trong fixture: {kind!r}")
    if header is None:
        raise ValueError(f"{meta_path} không có dòng header")

    with np.load(npz_path, allow_pickle=False) as data:
        query_vectors = _unpack(data, "query")
        chunk_vectors = _unpack(data, "chunk")

    return SmokeFixture(
        version=int(header["version"]),
        built_at=str(header["built_at"]),
        source_collection=str(header["source_collection"]),
        embedding_model=str(header["embedding_model"]),
        dimension=int(header["dimension"]),
        queries=queries,
        chunks=chunks,
        query_vectors={q.query: query_vectors[q.query_id] for q in queries},
        chunk_vectors=chunk_vectors,
    )


def _unpack(data: Any, prefix: str) -> dict[str, dict[str, Any]]:
    """CSR phẳng → dict theo id. Một mảng float16 cho dense, ba mảng cho sparse."""
    ids = [str(x) for x in data[f"{prefix}_ids"]]
    dense = data[f"{prefix}_dense"].astype(np.float32)
    offsets = data[f"{prefix}_sparse_offsets"]
    indices = data[f"{prefix}_sparse_indices"]
    values = data[f"{prefix}_sparse_values"].astype(np.float32)
    out: dict[str, dict[str, Any]] = {}
    for n, key in enumerate(ids):
        lo, hi = int(offsets[n]), int(offsets[n + 1])
        out[key] = {
            "dense": dense[n],
            "sparse": {"indices": indices[lo:hi].tolist(), "values": values[lo:hi].tolist()},
        }
    return out


# -------------------------------------------------------------------- chạy


@dataclass
class SmokeResult:
    metrics: dict[str, float]
    n_queries: int
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"n_queries": self.n_queries, "metrics": self.metrics, "per_query": self.per_query}


def seed_collection(store: Any, fixture: SmokeFixture, *, recreate: bool = True) -> int:
    """Dựng collection rỗng rồi ghi thẳng vector đóng băng vào."""
    store.ensure_collection(recreate=recreate)
    store.ensure_payload_indexes()
    return int(store.upsert_precomputed(fixture.chunks, fixture.chunk_vectors))


def run_smoke(retriever: Any, fixture: SmokeFixture, *, top_k: int = DEFAULT_TOP_K) -> SmokeResult:
    """Truy hồi từng câu rồi chấm. Không chạm mạng ngoài Qdrant cục bộ."""
    per_query: dict[str, dict[str, float]] = {}
    for query in fixture.queries:
        if not query.relevant_chunk_ids:
            # ⚠️ Không lặng lẽ bỏ qua: ba metric ở đây đều trả `None` khi không
            # có nhãn, và một `None` lọt vào phép trung bình sẽ thành `TypeError`
            # ở giữa CI thay vì một lời từ chối ở đây. Câu `unanswerable` bị loại
            # ngay lúc dựng fixture — xem `smoke_fixture.py`.
            raise ValueError(f"câu {query.query_id} không có nhãn; fixture dựng sai")
        hits = retriever.retrieve(query.query, top_k=top_k)
        got = [hit.chunk.chunk_id for hit in hits]
        relevant = set(query.relevant_chunk_ids)
        scores = {
            f"{SMOKE_PREFIX}recall@{top_k}": recall_at_k(got, relevant, top_k),
            f"{SMOKE_PREFIX}ndcg@{top_k}": ndcg_at_k(got, relevant, top_k),
            f"{SMOKE_PREFIX}mrr": reciprocal_rank(got, relevant),
        }
        per_query[query.query_id] = {k: float(v) for k, v in scores.items() if v is not None}
    names = sorted({name for scores in per_query.values() for name in scores})
    metrics = {
        name: round(sum(scores[name] for scores in per_query.values()) / max(1, len(per_query)), 6)
        for name in names
    }
    return SmokeResult(metrics=metrics, n_queries=len(per_query), per_query=per_query)


# ------------------------------------------------------------------- ngưỡng


def _plain_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """JSON không có `tuple`, nên so hai bên phải cùng một hình dạng."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in sorted(options.items())}


def compare_to_baseline(
    result: SmokeResult,
    baseline: dict[str, Any],
    *,
    tolerance: float,
    options: dict[str, Any] | None = None,
) -> list[str]:
    """Danh sách vi phạm. Rỗng = qua.

    ⭐ So bằng **dung sai tuyệt đối**, không bằng kiểm định thống kê như
    `W2-09`/`W5-05`. Ở đây không có ngẫu nhiên nào để kiểm định: cùng vector,
    cùng index, cùng truy vấn ⇒ chênh lệch duy nhất còn lại là HNSW xấp xỉ. Một
    p-value trên một đại lượng gần như tất định là một con số trông có thẩm
    quyền mà không nói gì.
    """
    failures: list[str] = []
    expected = baseline.get("metrics", {})
    if options is not None and _plain_options(options) != baseline.get("retrieval_options"):
        # ⭐ So **bộ tham số**, không so `retriever.name`: cái tên mang cả tên
        # collection, thứ khác nhau giữa CI (`rag_smoke_ci`) và test
        # (`rag_smoke_pytest`) — nó trộn *chỗ chạy* với *cách chạy*, và cổng chỉ
        # quan tâm cái sau. Một thay đổi cấu hình đi qua cổng thành một dòng nói
        # rõ trường nào đổi, thay vì thành "vài phần nghìn dao động".
        failures.append(
            f"tham số truy hồi đổi: {_plain_options(options)} vs "
            f"{baseline.get('retrieval_options')} trong baseline"
        )
    if result.n_queries != baseline.get("n_queries"):
        failures.append(
            f"số câu đổi: {result.n_queries} vs {baseline.get('n_queries')} trong baseline — "
            "fixture và baseline phải dựng cùng lúc"
        )
    for name, want in sorted(expected.items()):
        got = result.metrics.get(name)
        if got is None:
            failures.append(f"thiếu metric {name} (baseline có)")
            continue
        if got < want - tolerance:
            failures.append(f"{name}: {got:.4f} < {want:.4f} − {tolerance} (baseline)")
    for name in sorted(set(result.metrics) - set(expected)):
        # Metric mới không làm đỏ, nhưng phải nói ra: một baseline thiếu dòng là
        # một cổng không gác dòng ấy.
        logger.warning("metric %s chưa có trong baseline — không được gác", name)
    return failures


# ---------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.eval.smoke",
        description="Smoke eval truy hồi trên index đóng băng (W5-09).",
    )
    parser.add_argument("--fixture", type=Path, default=Path("data/eval/smoke/smoke_v1.jsonl.gz"))
    parser.add_argument("--baseline", type=Path, default=Path("data/eval/smoke/baseline.json"))
    parser.add_argument("--collection", default="rag_smoke_ci")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help=(
            "manifest bundle để lấy tham số nhánh hybrid — xem `retrieval_options`. "
            f"Mặc định: bundle mà `bundles/{POINTER_NAME}` trỏ tới."
        ),
    )
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="mức tụt tuyệt đối còn chấp nhận cho mỗi metric",
    )
    parser.add_argument("--out", type=Path, default=None, help="ghi kết quả JSON")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="ghi ĐÈ baseline bằng kết quả lần này. Chỉ dùng khi đã đọc diff.",
    )
    args = parser.parse_args(argv)

    from rag_core.retrieval import QdrantHybridRetriever
    from rag_core.retrieval.qdrant_store import QdrantDenseRetriever
    from rag_core.settings import get_settings

    fixture = load_fixture(args.fixture)
    logger.info(
        "fixture %s: %d câu · %d chunk · model %s · dựng %s",
        args.fixture,
        len(fixture.queries),
        len(fixture.chunks),
        fixture.embedding_model,
        fixture.built_at,
    )

    embedder = FrozenEmbedder(
        {**fixture.query_vectors},
        dimension=fixture.dimension,
        model_name=fixture.embedding_model,
    )
    url = args.qdrant_url or get_settings().qdrant_url
    store = QdrantDenseRetriever(embeddings=embedder, collection=args.collection, url=url)
    written = seed_collection(store, fixture)
    logger.info("đã ghi %d point vào %s", written, args.collection)

    bundle = args.bundle if args.bundle is not None else default_bundle()
    options = retrieval_options(bundle)
    logger.info("tham số truy hồi đọc từ %s: %s", bundle, options)
    retriever = QdrantHybridRetriever(store, **options)
    result = run_smoke(retriever, fixture, top_k=args.top_k)
    for name, value in sorted(result.metrics.items()):
        logger.info("%-22s %.4f", name, value)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.write_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(
                {
                    "fixture": args.fixture.name,
                    "built_at": fixture.built_at,
                    "top_k": args.top_k,
                    "retriever": retriever.name,
                    "retrieval_options": _plain_options(options),
                    "n_queries": result.n_queries,
                    "metrics": result.metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info("đã ghi baseline → %s", args.baseline)
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    failures = compare_to_baseline(result, baseline, tolerance=args.tolerance, options=options)
    if failures:
        for line in failures:
            logger.error("TỤT: %s", line)
        logger.error(
            "smoke eval ĐỎ. Nếu đây là thay đổi có chủ đích, chạy lại với "
            "`--write-baseline` và giải thích trong PR."
        )
        return 1
    logger.info("smoke eval XANH (dung sai %.3f)", args.tolerance)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
