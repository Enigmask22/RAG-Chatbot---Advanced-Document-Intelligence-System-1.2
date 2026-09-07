"""`qdrant-client` local mode có chạy nổi **nguyên** đường truy hồi hiện tại không?

Đây là phép đo đứng trước toàn bộ thiết kế của `W6-02`. HF Space chạy Gradio SDK
— không Docker, không Qdrant server. Nếu câu trả lời là *không*, Space buộc phải
cài lại truy hồi bằng numpy, tức một bản sao thứ hai của logic (`AU-12`) và một
demo trình diễn hệ thống *khác* hệ thống có số đo. Nếu là *có*, `rag_core` đi
lên nguyên vẹn.

Chín bước dưới đây là **mọi** lời gọi Qdrant mà đường serving thật đi qua, kể
cả `query_batch_points` (hai nhánh trong một request, `W2-04`) và filter theo
tenant (`W2-06`/`TD-40`) — không phải một `search()` tượng trưng.

Chạy: `uv run python scripts/probe_local_mode.py`
"""

from __future__ import annotations

import json
import traceback
import warnings
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from rag_core.embedding.hashing import HashingEmbeddingProvider
from rag_core.retrieval.hybrid import QdrantHybridRetriever
from rag_core.retrieval.qdrant_store import QdrantDenseRetriever
from rag_core.schemas import Chunk, DocumentMetadata

OUT = Path("plans/reports/probes/w602-local-mode.json")
META = DocumentMetadata(source_url="https://example.org/probe", license="CC BY 3.0 IGO")


def main() -> int:
    # ⭐ Cảnh báo của chính client là một **kết quả**, không phải nhiễu:
    # "Payload indexes have no effect in the local Qdrant" nghĩa là filter vẫn
    # ĐÚNG nhưng quét tuyến tính. Ở 15.814 point thì chấp nhận được; ở 1 triệu
    # thì không, và người đọc báo cáo phải biết ranh giới ấy.
    caught: list[str] = []
    steps: dict[str, str] = {}

    def step(name: str, fn: Any) -> None:
        try:
            fn()
            steps[name] = "OK"
        except Exception as exc:
            steps[name] = f"FAIL: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")

        emb = HashingEmbeddingProvider(dimension=64, sparse=True)
        client = QdrantClient(":memory:")
        store = QdrantDenseRetriever(emb, collection="probe", client=client, tenant_id="public")

        step("ensure_collection", lambda: store.ensure_collection(recreate=True))
        chunks = [
            Chunk(
                chunk_id=f"c{i}",
                doc_id="d1",
                content=f"World Bank Vietnam poverty report section {i} tang truong kinh te",
                chunk_index=i,
                metadata=META,
            )
            for i in range(30)
        ]
        step("upsert", lambda: store.upsert(chunks))
        step("count", lambda: store.count())
        step("retrieve_dense", lambda: store.retrieve("poverty", top_k=5))
        step("retrieve_sparse", lambda: store.retrieve_sparse("poverty", top_k=5))
        hybrid = QdrantHybridRetriever(store, k=1, candidate_k=20, weights=(1.0, 0.25))
        step("hybrid_query_batch_points", lambda: hybrid.retrieve("poverty vietnam", top_k=5))
        step(
            "hybrid_with_filter",
            lambda: hybrid.retrieve("poverty", top_k=5, filters={"doc_id": "d1"}),
        )
        step("scroll", lambda: store.client.scroll("probe", limit=3))
        step("fetch_chunks", lambda: store.fetch_chunks(["c1", "c2"]))

        caught = sorted({str(w.message) for w in seen})

    ok = sum(1 for v in steps.values() if v == "OK")
    report = {
        "probe": "w602-local-mode",
        "cau_hoi": "qdrant-client local mode co chay noi duong truy hoi hien tai khong",
        "buoc": steps,
        "ket_qua": f"{ok}/{len(steps)}",
        "canh_bao_client_tu_phat": caught,
        "ket_luan": (
            "Chay duoc nguyen duong code — Space KHONG can cai lai truy hoi. "
            "Danh doi duy nhat: payload index khong co tac dung o local mode nen "
            "filter quet tuyen tinh."
        )
        if ok == len(steps)
        else "KHONG chay duoc — xem buoc FAIL.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok == len(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
