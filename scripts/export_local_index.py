"""Sao index từ Qdrant **server** sang một kho `qdrant-client` **local mode**.

Lý do tồn tại (`W6-02`): HF Space chạy trên Gradio SDK — không có Docker, không
có Qdrant server. Nhưng `qdrant-client` có một cài đặt thuần Python cùng API, và
một phép đo (`probes/w602-local-mode.json`) cho thấy nó chạy **nguyên** đường
truy hồi hiện tại: named vector `dense`, `sparse`, `query_batch_points`, filter.

⭐⭐ Nên Space **không** cài lại truy hồi. Đó là điều kiện tiên quyết chứ không
phải một tối ưu: một bản cài thứ hai của cùng một logic là họ `AU-12`, và ở đây
nó tệ hơn — demo sẽ trình diễn một hệ thống *khác* hệ thống đã có số đo, mà
không có gì đỏ để nói ra điều đó.

⚠️ **Hai khác biệt thật, và script này ĐO chúng chứ không giả vờ chúng không có:**

1. **Local mode tìm chính xác, server tìm xấp xỉ.** Server dựng HNSW
   (`indexed_vectors_count: 15814`); local mode nhân ma trận toàn bộ. Ở 15.814
   điểm phép nhân ấy là vài mili giây, nên đây không phải vấn đề tốc độ — nó là
   vấn đề *danh tính*: kết quả local có thể **tốt hơn** kết quả đã eval. Script
   in ra overlap@k giữa hai bên trên truy vấn golden thật.
2. **Payload index không có tác dụng ở local mode** (client tự cảnh báo). Filter
   vẫn cho đúng kết quả, chỉ là quét tuyến tính. Ở 15.814 điểm chấp nhận được;
   ở 1 triệu thì không.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "sparse"


def _copy_points(
    src: QdrantClient,
    dst: QdrantClient,
    collection: str,
    *,
    batch: int,
) -> int:
    """Cuốn toàn bộ point qua, giữ nguyên id, vector và payload.

    `with_vectors=True` là bắt buộc và không có mặc định an toàn: thiếu nó thì
    script chạy xong, đếm đủ 15.814 point, và mọi truy vấn trả về rỗng — hỏng
    theo đúng kiểu mọi con số đều xanh.
    """
    written = 0
    offset: Any = None
    while True:
        points, offset = src.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        dst.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=p.id, vector=_vector_of(p), payload=p.payload) for p in points
            ],
            wait=True,
        )
        written += len(points)
        print(f"  … {written}", end="\r", flush=True)
        if offset is None:
            break
    print()
    return written


def _vector_of(point: Any) -> dict[str, Any]:
    """Named-vector dict của một point đọc từ server → dạng ghi được.

    Sparse về từ server là `SparseVector` của client; dense là `list[float]`.
    Cả hai đi thẳng được, nhưng viết tường minh để chỗ hỏng lộ ra ở đây chứ
    không lộ ra dưới dạng "truy hồi sparse trả rỗng".
    """
    vectors = point.vector
    if not isinstance(vectors, dict):
        raise SystemExit(
            f"point {point.id} không dùng named vector — collection này không "
            "phải collection mà bundle mô tả"
        )
    out: dict[str, Any] = {}
    for name, value in vectors.items():
        if hasattr(value, "indices"):
            out[name] = models.SparseVector(indices=list(value.indices), values=list(value.values))
        else:
            out[name] = [float(x) for x in value]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default="rag_bgem3_ctx")
    ap.add_argument("--url", default="http://127.0.0.1:6333")
    ap.add_argument("--out", type=Path, required=True, help="thư mục kho local mode")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="xoá thư mục đích nếu đã có")
    args = ap.parse_args()

    src = QdrantClient(url=args.url, timeout=120)
    info = src.get_collection(args.collection)
    expected = int(info.points_count or 0)
    print(f"nguồn: {args.url}/{args.collection} — {expected} point")

    out: Path = args.out
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} đã tồn tại — dùng --force để ghi đè")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dst = QdrantClient(path=str(out))
    params = info.config.params
    dense_params = params.vectors[DENSE]  # type: ignore[index]
    dst.create_collection(
        collection_name=args.collection,
        vectors_config={
            DENSE: models.VectorParams(size=dense_params.size, distance=dense_params.distance)
        },
        sparse_vectors_config=(
            {SPARSE: models.SparseVectorParams()} if params.sparse_vectors else None
        ),
    )

    started = time.perf_counter()
    written = _copy_points(src, dst, args.collection, batch=args.batch)
    elapsed = time.perf_counter() - started

    got = int(dst.count(args.collection, exact=True).count)
    print(f"ghi {written} point trong {elapsed:.1f}s — kho local đếm {got}")
    if got != expected:
        raise SystemExit(f"LỆCH: nguồn {expected}, đích {got}")

    dst.close()
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    meta = {
        "collection": args.collection,
        "points": got,
        "bytes": size,
        "mb": round(size / 1024 / 1024, 1),
        "export_seconds": round(elapsed, 1),
    }
    (out / "export.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
