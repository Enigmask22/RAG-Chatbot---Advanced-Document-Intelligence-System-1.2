"""`W6-02`: một Space tier miễn phí có chạy nổi pipeline này không?

⭐⭐ **Câu hỏi này phải hỏi trước khi thiết kế, không phải sau khi deploy.**
`W6-05` đo trần thông lượng nằm ở rerank — **685 ms trên GPU**. Space tier miễn
phí không có GPU và chỉ có **2 vCPU**. Nếu rerank trên CPU vượt ngân sách 30 giây
của `G6` thì mọi thiết kế "bê nguyên serving lên Space" là ngõ cụt, và biết điều
ấy ở đây rẻ hơn biết nó sau một buổi dựng image.

Phép đo giả lập 2 vCPU bằng `torch.set_num_threads(2)`. ⚠️ Đây là **cận dưới của
thời gian**, tức cận trên của tính khả thi: CPU laptop này nhanh hơn vCPU chia sẻ
của Space theo từng nhân, nên Space sẽ **chậm hơn** con số in ra chứ không nhanh
hơn. Một kết luận "không khả thi" ở đây là chắc chắn; một kết luận "khả thi" mới
là thứ cần đo lại trên chính Space.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

# Phải đặt TRƯỚC khi torch nạp: các biến này chỉ được đọc lúc khởi tạo thread
# pool, đặt sau `import torch` là không có tác dụng và không có gì báo.
THREADS = int(os.environ.get("PROBE_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import torch  # noqa: E402

from rag_core.embedding.bge_m3 import BgeM3EmbeddingProvider  # noqa: E402
from rag_core.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402

QUERY = "Chính sách tiền tệ ảnh hưởng thế nào tới lạm phát trong giai đoạn nghiên cứu?"


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1024 / 1024


def _load_chunks(n: int, url: str, collection: str) -> list[str]:
    """Lấy chunk THẬT từ index. Văn bản giả sẽ cho token count sai."""
    import httpx

    r = httpx.post(
        f"{url}/collections/{collection}/points/scroll",
        json={"limit": n, "with_payload": True, "with_vector": False},
        timeout=30.0,
    )
    r.raise_for_status()
    points = r.json()["result"]["points"]
    # Văn bản nằm trong `payload["chunk"]["content"]`, không phải ở gốc payload —
    # gốc chỉ mang metadata để lọc.
    texts = [(p["payload"].get("chunk") or {}).get("content") or "" for p in points]
    return [t for t in texts if t]


def _time(fn: Any, repeats: int) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "min_ms": round(samples[0], 1),
        "median_ms": round(samples[len(samples) // 2], 1),
        "max_ms": round(samples[-1], 1),
        "n": repeats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="rag_bgem3_ctx")
    ap.add_argument("--candidates", type=int, nargs="+", default=[50, 20, 10])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--out", type=Path, default=Path("plans/reports/probes/w602-cpu-feasibility.json")
    )
    args = ap.parse_args()

    torch.set_num_threads(THREADS)

    report: dict[str, Any] = {
        "probe": "w602-cpu-feasibility",
        "cau_hoi": "Space tier mien phi (2 vCPU, khong GPU) co chay noi pipeline nay khong?",
        "may_do": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "cpu_count_that": os.cpu_count(),
        },
        "canh_bao": (
            "Day la CAN DUOI cua thoi gian: vCPU chia se cua Space cham hon nhan "
            "cua laptop nay. Ket luan 'khong kha thi' la chac chan; ket luan "
            "'kha thi' phai do lai tren chinh Space."
        ),
        "rss_truoc_khi_nap_mb": _rss_mb(),
    }

    max_c = max(args.candidates)
    chunks = _load_chunks(max_c, args.qdrant, args.collection)
    if len(chunks) < max_c:
        print(f"⚠️  chi lay duoc {len(chunks)}/{max_c} chunk")
    report["n_chunk_that"] = len(chunks)
    report["do_dai_chunk_ky_tu"] = {
        "median": sorted(len(c) for c in chunks)[len(chunks) // 2],
        "max": max(len(c) for c in chunks),
    }

    # --- nạp model: đây là cold-start, và Space NGỦ rồi cold-start ---
    t0 = time.perf_counter()
    embedder = BgeM3EmbeddingProvider(device="cpu")
    embedder.embed_query_hybrid("làm nóng")
    report["nap_embedder_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    report["rss_sau_embedder_mb"] = _rss_mb()

    t0 = time.perf_counter()
    reranker = CrossEncoderReranker(device="cpu", dtype="float32")
    reranker.score("làm nóng", chunks[:1])
    report["nap_reranker_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    report["rss_sau_reranker_mb"] = _rss_mb()

    # --- embed truy vấn ---
    report["embed_query"] = _time(lambda: embedder.embed_query_hybrid(QUERY), args.repeats)

    # --- rerank theo số ứng viên ---
    report["rerank"] = {}
    for c in args.candidates:
        if c > len(chunks):
            continue
        docs = chunks[:c]
        report["rerank"][f"c{c}"] = _time(lambda d=docs: reranker.score(QUERY, d), args.repeats)

    # --- kết luận: cộng lại và so với ngân sách ---
    embed_ms = report["embed_query"]["median_ms"]
    budget_ms = 30_000
    report["ket_luan"] = {}
    for key, stat in report["rerank"].items():
        local_ms = embed_ms + stat["median_ms"]
        report["ket_luan"][key] = {
            "embed_cong_rerank_ms": round(local_ms, 1),
            "phan_tram_ngan_sach_30s": round(100 * local_ms / budget_ms, 1),
            # Sinh chưa tính vào: `W6-05` đo p50 completion ~4,9 s qua DeepSeek.
            "cong_them_sinh_4900ms": round(local_ms + 4900, 1),
            "vuot_30s": local_ms + 4900 > budget_ms,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
