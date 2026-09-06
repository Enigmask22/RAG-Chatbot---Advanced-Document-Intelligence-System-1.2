"""Đo `TD-72` trên hệ đang phục vụ: request đầu tiên sau khi tiến trình lên. `W6-05`.

    uv run python -m loadtest.coldstart_probe --n 4 \\
        --out plans/reports/probes/w605-td72-coldstart.json

## ⭐⭐ `/ready` xanh **không** có nghĩa là phục vụ được

`W5-06` đo được rerank lạnh tốn 10,7× rerank nóng: trọng số nạp lúc
`activate()`, nhưng kernel CUDA chỉ khởi tạo ở lời gọi `score()` **đầu tiên** —
và `/ready` xanh từ trước đó. Nên một health check xanh mời traffic vào một
tiến trình chưa phục vụ được, và người dùng đầu tiên sau mỗi lần deploy trả
tiền cho việc khởi tạo ấy.

`p95` pha loãng chuyện này tới vô hình: nó xảy ra **một lần** mỗi deploy. Đó
chính là lý do phải có một probe riêng — thứ đo `n` request **đầu tiên** theo
thứ tự, không phải một phân vị.

⚠️ Chạy nó ngay sau khi tiến trình lên, và **trước** bất cứ request nào khác.
Một lượt smoke test chen vào trước sẽ làm nóng hộ, và probe sẽ báo là không có
vấn đề gì.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["probe"]

logger = logging.getLogger("loadtest.coldstart_probe")


async def _turn(client: Any, base_url: str, key: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    ttft: float | None = None
    done: dict[str, Any] = {}
    event = ""
    async with client.stream(
        "POST",
        f"{base_url}/chat",
        json={"message": question, "top_k": 5},
        headers={"Authorization": f"Bearer {key}"},
        timeout=300.0,
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                if event == "delta" and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000.0
                elif event == "done":
                    done = json.loads(line[5:])
    return {
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "prepare_ms": done.get("prepare_ms"),
        "ttfb_ms": done.get("ttfb_ms"),
        "total_ms": done.get("total_ms"),
    }


async def probe(n: int, *, base_url: str, key: str) -> dict[str, Any]:
    import httpx

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for i in range(n):
            # Câu hỏi KHÁC nhau mỗi lượt: cùng một câu sẽ trúng semantic cache
            # từ lượt hai và probe sẽ đo cache thay vì đo lượt nóng.
            rows.append(await _turn(client, base_url, key, f"Nợ công Việt Nam ra sao ({i})?"))

    first, rest = rows[0], rows[1:]
    warm = min((r["wall_ms"] for r in rest), default=first["wall_ms"])
    return {
        "n": n,
        "turns": rows,
        "cold_wall_ms": first["wall_ms"],
        "warm_wall_ms": warm,
        "penalty_ms": round(first["wall_ms"] - warm, 1),
        "ratio": round(first["wall_ms"] / warm, 2) if warm else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m loadtest.coldstart_probe",
        description="W6-05 / TD-72 — giá của request đầu tiên sau khi tiến trình lên",
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    key = os.environ.get("LOADTEST_API_KEY", "")
    if not key:
        parser.error("thiếu LOADTEST_API_KEY")

    result = asyncio.run(probe(args.n, base_url=args.base_url, key=key))
    logger.info(
        "lạnh %.0f ms · nóng %.0f ms · phạt %.0f ms (%.1f×)",
        result["cold_wall_ms"],
        result["warm_wall_ms"],
        result["penalty_ms"],
        result["ratio"] or 0.0,
    )
    for i, row in enumerate(result["turns"]):
        logger.info(
            "  #%d wall %.0f · ttft %s · prepare %s · ttfb %s",
            i + 1,
            row["wall_ms"],
            row["ttft_ms"],
            row["prepare_ms"],
            row["ttfb_ms"],
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("đã ghi %s", args.out.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
