"""Đo `AU-11` trên hệ đang phục vụ: N câu hỏi **giống hệt nhau**, cùng lúc. `W6-05`.

    uv run python -m loadtest.singleflight_probe --n 8 \\
        --out plans/reports/probes/w605-au11-singleflight.json

## ⭐⭐ Vì sao một probe riêng, không phải một chế độ của locust

Cái cần đo là **đợt đầu tiên**, và chỉ đợt đầu tiên. Sau khi một lượt ghi được
câu trả lời vào semantic cache thì mọi lượt sau trúng cache và gần như miễn
phí — nên một lần chạy locust 60 giây trên cùng một câu hỏi sẽ pha loãng đúng
thứ đáng đo: `provider_calls / requests` sẽ ra 0,2 và trông như cache đang làm
tốt việc của nó.

`AU-11` không nói cache hỏng. Nó nói **cache miss không có single-flight**: N
request đến trước khi có gì để trúng thì cả N cùng trượt, cùng gọi, cùng trả
tiền. Probe này bắn đúng một đợt N và đếm.

## ⭐ Đếm ở phía **bị gọi**, không ở phía client

Client thấy N câu trả lời dù có single-flight hay không — nó không phân biệt
được. Con số duy nhất phân biệt được nằm ở `loadtest/stub_llm.py:/__stats`.

Kỳ vọng khi CHƯA có single-flight: `provider_calls == n`.
Kỳ vọng sau khi vá: `provider_calls == 1`, và N−1 lượt kia đợi rồi dùng chung
kết quả. `concurrent_peak` của stub phân biệt hai chế độ ấy một cách không cãi
được: nó bằng `n` ở chế độ đầu và bằng 1 ở chế độ sau.
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

__all__ = ["ProbeResult", "probe"]

logger = logging.getLogger("loadtest.singleflight_probe")

QUESTION = "Tăng trưởng GDP của Việt Nam gần đây ra sao?"


class ProbeResult(dict[str, Any]):
    """Chỉ là một dict — nhưng có tên để chỗ gọi đọc được ý định."""


async def _one_turn(client: Any, base_url: str, key: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    ttft: float | None = None
    frames = 0
    done: dict[str, Any] = {}
    event = ""
    async with client.stream(
        "POST",
        f"{base_url}/chat",
        json={"message": question, "top_k": 5},
        headers={"Authorization": f"Bearer {key}"},
        timeout=180.0,
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", "replace")[:200]
            return {"ok": False, "status": response.status_code, "detail": body}
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                if event == "delta":
                    frames += 1
                    if ttft is None:
                        ttft = (time.perf_counter() - started) * 1000.0
                elif event == "done":
                    done = json.loads(line[5:])
    return {
        "ok": bool(done),
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "deltas": frames,
        # ⚠️ Tín hiệu trúng cache là `finish_reason == "cache"` (`W4-10`), KHÔNG
        # phải một trường `cache_hit` — bản đầu của probe này đọc trường ấy và
        # báo "0 trúng cache" ở đúng lượt mà provider được gọi 0 lần.
        "cached": done.get("finish_reason") == "cache",
        "done": done,
    }


async def probe(
    n: int, *, base_url: str, stub_url: str, key: str, question: str = QUESTION
) -> ProbeResult:
    import httpx

    async with httpx.AsyncClient() as client:
        await client.post(f"{stub_url}/__reset", timeout=10.0)
        before = (await client.get(f"{stub_url}/__stats", timeout=10.0)).json()
        started = time.perf_counter()
        turns = await asyncio.gather(
            *(_one_turn(client, base_url, key, question) for _ in range(n)),
            return_exceptions=True,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        after = (await client.get(f"{stub_url}/__stats", timeout=10.0)).json()

    rows = [t for t in turns if isinstance(t, dict)]
    errors = [repr(t) for t in turns if not isinstance(t, dict)]
    ok = [r for r in rows if r.get("ok")]
    return ProbeResult(
        n=n,
        question=question,
        wall_ms=round(wall_ms, 1),
        completed=len(ok),
        errors=errors + [r for r in rows if not r.get("ok")],
        provider_calls=after.get("calls_total", 0) - before.get("calls_total", 0),
        provider_streaming=after.get("calls_streaming", 0) - before.get("calls_streaming", 0),
        provider_concurrent_peak=after.get("concurrent_peak", 0),
        single_flight=after.get("calls_streaming", 0) - before.get("calls_streaming", 0) <= 1,
        cache_hits=sum(1 for r in ok if r.get("cached")),
        ttft_ms=[r.get("ttft_ms") for r in ok],
        total_ms=[r.get("total_ms") for r in ok],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m loadtest.singleflight_probe",
        description="W6-05 / AU-11 — N request trùng nhau đồng thời, đếm lời gọi provider",
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--stub-url", default="http://127.0.0.1:8199")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    key = os.environ.get("LOADTEST_API_KEY", "")
    if not key:
        parser.error("thiếu LOADTEST_API_KEY")

    result = asyncio.run(
        probe(
            args.n,
            base_url=args.base_url,
            stub_url=args.stub_url,
            key=key,
            question=args.question,
        )
    )
    logger.info(
        "n=%d · hoàn thành %d · gọi provider %d · trúng cache %d · single-flight: %s",
        result["n"],
        result["completed"],
        result["provider_calls"],
        result["cache_hits"],
        "CÓ" if result["single_flight"] else "KHÔNG",
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
