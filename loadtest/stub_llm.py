"""Một provider OpenAI-compat **giả** để đo trần thông lượng của chính hệ thống. `W6-05`.

    uv run python -m loadtest.stub_llm --port 8199 --profile deepseek

Rồi chạy server thật trỏ vào nó:

    DEEPSEEK_BASE_URL=http://127.0.0.1:8199 DEEPSEEK_API_KEY=stub make serve

## ⭐⭐ Vì sao load test **không** được gọi DeepSeek thật

Ngân sách p95 hiện tại là 4.842 ms, trong đó phần của chúng ta (truy hồi +
rerank) là **787 ms** — 16%. 84% còn lại là thời gian DeepSeek sinh token, thứ
nằm sau hàng đợi của một bên thứ ba và thay đổi theo giờ trong ngày.

Một load test mà số hạng trội là hàng đợi của người khác thì **đo người khác**.
Nó sẽ cho ra một đường cong đẹp, và đường cong ấy sẽ mô tả tải hiện tại của
DeepSeek chứ không mô tả điểm bão hoà của cái stack này. Tệ hơn: nó tốn tiền
theo số request, nên phép đo càng đáng tin (càng nhiều mẫu) thì càng đắt, và
`AU-11` (không có single-flight) biến mỗi lần trùng câu hỏi thành một lần trả
tiền nữa.

Nên stub này thay **đúng một thứ**: lời gọi HTTP tới nhà cung cấp. Truy hồi
thật, rerank thật (kể cả khoá GPU của `TD-63`), Qdrant thật, Postgres thật,
Redis thật, xác minh trích dẫn thật. Cái duy nhất giả là token, và nó giả theo
một phân phối **đo được từ `W5-11`**, không phải bịa:

| | đo được (242 request thật, `w511-deepseek.jsonl`) |
|---|---|
| prompt | p50 **3.081** token |
| completion | p50 **185** token, p95 622 |
| end-to-end | p50 2.598 ms · p95 4.842 ms · p99 11.142 ms |

## ⭐ Stub **trích nguyên văn** từ ngữ cảnh nó nhận được

Một stub trả `"xin chào [1]"` sẽ làm đường xác minh trích dẫn (`W4-09`) từ chối
ngay ở bước đầu, tức load test bỏ qua đúng phần đắt nhất sau tầng sinh: bộ so
`_quote_matches` chạy trên toàn văn nguồn. Nên stub đọc lại khối
`<<<NGUON n nonce>>>` trong prompt — cùng thứ model thật đọc — và chép ra một
đoạn có thật. Đường xác minh chạy đầy đủ và trả `verified`, không phải `invalid`.

## ⭐ `/__stats` là dụng cụ đo `AU-11`, không phải tiện ích debug

`AU-11`: cache miss không có single-flight, nên N request trùng nhau đồng thời
= N lời gọi trả tiền. Không có cách nào đo điều đó từ phía client — client chỉ
thấy N câu trả lời. Phải đếm ở phía **bị gọi**. `calls_streaming` sau một đợt
N request giống hệt nhau chính là con số ấy, và `concurrent_peak` nói nó đã
trùng nhau thật chứ không phải nối đuôi.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["PROFILES", "Profile", "Stats", "build_app", "main"]

logger = logging.getLogger("loadtest.stub_llm")

#: Khối nguồn do `rag_core.generation.guardrails.wrap_context` dựng.
_SOURCE_RE = re.compile(
    r"<<<NGUON (\d+) (\S+)>>>\n(.*?)\n<<<HET NGUON \1 \2>>>",
    re.DOTALL,
)


@dataclass(frozen=True)
class Profile:
    """Hình dạng thời gian của một nhà cung cấp giả.

    Hai hồ sơ, và **cả hai đều cần** — chúng trả lời hai câu hỏi khác nhau:

    * `fast` (không độ trễ): trần thông lượng của stack. Đây là chỗ khoá GPU
      (`TD-63`) và một worker (`TD-75`) lộ ra, vì không còn gì che chúng.
    * `deepseek` (hiệu chỉnh theo `W5-11`): điểm bão hoà *thực tế*. Một câu trả
      lời kéo dài 2 giây giữ một kết nối SSE, một hàng Postgres đang mở và một
      chỗ trong pool suốt 2 giây ấy — nên số người dùng đồng thời chịu được
      **không** suy ra được từ hồ sơ `fast`.
    """

    name: str
    ttft_ms: float
    """Thời gian tới token đầu. Phần này ở nhà cung cấp thật là hàng đợi + prefill."""
    token_ms: float
    """Khoảng cách giữa hai token."""
    completion_tokens: int

    @property
    def nominal_ms(self) -> float:
        return self.ttft_ms + self.token_ms * self.completion_tokens


#: `deepseek`: suy ra từ 242 request thật của `W5-11`. p50 end-to-end 2.598 ms
#: trừ đi p50 `prepare` 787 ms (`W5-06`) còn ~1.811 ms cho 185 token — tức
#: ~600 ms tới token đầu rồi ~6,5 ms/token. Không làm tròn cho đẹp: con số này
#: là mốc để nói "stub nhanh hơn/chậm hơn thật bao nhiêu".
PROFILES: dict[str, Profile] = {
    "fast": Profile("fast", ttft_ms=0.0, token_ms=0.0, completion_tokens=185),
    "deepseek": Profile("deepseek", ttft_ms=600.0, token_ms=6.5, completion_tokens=185),
    "slow": Profile("slow", ttft_ms=1200.0, token_ms=13.0, completion_tokens=622),
}


@dataclass
class Stats:
    """Sổ đếm phía **bị gọi**. Xem docstring module, phần `AU-11`."""

    calls_streaming: int = 0
    calls_blocking: int = 0
    concurrent_now: int = 0
    concurrent_peak: int = 0
    prompt_chars: int = 0
    started_at: float = field(default_factory=time.time)

    def enter(self, *, streaming: bool, prompt_chars: int) -> None:
        if streaming:
            self.calls_streaming += 1
        else:
            self.calls_blocking += 1
        self.prompt_chars += prompt_chars
        self.concurrent_now += 1
        self.concurrent_peak = max(self.concurrent_peak, self.concurrent_now)

    def leave(self) -> None:
        self.concurrent_now -= 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls_streaming": self.calls_streaming,
            "calls_blocking": self.calls_blocking,
            "calls_total": self.calls_streaming + self.calls_blocking,
            "concurrent_now": self.concurrent_now,
            "concurrent_peak": self.concurrent_peak,
            "prompt_chars": self.prompt_chars,
            "uptime_s": round(time.time() - self.started_at, 1),
        }


def _sources(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """`[(n, nội dung nguồn)]` đọc từ prompt — cùng thứ model thật đọc."""
    blob = "\n".join(str(m.get("content") or "") for m in messages)
    return [(int(n), body) for n, _nonce, body in _SOURCE_RE.findall(blob)]


def _answer(messages: list[dict[str, Any]], profile: Profile) -> str:
    """Câu trả lời giả **có trích dẫn kiểm chứng được**.

    Quote lấy nguyên văn từ nguồn nên `verify_citations` chạy hết đường và trả
    `verified`. Một stub trả quote bịa sẽ khiến mọi lượt đi nhánh `invalid` —
    nhánh rẻ hơn — và load test sẽ báo một con số lạc quan.
    """
    sources = _sources(messages)
    if not sources:
        return "Không đủ thông tin trong ngữ cảnh để trả lời.\nCITATIONS: []"

    n, body = sources[0]
    quote = " ".join(body.split())[:180]
    # Đệm cho đủ `completion_tokens` để phần streaming có thời lượng đúng hồ sơ.
    filler = f"Theo nguồn [{n}], nội dung liên quan được nêu trong tài liệu. "
    padding = filler * max(1, profile.completion_tokens // 12)
    citations = json.dumps([{"n": n, "quote": quote}], ensure_ascii=False)
    return f"{padding}\nCITATIONS: {citations}"


def _usage(prompt_chars: int, profile: Profile) -> dict[str, int]:
    # ~3,5 ký tự/token cho tiếng Việt + tiếng Anh trộn lẫn. Xấp xỉ có chủ ý:
    # stub không tính tiền, và con số này chỉ để đường đọc `usage` có thứ để đọc.
    return {
        "prompt_tokens": max(1, prompt_chars // 4),
        "completion_tokens": profile.completion_tokens,
        "total_tokens": max(1, prompt_chars // 4) + profile.completion_tokens,
        "prompt_cache_hit_tokens": 0,
    }


def build_app(profile: Profile) -> FastAPI:
    app = FastAPI(title="stub OpenAI-compat provider", docs_url=None, redoc_url=None)
    stats = Stats()
    app.state.stats = stats
    app.state.profile = profile

    @app.get("/__stats")
    async def read_stats() -> JSONResponse:
        return JSONResponse({**stats.as_dict(), "profile": profile.name})

    @app.post("/__reset")
    async def reset() -> JSONResponse:
        """Đặt lại sổ đếm giữa hai bậc concurrency — **không** đặt lại
        `concurrent_now`, vì có thể còn request đang bay."""
        stats.calls_streaming = 0
        stats.calls_blocking = 0
        stats.concurrent_peak = stats.concurrent_now
        stats.prompt_chars = 0
        stats.started_at = time.time()
        return JSONResponse(stats.as_dict())

    @app.post("/chat/completions")
    async def completions(request: Request) -> Any:
        payload = await request.json()
        messages = list(payload.get("messages") or [])
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        model = str(payload.get("model") or "stub-model")
        streaming = bool(payload.get("stream"))
        stats.enter(streaming=streaming, prompt_chars=prompt_chars)

        text = _answer(messages, profile)
        usage = _usage(prompt_chars, profile)

        if not streaming:
            # Đường `complete()` — `W4-07` viết lại truy vấn đi qua đây. Nó ngắn
            # hơn nhiều lượt sinh chính, nên dùng 1/10 thời lượng thay vì hồ sơ đầy.
            try:
                await asyncio.sleep(profile.ttft_ms / 10_000.0)
                body = {
                    "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                    "model": model,
                    "usage": usage,
                }
                return JSONResponse(body)
            finally:
                stats.leave()

        async def stream() -> AsyncIterator[bytes]:
            try:
                await asyncio.sleep(profile.ttft_ms / 1000.0)
                step = max(1, len(text) // profile.completion_tokens)
                for i in range(0, len(text), step):
                    chunk = {
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": text[i : i + step]}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                    if profile.token_ms:
                        await asyncio.sleep(profile.token_ms / 1000.0)
                tail = {
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(tail)}\n\n".encode()
                # Mẩu mang `usage` có `choices` RỖNG — đúng hình dạng OpenAI-compat
                # thật, và `astream` của chúng ta đọc `usage` ngoài nhánh `choices`
                # chính vì thế. Stub trả sai chỗ này thì `cost_usd` luôn bằng 0 mà
                # không ai biết.
                tail_usage = {"model": model, "choices": [], "usage": usage}
                yield f"data: {json.dumps(tail_usage)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                stats.leave()

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="python -m loadtest.stub_llm",
        description="W6-05 — provider OpenAI-compat giả, hiệu chỉnh theo số đo W5-11",
    )
    # ⚠️ 8199 chứ không 8099: `tests/integration/test_chat_stream.py` cấp cổng
    # từ dải **8091–8119** cho các tiến trình uvicorn của nó. Một stub đang
    # chạy ở 8099 làm đúng một bài trong dải ấy đỏ, với thông báo "uvicorn
    # chết lúc khởi động" — không nhắc gì tới cổng. Mất một lượt chẩn đoán.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="deepseek")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    profile = PROFILES[args.profile]
    logger.info(
        "stub %r: ttft %.0f ms + %d token × %.1f ms = %.0f ms danh nghĩa",
        profile.name,
        profile.ttft_ms,
        profile.completion_tokens,
        profile.token_ms,
        profile.nominal_ms,
    )
    uvicorn.run(build_app(profile), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
