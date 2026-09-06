"""`W4-10` — semantic cache qua HTTP thật + Redis thật.

Ba ca của DoD: hit paraphrase, miss câu khác chủ đề, TTL. "Invalidate khi đổi
bundle" là thuộc tính của khoá (namespace mang `bundle_version`) và được kiểm
ở `tests/unit/test_semantic_cache.py` — dựng hai bundle thật chỉ để xem một
cái khoá đổi tên là trả giá integration cho một phép so chuỗi.

Embedder trong app test là giả (vector theo TỪ CUỐI câu — xem `chat_app.py`),
nên "paraphrase" ở đây nghĩa là "cùng từ cuối". Ngưỡng thật 0,96 trên BGE-M3
được đo ở `probes/w4-10-cosine-threshold.json`, không phải ở đây.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import redis as redis_sync
from sqlalchemy import Engine

from serving.core.chat import cache_namespace
from tests.integration.test_chat_stream import _chat, _serve

from .chat_app import GENERATOR

pytestmark = pytest.mark.integration

REDIS_URL = "redis://127.0.0.1:6379/0"


def _flush_semcache() -> None:
    client = redis_sync.Redis.from_url(REDIS_URL)
    for key in client.scan_iter("semcache:*"):
        client.delete(key)
    client.close()


def _ask_until_cached(
    client: httpx.Client, base: str, message: str, *, deadline_s: float = 5.0
) -> list[tuple[float, str, dict[str, Any]]]:
    """Ghi cache chạy NỀN sau khung `done` — lần hỏi lại ngay sau đó là một
    cuộc đua thật. Test chờ tới khi hit thay vì giả vờ đua không tồn tại."""
    deadline = time.monotonic() + deadline_s
    while True:
        _, frames = _chat(client, base, message)
        done = next(data for _, name, data in frames if name == "done")
        if done["finish_reason"] == "cache" or time.monotonic() > deadline:
            return frames
        time.sleep(0.2)


def test_a_paraphrase_hits_and_replays_the_first_answer(database: Engine, workspace: Path) -> None:
    _flush_semcache()
    proc, base = _serve(workspace, CHAT_TEST_CACHE="1", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=30.0) as client:
            first = "Ngân sách nhà nước chi bao nhiêu cho giáo dục?"
            _, frames1 = _chat(client, base, first)
            done1 = next(data for _, name, data in frames1 if name == "done")
            assert done1["finish_reason"] == "stop"
            text1 = "".join(d["text"] for _, n, d in frames1 if n == "delta")

            frames2 = _ask_until_cached(client, base, "Tỷ trọng đầu tư công dành cho giáo dục?")
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    done2 = next(data for _, name, data in frames2 if name == "done")
    assert done2["finish_reason"] == "cache"
    assert done2["usage"] == {}
    meta2 = frames2[0][2]
    assert meta2["cache"]["hit"] is True
    assert meta2["cache"]["matched_question"] == first
    text2 = "".join(d["text"] for _, n, d in frames2 if n == "delta")
    assert text2 == text1
    # Khung sources phát lại từ cache — của lượt ĐẦU, không phải một lượt truy
    # hồi mới (SlowRetriever sinh content theo query nên hai lượt sẽ khác nhau
    # nếu có ai lỡ truy hồi lại).
    sources2 = next(data for _, name, data in frames2 if name == "sources")
    assert sources2["sources"][0]["chunk_id"] == "chunk-1"


def test_a_different_topic_misses(database: Engine, workspace: Path) -> None:
    _flush_semcache()
    proc, base = _serve(workspace, CHAT_TEST_CACHE="1", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=30.0) as client:
            _chat(client, base, "Ngân sách nhà nước chi bao nhiêu cho giáo dục?")
            time.sleep(0.5)
            _, frames = _chat(client, base, "Cách đổi mật khẩu wifi?")
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    done = next(data for _, name, data in frames if name == "done")
    assert done["finish_reason"] == "stop"


def test_the_namespace_carries_a_ttl_and_the_bundle_version(
    database: Engine, workspace: Path
) -> None:
    """DoD "có TTL": mọi lần ghi refresh TTL của cả namespace. Và tên khoá mang
    `bundle_version` — đó chính là cơ chế invalidate."""
    _flush_semcache()
    proc, base = _serve(workspace, CHAT_TEST_CACHE="1", CHAT_TEST_DELTA_MS="1")
    try:
        with httpx.Client(timeout=30.0) as client:
            _, frames = _chat(client, base, "Ngân sách chi bao nhiêu cho giáo dục?")
            bundle_version = frames[0][2]["bundle_version"]
            deadline = time.monotonic() + 5.0
            client_r = redis_sync.Redis.from_url(REDIS_URL)
            # `W4-11`: namespace mang version prompt; `NEW-08`/`AU-02`: mang cả
            # `top_k` (request không khai nên là mặc định 5 của `ChatRequest`).
            key = f"semcache:acme:{cache_namespace(bundle_version, 5, GENERATOR)}"
            # redis-py sync client khai kiểu union với Awaitable — ép int
            # cho mypy; runtime luôn là int ở client đồng bộ.
            while int(cast("int", client_r.ttl(key))) < 0 and time.monotonic() < deadline:
                time.sleep(0.2)
            ttl = int(cast("int", client_r.ttl(key)))
            client_r.close()
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    assert 0 < ttl <= 86_400
