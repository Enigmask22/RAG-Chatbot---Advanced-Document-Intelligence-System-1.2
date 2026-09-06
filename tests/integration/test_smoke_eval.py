"""Smoke eval chạy thật trên Qdrant — `W5-09`.

Module này là bản sao trong-tiến-trình của job `smoke-eval` trong
`.github/workflows/ci.yml`: cùng fixture, cùng baseline, cùng tham số retriever.

## ⭐⭐ Bài đáng giá nhất ở đây là bài chứng minh cổng **đỏ được**

`G5` đòi *"mở 1 PR cố ý làm tụt retrieval → CI phải đỏ"*. Một cổng chỉ được
kiểm ở trạng thái xanh là một cổng chưa ai biết có gác hay không — và cách hỏng
phổ biến nhất của loại cổng này không phải báo động giả, mà là **không bao giờ
báo**. `TestTheGateHasTeeth` tiêm một lần tụt truy hồi thật rồi đòi phán quyết
đổi màu.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pipeline.eval.smoke import (
    SMOKE_PREFIX,
    FrozenEmbedder,
    SmokeFixture,
    compare_to_baseline,
    default_bundle,
    load_fixture,
    retrieval_options,
    run_smoke,
    seed_collection,
)

pytestmark = pytest.mark.integration

FIXTURE = Path("data/eval/smoke/smoke_v1.jsonl.gz")
BASELINE = Path("data/eval/smoke/baseline.json")
COLLECTION = "rag_smoke_pytest"

# Đọc từ **manifest bundle**, không gõ lại: một bản sao thứ hai của cấu hình
# production biến cổng thành cổng của một hệ thống không tồn tại.
PRODUCTION_OPTIONS = retrieval_options(default_bundle())


@pytest.fixture(scope="module")
def fixture() -> SmokeFixture:
    return load_fixture(FIXTURE)


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(BASELINE.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def store(fixture: SmokeFixture) -> Iterator[Any]:
    from qdrant_client.http.exceptions import ResponseHandlingException

    from rag_core.retrieval.qdrant_store import QdrantDenseRetriever
    from rag_core.settings import get_settings

    embedder = FrozenEmbedder(
        dict(fixture.query_vectors),
        dimension=fixture.dimension,
        model_name=fixture.embedding_model,
    )
    retriever = QdrantDenseRetriever(
        embeddings=embedder, collection=COLLECTION, url=get_settings().qdrant_url
    )
    try:
        written = seed_collection(retriever, fixture)
    except (ResponseHandlingException, OSError) as exc:  # pragma: no cover - phụ thuộc máy
        pytest.skip(f"không có Qdrant: {exc}")
    assert written == len(fixture.chunks)
    yield retriever
    retriever.client.delete_collection(collection_name=COLLECTION)


def _retriever(store: Any, **overrides: Any) -> Any:
    from rag_core.retrieval import QdrantHybridRetriever

    return QdrantHybridRetriever(store, **{**PRODUCTION_OPTIONS, **overrides})


class TestTheFrozenIndexReproducesTheBaseline:
    def test_the_gate_is_green_on_main(
        self, store: Any, fixture: SmokeFixture, baseline: dict[str, Any]
    ) -> None:
        """Chính là job `smoke-eval`, chạy trong tiến trình test."""
        result = run_smoke(_retriever(store), fixture, top_k=baseline["top_k"])
        assert (
            compare_to_baseline(result, baseline, tolerance=0.02, options=PRODUCTION_OPTIONS) == []
        )

    def test_the_options_come_from_the_bundle_not_from_this_file(self) -> None:
        """⭐ Cổng gác cả **cấu hình**: đổi `components.retrieval.options` trong
        manifest là đổi `retrieval_options`, và bộ ấy nằm trong baseline."""
        assert retrieval_options(default_bundle()) == PRODUCTION_OPTIONS
        assert isinstance(PRODUCTION_OPTIONS["weights"], tuple)

    def test_the_numbers_are_stable_across_two_runs(
        self, store: Any, fixture: SmokeFixture
    ) -> None:
        """HNSW là xấp xỉ, nên "tất định" phải được **đo** chứ không được giả
        định. Ở kích thước này Qdrant quét toàn bộ nên nó tất định — và bài này
        là chỗ điều đó thôi là một lời khai."""
        first = run_smoke(_retriever(store), fixture, top_k=10)
        second = run_smoke(_retriever(store), fixture, top_k=10)
        assert first.metrics == second.metrics

    def test_every_query_was_actually_asked(self, store: Any, fixture: SmokeFixture) -> None:
        """`FrozenEmbedder.seen` chứng minh cả 30 câu đã đi qua embedder — một
        bộ lọc lặng lẽ ở giữa sẽ làm metric tính trên ít câu hơn mà vẫn ra số."""
        store.embeddings.seen.clear()
        run_smoke(_retriever(store), fixture, top_k=10)
        assert sorted(store.embeddings.seen) == sorted(q.query for q in fixture.queries)


class TestTheGateHasTeeth:
    """⭐⭐ `G5`: một lần tụt truy hồi thật phải làm cổng đỏ."""

    @pytest.mark.parametrize(
        ("label", "overrides"),
        [
            ("RRF k về mặc định thư viện", {"k": 60}),
            ("bỏ trọng số nhánh", {"weights": None}),
            ("đảo trọng số dense/sparse", {"weights": (0.25, 1.0)}),
        ],
    )
    def test_a_real_regression_turns_the_gate_red(
        self,
        store: Any,
        fixture: SmokeFixture,
        baseline: dict[str, Any],
        label: str,
        overrides: dict[str, Any],
    ) -> None:
        retriever = _retriever(store, **overrides)
        result = run_smoke(retriever, fixture, top_k=10)
        # ⚠️ KHÔNG truyền `options`: nếu truyền thì cổng đỏ vì tham số đổi,
        # và bài này sẽ xanh mà không chứng minh được rằng **con số** đã tụt.
        failures = compare_to_baseline(result, baseline, tolerance=0.02)
        assert failures, f"{label} không làm cổng đỏ — cổng đang trang trí"

    def test_a_change_the_code_ignores_does_not_move_the_gate(
        self, store: Any, fixture: SmokeFixture, baseline: dict[str, Any]
    ) -> None:
        """⚠️ Mặt kia của cùng đồng xu, và nó là một **giới hạn đã đo**:
        `candidate_k` nhỏ hơn `top_k` bị `QdrantHybridRetriever._depth()` kẹp
        lên bằng `top_k`, nên nó không đổi gì cả. Cổng mù ở đúng chỗ mã cũng
        mù — nhưng trên index thật `candidate_k` vẫn quan trọng cho tầng
        rerank, thứ smoke này **không** phủ. Xem `TD-81`.
        """
        result = run_smoke(_retriever(store, candidate_k=5), fixture, top_k=10)
        assert compare_to_baseline(result, baseline, tolerance=0.02) == []


class TestTheFixtureIsNotSecretlyEasy:
    def test_the_distractors_actually_compete(self, store: Any, fixture: SmokeFixture) -> None:
        """Nếu mọi câu đều đạt 1,0 thì nhiễu không cạnh tranh và cổng chỉ bắt
        được thảm hoạ. Đòi có ít nhất một câu **chưa** hoàn hảo."""
        result = run_smoke(_retriever(store), fixture, top_k=10)
        imperfect = [
            qid for qid, scores in result.per_query.items() if scores[f"{SMOKE_PREFIX}mrr"] < 1.0
        ]
        assert imperfect, "mọi câu đều hoàn hảo — fixture quá dễ"

    def test_an_unknown_query_is_refused_by_the_embedder(
        self, store: Any, fixture: SmokeFixture
    ) -> None:
        from pipeline.eval.smoke import UnknownTextError

        with pytest.raises(UnknownTextError):
            _retriever(store).retrieve("một câu hỏi chưa từng đóng băng", top_k=5)
