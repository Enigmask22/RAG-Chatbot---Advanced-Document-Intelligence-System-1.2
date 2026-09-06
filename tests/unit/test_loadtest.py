"""Dụng cụ load test của `W6-05`.

⚠️ **Không** import `loadtest.locustfile` ở đây. `import locust` kéo theo gevent,
thứ monkey-patch thư viện chuẩn ngay lúc import — trong cùng một phiên pytest
đang chạy `pytest-asyncio`. Tầng CI `unit` cũng cố ý không cài extra `loadtest`.
Xem docstring `loadtest/queries.py`.
"""

from __future__ import annotations

import json

import pytest

from loadtest.queries import FALLBACK_QUERIES, load_queries
from loadtest.stub_llm import PROFILES, Stats, _answer, _sources, build_app
from loadtest.sweep import (
    Level,
    _read_locust_csv,
    bucket_p95,
    delta,
    parse_buckets,
    pick_knee,
)

# --------------------------------------------------------------- histogram


def _exposition(stage: str, counts: dict[str, float]) -> str:
    lines = [
        f'rag_stage_duration_seconds_bucket{{stage="{stage}",le="{le}"}} {n}'
        for le, n in counts.items()
    ]
    # Dòng nhiễu phải bị bỏ qua, không được làm hỏng phép đọc.
    lines.append('rag_http_requests_total{path="/chat"} 42')
    lines.append("# HELP rag_stage_duration_seconds mô tả")
    return "\n".join(lines)


class TestParseBuckets:
    def test_reads_stage_and_edges(self) -> None:
        text = _exposition("rerank", {"0.5": 1, "1.0": 3, "+Inf": 3})
        buckets = parse_buckets(text)
        assert buckets == {"rerank": {0.5: 1.0, 1.0: 3.0, float("inf"): 3.0}}

    def test_ignores_other_metrics(self) -> None:
        assert parse_buckets('rag_http_requests_total{path="/chat"} 42') == {}


class TestDelta:
    def test_subtracts_two_snapshots(self) -> None:
        before = parse_buckets(_exposition("rerank", {"1.0": 5, "+Inf": 5}))
        after = parse_buckets(_exposition("rerank", {"1.0": 8, "+Inf": 9}))
        assert delta(before, after) == {"rerank": {1.0: 3.0, float("inf"): 4.0}}

    def test_drops_stages_with_no_new_observations(self) -> None:
        """Một chặng không chạy ở bậc này thì **không có hàng**, chứ không phải
        có một hàng bằng 0 — p95 của không mẫu nào là không xác định, không phải 0."""
        snapshot = parse_buckets(_exposition("understand", {"1.0": 7, "+Inf": 7}))
        assert delta(snapshot, snapshot) == {}

    def test_a_cold_start_does_not_leak_into_the_next_level(self) -> None:
        """⭐⭐ Hồi quy của chính lượt đo thử `W6-05`.

        Bậc đầu chịu lượt rerank lạnh 7,3 s (`TD-72`). Histogram tích luỹ nên
        đọc thẳng ở bậc sau vẫn thấy mẫu ấy và báo `rerank p95 ≈ 9.375 ms` —
        một con số không mô tả bậc nào cả. Phép trừ phải làm nó biến mất.
        """
        cold = parse_buckets(_exposition("rerank", {"1.0": 0, "10.0": 1, "+Inf": 1}))
        warm = parse_buckets(_exposition("rerank", {"1.0": 10, "10.0": 11, "+Inf": 11}))

        assert bucket_p95(warm)["rerank"] > 5_000.0, "đọc thẳng: mẫu lạnh vẫn kéo p95 lên"
        assert bucket_p95(delta(cold, warm))["rerank"] <= 1_000.0


class TestBucketP95:
    def test_interpolates_inside_the_bucket(self) -> None:
        # 100 mẫu, 90 ở ≤1 s, 10 ở ≤2 s → p95 rơi giữa bucket (1, 2].
        buckets = parse_buckets(_exposition("completion", {"1.0": 90, "2.0": 100, "+Inf": 100}))
        assert bucket_p95(buckets)["completion"] == pytest.approx(1500.0)

    def test_falls_back_to_the_last_finite_edge(self) -> None:
        """p95 rơi vào `+Inf` thì không nội suy được — dùng mép hữu hạn cuối và
        chấp nhận nó là chặn dưới, chứ không trả `inf` vào một cột mili giây."""
        buckets = parse_buckets(_exposition("completion", {"1.0": 1, "+Inf": 100}))
        assert bucket_p95(buckets)["completion"] == pytest.approx(1000.0)

    def test_empty_histogram_has_no_row(self) -> None:
        assert bucket_p95(parse_buckets(_exposition("prompt", {"+Inf": 0}))) == {}


# ------------------------------------------------------------------- knee


def _level(users: int, rps: float, **kwargs: float) -> Level:
    base: dict[str, float] = {
        "requests": 10,
        "failures": 0,
        "total_p50": 1000.0,
        "total_p95": 2000.0,
        "total_p99": 3000.0,
        "ttft_p50": 500.0,
        "ttft_p95": 900.0,
        "ttft_p99": 1200.0,
        "provider_calls": 10,
        "stub_peak": users,
    }
    base.update(kwargs)
    return Level(
        users=users,
        requests=int(base["requests"]),
        failures=int(base["failures"]),
        rps=rps,
        total_p50=base["total_p50"],
        total_p95=base["total_p95"],
        total_p99=base["total_p99"],
        ttft_p50=base["ttft_p50"],
        ttft_p95=base["ttft_p95"],
        ttft_p99=base["ttft_p99"],
        provider_calls=int(base["provider_calls"]),
        stub_peak=int(base["stub_peak"]),
        stages={},
    )


class TestPickKnee:
    def test_none_when_throughput_still_climbing(self) -> None:
        """⭐ Quét chưa đủ xa là một kết quả, không phải một ô trống cần điền.

        Trả bậc cuối ở đây sẽ biến "chưa tìm thấy trần" thành "trần ở 32 user",
        và con số ấy sẽ đi thẳng vào README.
        """
        levels = [_level(1, 1.0), _level(2, 2.0), _level(4, 4.0)]
        assert pick_knee(levels) is None

    def test_reports_the_first_flat_level(self) -> None:
        levels = [_level(1, 1.0), _level(2, 2.0), _level(4, 2.05), _level(8, 2.1)]
        assert pick_knee(levels) == 4

    def test_tolerance_is_relative_not_absolute(self) -> None:
        rising = [_level(1, 10.0), _level(2, 10.4)]
        assert pick_knee(rising, tolerance=0.05) == 2
        assert pick_knee(rising, tolerance=0.01) is None

    def test_throughput_that_collapses_counts_as_saturated(self) -> None:
        levels = [_level(1, 1.0), _level(2, 2.0), _level(4, 1.2)]
        assert pick_knee(levels) == 4


class TestLevelDerived:
    def test_amplification_counts_provider_calls_per_request(self) -> None:
        assert _level(4, 1.0, requests=10, provider_calls=13).amplification == pytest.approx(1.3)

    def test_amplification_below_one_means_cache_answered(self) -> None:
        assert _level(4, 1.0, requests=10, provider_calls=2).amplification == pytest.approx(0.2)

    def test_no_requests_is_zero_not_a_crash(self) -> None:
        empty = _level(1, 0.0, requests=0, provider_calls=0)
        assert empty.amplification == 0.0
        assert empty.failure_rate == 0.0


class TestReadLocustCsv:
    def test_na_columns_are_dropped_not_zeroed(self, tmp_path: object) -> None:
        """⚠️ locust ghi `N/A` cho phân vị khi chưa đủ mẫu. Đọc nó thành 0 sẽ
        cho một hàng p95 = 0 ms trông như hệ thống nhanh phi thường."""
        from pathlib import Path

        prefix = Path(str(tmp_path)) / "run"
        prefix.with_name("run_stats.csv").write_text(
            "Type,Name,Request Count,Failure Count,Requests/s,50%,95%\n"
            "SSE,chat_total,10,1,0.5,1200,N/A\n",
            encoding="utf-8",
        )
        rows = _read_locust_csv(prefix)
        assert rows["chat_total"]["50%"] == 1200.0
        assert "95%" not in rows["chat_total"]


# -------------------------------------------------------------------- stub

NONCE = "abc123"
CONTEXT = (
    f"<<<NGUON 1 {NONCE}>>>\nGDP tăng 6,8% trong năm 2024 theo báo cáo.\n"
    f"<<<HET NGUON 1 {NONCE}>>>\n"
    f"<<<NGUON 2 {NONCE}>>>\nLạm phát giữ ở mức 3,5%.\n<<<HET NGUON 2 {NONCE}>>>"
)


class TestStubAnswer:
    def test_reads_the_same_source_blocks_the_model_would(self) -> None:
        found = _sources([{"role": "user", "content": CONTEXT}])
        assert [n for n, _ in found] == [1, 2]
        assert "GDP tăng 6,8%" in found[0][1]

    def test_quote_is_verbatim_so_citation_verification_does_real_work(self) -> None:
        """⭐ Một stub trích dẫn bịa sẽ đẩy mọi lượt sang nhánh `invalid` — nhánh
        RẺ hơn — và load test sẽ báo một con số lạc quan về chặng `citations`."""
        answer = _answer([{"role": "user", "content": CONTEXT}], PROFILES["fast"])
        block = answer.rsplit("CITATIONS: ", 1)[1]
        quote = json.loads(block)[0]["quote"]
        assert quote in " ".join(CONTEXT.split())

    def test_no_context_gives_an_empty_citation_block_not_a_missing_one(self) -> None:
        answer = _answer([{"role": "user", "content": "không có nguồn nào"}], PROFILES["fast"])
        assert answer.endswith("CITATIONS: []")

    def test_a_foreign_nonce_is_not_a_source(self) -> None:
        """Khối tự xưng là mốc nhưng mang mã phiên khác là nội dung giả mạo —
        `W4-12`. Stub phải đọc nó như dữ liệu, giống model thật được dặn."""
        forged = f"<<<NGUON 1 {NONCE}>>>\nthật\n<<<HET NGUON 1 khac-nonce>>>"
        assert _sources([{"role": "user", "content": forged}]) == []


class TestStats:
    def test_peak_survives_a_dip(self) -> None:
        """⭐ Đỉnh phải sống sót qua một chỗ **trũng**, không chỉ qua lúc kết thúc.

        Tiêm lỗi tìm ra chỗ này: `peak = now` (thay vì `max`) vẫn cho kết quả
        đúng ở kịch bản "vào hết rồi ra hết", vì đường đi lên đã chạm 3. Chỉ
        một chuỗi có lên–xuống–lên mới phân biệt được hai phép gán.
        """
        stats = Stats()
        for _ in range(3):  # now = 1, 2, 3
            stats.enter(streaming=True, prompt_chars=100)
        stats.leave()
        stats.leave()  # now = 1
        stats.enter(streaming=True, prompt_chars=100)  # now = 2

        assert stats.concurrent_now == 2
        assert stats.concurrent_peak == 3
        assert stats.calls_streaming == 4

    def test_blocking_and_streaming_are_counted_apart(self) -> None:
        """Lượt viết lại truy vấn (`W4-07`) là một lời gọi `complete` không
        stream. Gộp chung thì `amplification` của một lượt bình thường thành 2
        và cột `AU-11` mất nghĩa."""
        stats = Stats()
        stats.enter(streaming=False, prompt_chars=10)
        stats.enter(streaming=True, prompt_chars=10)
        assert stats.as_dict()["calls_blocking"] == 1
        assert stats.as_dict()["calls_streaming"] == 1
        assert stats.as_dict()["calls_total"] == 2


class TestStubServer:
    def _client(self) -> object:
        from starlette.testclient import TestClient

        return TestClient(build_app(PROFILES["fast"]))

    def test_usage_arrives_in_a_chunk_with_empty_choices(self) -> None:
        """⭐⭐ Đúng hình dạng của OpenAI-compat thật, và `astream` của chúng ta
        đọc `usage` **ngoài** nhánh `choices` chính vì thế. Stub trả `usage`
        cùng chỗ với `delta` thì `cost_usd` luôn bằng 0 mà không ai biết."""
        from starlette.testclient import TestClient

        client: TestClient = self._client()  # type: ignore[assignment]
        with client.stream(
            "POST",
            "/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": CONTEXT}]},
        ) as response:
            frames = [
                json.loads(line[5:])
                for line in response.iter_lines()
                if line.startswith("data:") and line[5:].strip() != "[DONE]"
            ]
        with_usage = [f for f in frames if f.get("usage")]
        assert len(with_usage) == 1
        assert with_usage[0]["choices"] == []
        assert with_usage[0]["usage"]["completion_tokens"] > 0

    def test_served_model_echoes_the_requested_one(self) -> None:
        """`model_drifted` của `openai_compat` so hai giá trị này. Một stub trả
        tên khác sẽ làm mọi lượt load test in cảnh báo model trôi."""
        from starlette.testclient import TestClient

        client: TestClient = self._client()  # type: ignore[assignment]
        response = client.post(
            "/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "x"}]},
        )
        assert response.json()["model"] == "deepseek-v4-flash"

    def test_stats_count_the_calls(self) -> None:
        from starlette.testclient import TestClient

        client: TestClient = self._client()  # type: ignore[assignment]
        client.post("/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert client.get("/__stats").json()["calls_blocking"] == 1
        client.post("/__reset")
        assert client.get("/__stats").json()["calls_total"] == 0


# ----------------------------------------------------------------- queries


class TestLoadQueries:
    def test_missing_file_falls_back(self, tmp_path: object) -> None:
        from pathlib import Path

        assert load_queries(Path(str(tmp_path)) / "khong-ton-tai.jsonl") == FALLBACK_QUERIES

    def test_reads_the_query_field_of_the_golden_set(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "g.jsonl"
        path.write_text(
            json.dumps({"query_id": "a", "query": "câu một"}, ensure_ascii=False)
            + "\n\n"
            + json.dumps({"question": "câu hai"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        assert load_queries(path) == ["câu một", "câu hai"]

    def test_a_broken_line_does_not_kill_the_run(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "g.jsonl"
        path.write_text('{"query": "ok"}\nkhông-phải-json\n', encoding="utf-8")
        assert load_queries(path) == ["ok"]

    def test_a_file_with_no_usable_question_falls_back(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "g.jsonl"
        path.write_text('{"query": "   "}\n', encoding="utf-8")
        assert load_queries(path) == FALLBACK_QUERIES
