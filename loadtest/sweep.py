"""Quét concurrency và tìm **điểm bão hoà**. `W6-05`.

    uv run python -m loadtest.sweep --levels 1,2,4,8,16,32 --duration 60 \\
        --out plans/reports/runs/w605-sweep-fast.json

Mỗi bậc chạy một tiến trình `locust --headless` riêng, rồi gom ba nguồn số về
cùng một hàng:

1. **CSV của locust** — phân vị `chat_ttft` và `chat_total`, tỉ lệ hỏng, RPS.
2. **`/__stats` của stub** — số lời gọi nhà cung cấp *thực sự* phát ra
   (`AU-11`), và `concurrent_peak` để chứng minh tải đã chồng nhau thật.
3. **`/metrics` của server** — `rag_stage_duration_seconds` theo từng chặng, thứ
   nói **chặng nào** phình ra khi tải tăng. Không có nó thì đường cong chỉ nói
   "chậm hơn" mà không nói tại sao.

## ⭐⭐ Điểm bão hoà là chỗ **thông lượng ngừng tăng**, không phải chỗ p95 xấu

Hai định nghĩa cho hai kết luận khác nhau. p95 xấu đi ngay từ bậc thứ hai ở gần
như mọi hệ thống — đó là hàng đợi, không phải bão hoà. Bão hoà là khi tăng
người dùng **không** làm tăng số request hoàn thành mỗi giây nữa: từ đó trở đi
mọi người dùng thêm vào chỉ đứng xếp hàng dài hơn.

Module này báo cả hai cột và chọn điểm bão hoà theo cột RPS (bậc đầu tiên mà
RPS không hơn bậc trước quá `--knee-tolerance`, mặc định 5%), vì đó là con số
quyết định "một instance phục vụ được bao nhiêu người".

## ⚠️ Một tiến trình locust cũng có trần của chính nó

Ở bậc cao, nếu máy phát tải là thứ bão hoà trước thì đường cong đo máy phát
tải. Bằng chứng phân biệt: CPU của tiến trình locust, và `concurrent_peak` của
stub — nếu peak không theo kịp số user thì tải chưa bao giờ chồng như khai báo.
Cột `stub_peak` có mặt trong báo cáo chính vì thế.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Level", "bucket_p95", "delta", "parse_buckets", "pick_knee", "run_level", "sweep"]

logger = logging.getLogger("loadtest.sweep")

DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Level:
    users: int
    requests: int
    failures: int
    rps: float
    total_p50: float
    total_p95: float
    total_p99: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    provider_calls: int
    stub_peak: int
    stages: dict[str, float]
    """`chặng -> p95 (ms)` đọc từ histogram Prometheus. Xem `TD-76`."""

    @property
    def failure_rate(self) -> float:
        return self.failures / self.requests if self.requests else 0.0

    @property
    def amplification(self) -> float:
        """Số lời gọi nhà cung cấp trên mỗi request hoàn thành.

        `1.0` = mỗi request một lời gọi. Lớn hơn 1 nghĩa là có lượt viết lại
        truy vấn (`W4-07`, một lời gọi `complete` nữa). **Nhỏ hơn** 1 nghĩa là
        cache đang trả lời — và ở chế độ duplicate của `AU-11`, đó chính là con
        số phải nhìn: single-flight có nghĩa là nó tiến về `1/N`.
        """
        return self.provider_calls / self.requests if self.requests else 0.0


def _get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urlopen(Request(url), timeout=timeout) as response:
        payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        return payload


def _post(url: str, timeout: float = 5.0) -> None:
    with urlopen(Request(url, method="POST", data=b""), timeout=timeout):
        pass


def _read_metrics(url: str, token: str, timeout: float = 5.0) -> str:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=timeout) as response:
        text: str = response.read().decode("utf-8")
        return text


Buckets = dict[str, dict[float, float]]
"""`chặng -> {mép trên (giây): số đếm tích luỹ}`."""


def parse_buckets(exposition: str) -> Buckets:
    """Đọc `rag_stage_duration_seconds_bucket` thành số đếm **tích luỹ**."""
    out: Buckets = {}
    for line in exposition.splitlines():
        if not line.startswith("rag_stage_duration_seconds_bucket"):
            continue
        labels, _, value = line.partition(" ")
        stage = _label(labels, "stage")
        le = _label(labels, "le")
        if stage is None or le is None:
            continue
        edge = float("inf") if le == "+Inf" else float(le)
        out.setdefault(stage, {})[edge] = float(value)
    return out


def delta(before: Buckets, after: Buckets) -> Buckets:
    """Histogram **của riêng một bậc**, bằng cách trừ hai ảnh chụp.

    ⭐⭐ Đây không phải tiện nghi mà là điều kiện để con số có nghĩa. Histogram
    Prometheus là **counter tích luỹ**: đọc thẳng nó sau bậc thứ ba cho ra phân
    vị của *cả ba* bậc cộng lại, cộng thêm lượt rerank lạnh 7,3 giây của
    `TD-72` mà request đầu tiên sau khi container lên phải chịu. Lượt đo thử
    đầu tiên của `W6-05` báo `rerank p95 = 9.375 ms` ở u=1 đúng vì thế — một
    con số không mô tả bậc nào cả.

    Trừ hai ảnh chụp là đúng phép `rate()` của Prometheus, làm bằng tay.
    """
    out: Buckets = {}
    for stage, edges in after.items():
        base = before.get(stage, {})
        diff = {edge: count - base.get(edge, 0.0) for edge, count in edges.items()}
        if diff.get(float("inf"), 0.0) > 0:
            out[stage] = diff
    return out


def bucket_p95(buckets: Buckets) -> dict[str, float]:
    """p95 mỗi chặng (ms) từ số đếm bucket.

    ⚠️ `TD-76`: đây là **nội suy tuyến tính trong bucket**, không phải một phép
    đo. Ở lưu lượng thấp nó lệch thấy rõ. Nó có mặt ở đây để trả lời "chặng nào
    phình ra", một câu hỏi về *hình dạng*, chứ không để công bố một con số.
    """
    out: dict[str, float] = {}
    for stage, edges in buckets.items():
        pairs = sorted(edges.items())
        total = pairs[-1][1] if pairs else 0.0
        if total <= 0:
            continue
        target = 0.95 * total
        previous_edge, previous_count = 0.0, 0.0
        for edge, count in pairs:
            if count >= target:
                if edge == float("inf"):
                    out[stage] = previous_edge * 1000.0
                    break
                span = count - previous_count
                frac = (target - previous_count) / span if span else 0.0
                out[stage] = (previous_edge + (edge - previous_edge) * frac) * 1000.0
                break
            previous_edge, previous_count = edge, count
    return out


def _label(labels: str, name: str) -> str | None:
    marker = f'{name}="'
    start = labels.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = labels.find('"', start)
    return labels[start:end] if end > start else None


def _read_locust_csv(prefix: Path) -> dict[str, dict[str, float]]:
    """`tên -> {cột: số}` từ `<prefix>_stats.csv`."""
    path = prefix.with_name(prefix.name + "_stats.csv")
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Name", "")
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key in {"Type", "Name"}:
                    continue
                # ⚠️ locust ghi `N/A` cho phân vị khi chưa đủ mẫu, và
                # `csv.DictReader` trả `None` cho cột thiếu. Đọc chúng thành 0
                # sẽ cho một hàng p95 = 0 ms trông như hệ thống nhanh phi
                # thường. Bản đầu còn kiểm `value in {None, "", "N/A"}` TRƯỚC
                # khối này — tiêm lỗi cho thấy nhánh ấy **chết**: `float("")`
                # và `float("N/A")` đều ném `ValueError`, chỉ `float(None)` ném
                # `TypeError`. Bắt cả hai kiểu là đủ, và không có hai chỗ cùng
                # định nghĩa "giá trị không đọc được".
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows[name] = parsed
    return rows


def run_level(
    users: int,
    *,
    host: str,
    duration: int,
    out_prefix: Path,
    stub_url: str,
    metrics_url: str,
    metrics_token: str,
    env_extra: dict[str, str] | None = None,
) -> Level:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        _post(f"{stub_url}/__reset")
    except URLError as exc:
        logger.warning("không đặt lại được sổ đếm stub (%s) — cột AU-11 sẽ cộng dồn", exc)

    # Ảnh chụp histogram TRƯỚC bậc này. Xem `delta()`.
    try:
        before = parse_buckets(_read_metrics(metrics_url, metrics_token))
    except (URLError, OSError) as exc:
        logger.warning("không đọc được /metrics trước bậc (%s)", exc)
        before = {}

    import os

    env = {**os.environ, **(env_extra or {})}
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "loadtest/locustfile.py",
        "--headless",
        "--host",
        host,
        "-u",
        str(users),
        # Spawn hết trong ~1 giây: mục tiêu là một bậc tải phẳng, không phải một
        # đường dốc. Đường dốc trộn lẫn hai chế độ vào cùng một phân vị.
        "-r",
        str(max(1, users)),
        "-t",
        f"{duration}s",
        "--csv",
        str(out_prefix),
        "--only-summary",
    ]
    logger.info("bậc u=%d trong %ds …", users, duration)
    proc = subprocess.run(command, check=False, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("locust thoát %d: %s", proc.returncode, proc.stderr[-800:])

    stats = _read_locust_csv(out_prefix)
    total = stats.get("chat_total", {})
    ttft = stats.get("chat_ttft", {})

    try:
        stub = _get_json(f"{stub_url}/__stats")
    except URLError:
        stub = {}
    try:
        after = parse_buckets(_read_metrics(metrics_url, metrics_token))
        stages = bucket_p95(delta(before, after))
    except (URLError, OSError) as exc:
        logger.warning("không đọc được /metrics (%s)", exc)
        stages = {}

    return Level(
        users=users,
        requests=int(total.get("Request Count", 0)),
        failures=int(total.get("Failure Count", 0)),
        rps=float(total.get("Requests/s", 0.0)),
        total_p50=float(total.get("50%", 0.0)),
        total_p95=float(total.get("95%", 0.0)),
        total_p99=float(total.get("99%", 0.0)),
        ttft_p50=float(ttft.get("50%", 0.0)),
        ttft_p95=float(ttft.get("95%", 0.0)),
        ttft_p99=float(ttft.get("99%", 0.0)),
        provider_calls=int(stub.get("calls_total", 0)),
        stub_peak=int(stub.get("concurrent_peak", 0)),
        stages={k: round(v, 1) for k, v in sorted(stages.items())},
    )


def pick_knee(levels: Sequence[Level], tolerance: float = 0.05) -> int | None:
    """Bậc đầu tiên mà RPS **không** hơn bậc trước quá `tolerance`.

    Trả `None` khi thông lượng còn tăng đều tới bậc cuối — và đó là một kết quả
    hợp lệ, phải nói ra chứ không được điền đại bậc cuối vào: nó nghĩa là quét
    chưa đủ xa, không phải là hệ thống bão hoà ở đó.
    """
    for previous, current in itertools.pairwise(levels):
        if previous.rps <= 0:
            continue
        if current.rps <= previous.rps * (1.0 + tolerance):
            return current.users
    return None


def sweep(
    levels: Sequence[int],
    *,
    host: str,
    duration: int,
    out: Path,
    stub_url: str,
    metrics_url: str,
    metrics_token: str,
    cooldown: int,
    label: str,
    env_extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows: list[Level] = []
    for index, users in enumerate(levels):
        if index:
            time.sleep(cooldown)
        rows.append(
            run_level(
                users,
                host=host,
                duration=duration,
                out_prefix=out.with_name(f"{out.stem}-u{users}"),
                stub_url=stub_url,
                metrics_url=metrics_url,
                metrics_token=metrics_token,
                env_extra=env_extra,
            )
        )
        last = rows[-1]
        logger.info(
            "  u=%-3d rps=%.2f  ttft p95=%.0f  total p95=%.0f  hỏng=%.1f%%  gọi provider=%d",
            last.users,
            last.rps,
            last.ttft_p95,
            last.total_p95,
            last.failure_rate * 100,
            last.provider_calls,
        )

    payload = {
        "label": label,
        "host": host,
        "duration_s": duration,
        "levels": [asdict(r) for r in rows],
        "knee_users": pick_knee(rows),
        "peak_rps": max((r.rps for r in rows), default=0.0),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("đã ghi %s", out.as_posix())
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m loadtest.sweep",
        description="W6-05 — quét concurrency, tìm điểm bão hoà",
    )
    parser.add_argument("--levels", default=",".join(str(n) for n in DEFAULT_LEVELS))
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--cooldown", type=int, default=10)
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--stub-url", default="http://127.0.0.1:8199")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--metrics-token", default="")
    parser.add_argument("--label", default="w605")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    sweep(
        levels,
        host=args.host,
        duration=args.duration,
        out=args.out,
        stub_url=args.stub_url,
        metrics_url=args.metrics_url,
        metrics_token=args.metrics_token,
        cooldown=args.cooldown,
        label=args.label,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
