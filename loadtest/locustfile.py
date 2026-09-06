"""Load test `POST /chat` theo concurrency. `W6-05`.

    uv run locust -f loadtest/locustfile.py --headless \\
        -u 8 -r 8 -t 60s --host http://127.0.0.1:8000 \\
        --csv plans/reports/runs/w605-c8

## ⭐⭐ Đo **hai** con số, vì hệ thống này stream

`p95 end-to-end` của `W5-05`/`W5-11` là thời gian tới byte **cuối**. Với một
API stream, con số đó không phải thứ người dùng cảm thấy: họ thấy chữ đầu tiên
hiện ra, rồi đọc trong lúc phần còn lại chảy về. Ngân sách 3.500 ms của `G2`
được viết ở `W1`, trước khi có streaming, nên nó đang đo một thứ mà giao diện
đã làm cho không còn quan trọng như cũ.

Nên mỗi request ở đây phát ra **hai** sự kiện locust:

* `chat_ttft` — tới khung `delta` đầu tiên. Đây là độ trễ có người đang đợi.
* `chat_total` — tới khung `done`. Đây là thứ so được với số cũ.

Đổi định nghĩa của một metric để nó vừa với ngưỡng là gian lận. Báo cả hai, nói
rõ cái nào so với ngưỡng cũ, rồi mới bàn tới việc ngưỡng nên là gì — theo thứ
tự đó.

## ⭐ `done` là điều kiện dừng, không phải hết dòng

Chính docstring của `POST /chat` đã ghi: một dòng `delta` dừng lại giống hệt
nhau khi model nói xong, khi kết nối đứt, và khi provider hết hạn mức. Một load
test đọc tới EOF rồi đánh dấu thành công sẽ báo 100% thành công trong đúng lúc
hệ thống hỏng — chế độ hỏng mà `W6-05` sinh ra để tìm.

## ⭐ Chế độ `LOADTEST_DUPLICATE=1` là dụng cụ đo `AU-11`

Mọi user gửi **cùng một câu hỏi** cùng lúc. Không có single-flight thì N
request đồng thời = N lần gọi nhà cung cấp, và `loadtest/stub_llm.py:/__stats`
đếm được đúng con số ấy từ phía bị gọi. Chạy cùng câu hỏi *nối đuôi* thì cache
sẽ trả lời từ lượt thứ hai — nên phép đo chỉ có nghĩa khi chúng thật sự chồng
nhau, và `concurrent_peak` của stub là bằng chứng cho điều đó.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from locust import HttpUser, between, constant, events, task

# ⚠️ Import TUYỆT ĐỐI, không tương đối: locust nạp file này bằng đường dẫn
# (`-f loadtest/locustfile.py`), không phải như một module trong gói — nên
# `from .queries import …` ném `ImportError: attempted relative import`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loadtest.queries import load_queries

__all__ = ["ChatUser"]

_QUERIES_PATH = os.environ.get("LOADTEST_QUERIES")
_QUERIES = load_queries(Path(_QUERIES_PATH) if _QUERIES_PATH else None)
_DUPLICATE = os.environ.get("LOADTEST_DUPLICATE") == "1"
_API_KEY = os.environ.get("LOADTEST_API_KEY", "")
_TOP_K = int(os.environ.get("LOADTEST_TOP_K", "5"))


@events.test_start.add_listener  # type: ignore[untyped-decorator]  # locust: không stub
def _announce(environment: Any, **_kwargs: Any) -> None:
    if not _API_KEY:
        msg = "thiếu LOADTEST_API_KEY — mọi request sẽ 401 và đường cong sẽ đo tầng auth"
        raise RuntimeError(msg)
    print(
        f"[w6-05] {len(_QUERIES)} câu hỏi · top_k={_TOP_K} · "
        f"{'DUPLICATE (đo AU-11)' if _DUPLICATE else 'xoay vòng'}"
    )


class ChatUser(HttpUser):
    """Một người dùng: hỏi, đọc stream tới `done`, nghỉ, hỏi tiếp.

    `wait_time` bằng 0 ở chế độ duplicate — mục tiêu ở đó là **chồng lấn**, chứ
    không phải mô phỏng người thật.
    """

    wait_time = constant(0) if _DUPLICATE else between(1, 3)

    def on_start(self) -> None:
        self.client.headers.update({"Authorization": f"Bearer {_API_KEY}"})
        # Mỗi user một câu hỏi cố định ở chế độ duplicate → tất cả cùng một câu.
        self._fixed = _QUERIES[0] if _DUPLICATE else None

    @task
    def ask(self) -> None:
        question = self._fixed or random.choice(_QUERIES)
        started = time.perf_counter()
        ttft: float | None = None
        done = False
        error: str | None = None
        body = {"message": question, "top_k": _TOP_K}

        try:
            with self.client.post(
                "/chat",
                json=body,
                stream=True,
                catch_response=True,
                name="/chat",
                timeout=120,
            ) as response:
                if response.status_code != 200:
                    error = f"HTTP {response.status_code}"
                    response.failure(error)
                else:
                    event = ""
                    for raw in response.iter_lines(decode_unicode=True):
                        if raw is None:
                            continue
                        line = raw.strip()
                        if line.startswith("event:"):
                            event = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        if event == "delta" and ttft is None:
                            ttft = (time.perf_counter() - started) * 1000.0
                        elif event == "done":
                            done = True
                            break
                        elif event == "error":
                            error = f"khung error: {line[5:].strip()[:160]}"
                            break
                    if error is None and not done:
                        # Xem docstring module: hết dòng KHÔNG phải là xong.
                        error = "stream kết thúc mà không có khung `done`"
                    if error:
                        response.failure(error)
                    else:
                        response.success()
        except Exception as exc:  # locust phải thấy MỌI lỗi client, kể cả lỗi lạ
            error = f"{type(exc).__name__}: {exc}"[:200]

        total = (time.perf_counter() - started) * 1000.0
        self.environment.events.request.fire(
            request_type="SSE",
            name="chat_total",
            response_time=total,
            response_length=0,
            exception=RuntimeError(error) if error else None,
        )
        # ⭐ TTFT chỉ phát khi có token thật. Ghi `total` vào ô TTFT cho lượt
        # hỏng sẽ kéo phân vị TTFT lên bằng chính những lượt không có token nào —
        # tức metric "người dùng đợi bao lâu tới chữ đầu" sẽ mang cả những lượt
        # không bao giờ có chữ đầu.
        if ttft is not None:
            self.environment.events.request.fire(
                request_type="SSE",
                name="chat_ttft",
                response_time=ttft,
                response_length=0,
                exception=None,
            )
