"""Nguồn câu hỏi cho load test. `W6-05`.

Tách khỏi `locustfile.py` vì một lý do cụ thể: `import locust` kéo theo
**gevent**, và gevent monkey-patch thư viện chuẩn ngay lúc import. Một file test
đơn vị chạm vào nó sẽ vá `socket`/`ssl`/`threading` cho cả phiên pytest — cùng
phiên đang chạy `pytest-asyncio`. Nên tầng CI `unit` cố ý **không** cài extra
`loadtest`, và mọi thứ đáng kiểm phải nằm ở chỗ nhập được mà không cần locust.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["DEFAULT_QUERIES", "FALLBACK_QUERIES", "load_queries"]

DEFAULT_QUERIES = Path("data/golden/golden_v1.jsonl")

#: Câu hỏi dự phòng khi không có golden set trên máy chạy load test. Cố ý ngắn
#: và chung chung: chúng chỉ cần đi qua đúng đường mã, không cần chấm điểm.
FALLBACK_QUERIES = [
    "Tăng trưởng GDP của Việt Nam gần đây ra sao?",
    "What are the main risks to Vietnam's economy?",
    "Nợ công của Việt Nam được đánh giá thế nào?",
    "How does the World Bank describe Vietnam's banking sector?",
    "Chính sách tài khóa được khuyến nghị là gì?",
]


def load_queries(path: Path | None = None) -> list[str]:
    """Câu hỏi thật từ golden set, để prompt có độ dài thật.

    ⚠️ Độ dài prompt **là** một biến của phép đo: `TD-51` cắt lịch sử theo số
    message chứ không theo ngân sách token, nên một load test dùng câu hỏi ngắn
    sẽ không bao giờ chạm vào chế độ hỏng ấy.

    Dòng hỏng thì **bỏ qua dòng đó**, không làm hỏng cả lần chạy: một load test
    không chạy được vì một dòng JSON lỗi là một load test không ai chạy.
    """
    target = path or DEFAULT_QUERIES
    if not target.exists():
        return list(FALLBACK_QUERIES)
    out: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        question = row.get("question") or row.get("query")
        if isinstance(question, str) and question.strip():
            out.append(question.strip())
    return out or list(FALLBACK_QUERIES)
