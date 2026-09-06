"""Đo bộ che log trên đúng đường log của production — `W6-06`.

    uv run python scripts/probe_log_redaction.py --out plans/reports/probes/w606-log-redaction.json

## ⭐⭐ Vì sao cần một probe khi đã có test

Test hỏi "hàm `redact_pii` có che email không". Probe hỏi một câu khác, và đó là
câu đã sai: **một dòng log đi hết đường thật — logger → filter → formatter →
stream — thì cái ra tới stream còn gì?** Hai câu ấy khác nhau ở đúng chỗ lỗ hổng
nằm: `logger.exception()` không đặt gì vào `record.msg`, nên mọi test gọi thẳng
`redact_pii` đều xanh trong khi traceback đi ra ngoài nguyên vẹn.

## ⭐ Hai nhánh trong một lượt chạy

Nhánh **chứng** dựng lại `RedactingFilter` **đúng như trước** `W6-06` rồi ghi
cùng ba bản ghi ấy. Không có nhánh chứng thì "0 rò" không phân biệt được với
"probe không đo được gì" — cùng bài học của lượt tiêm giả ở `W6-01`, nên probe
trả mã lỗi khi nhánh chứng *sạch*.

⚠️ Nhánh chứng phải chép **nguyên** hành vi cũ, không phải một bản rút gọn: bản
đầu của probe này chỉ che `record.msg` và vì thế tính luôn cả `record.args` vào
phần "bản vá cứu được" — trong khi bản cũ đã che `args` từ `W4-12`. Số chênh
lệch khi ấy phóng đại đúng một bản ghi.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any

from rag_core.generation.guardrails import _pii_only
from serving.core.logging import JsonFormatter, RedactingFilter

#: Giá trị mồi. Không phải bí mật thật — chúng chỉ cần **trông giống** đủ để
#: chạm luật, và một bí mật thật trong một file commit vào git là đúng thứ
#: `job_bundle.scan_text` tồn tại để chặn.
PLANTED = {
    "email": "nguyenvana@example.com",
    "platform_key": "rag_" + "0123456789abcdef" * 2 + "ABCDEFGH",
    "provider_key": "sk-" + "a1b2c3d4e5f6a7b8" * 2,
}


class _PreW606Filter(logging.Filter):
    """Nhánh chứng: bản `RedactingFilter` **đúng như trước** `W6-06`.

    ⚠️ Chép lại nguyên hành vi cũ — `msg` + `args` + mọi chuỗi trong `__dict__`,
    che bằng **vế PII thôi** (`_pii_only`). Một nhánh chứng yếu hơn bản thật (ví
    dụ chỉ đụng `msg`) sẽ tính cả những lỗ mà bản cũ vốn đã bịt, và con số chênh
    lệch khi ấy nói dối về giá trị của bản vá.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _pii_only(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _pii_only(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            else:
                record.args = tuple(_pii_only(a) if isinstance(a, str) else a for a in record.args)
        for key, value in list(record.__dict__.items()):
            if isinstance(value, str) and key not in ("name", "levelname", "pathname", "funcName"):
                record.__dict__[key] = _pii_only(value)
        return True


class _ProviderError(RuntimeError):
    pass


def _emit(log_filter: logging.Filter, *, patched: bool) -> list[dict[str, Any]]:
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(log_filter)
    logger = logging.getLogger(f"probe.w606.{'patched' if patched else 'control'}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.warning("khách %s hỏi", PLANTED["email"])
    try:
        raise _ProviderError(
            f"provider trả 401: {{'echo': 'Bearer {PLANTED['provider_key']}'}}, "
            f"liên hệ {PLANTED['email']}"
        )
    except _ProviderError:
        # ⚠️ Đúng dòng mà docstring của `RedactingFilter` lấy làm ví dụ, và đúng
        # dòng mà bản trước `W6-06` không phủ.
        logger.exception("gọi provider thất bại")
    logger.warning("header gửi đi: Authorization: Bearer %s", PLANTED["platform_key"])

    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def _leaks(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Bản ghi thứ mấy để lọt giá trị mồi nào (1-indexed, theo thứ tự ghi)."""
    out: dict[str, list[int]] = {name: [] for name in PLANTED}
    for index, record in enumerate(records, 1):
        blob = json.dumps(record, ensure_ascii=False)
        for name, value in PLANTED.items():
            if value in blob:
                out[name].append(index)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    control = _emit(_PreW606Filter(), patched=False)
    patched = _emit(RedactingFilter(), patched=True)
    report = {
        "task": "W6-06",
        "what": "một dòng log đi hết đường thật thì còn rò gì",
        "records_per_arm": len(patched),
        "control": {
            "what": "bản trước W6-06 — msg+args+__dict__, che PII thôi, không phủ exc_info",
            "leaks": _leaks(control),
        },
        "patched": {
            "what": "RedactingFilter hiện tại — phủ cả exc_info và credential",
            "leaks": _leaks(patched),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nđã ghi {args.out}")
    # Nhánh vá phải sạch; nhánh chứng phải BẨN — một nhánh chứng sạch nghĩa là
    # probe không còn đo được gì, và im lặng thì nó trông y hệt một lượt tốt.
    leaked = any(v for v in report["patched"]["leaks"].values())  # type: ignore[index]
    blind = not any(v for v in report["control"]["leaks"].values())  # type: ignore[index]
    if leaked:
        print("\n❌ nhánh vá vẫn rò")
        return 1
    if blind:
        print("\n❌ nhánh chứng KHÔNG rò — probe mù, số 0 ở trên vô nghĩa")
        return 2
    print("\n✅ nhánh vá sạch, nhánh chứng rò (probe đo được)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
