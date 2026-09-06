"""`TD-52` — bảng confusables không đầy đủ: đo trước, quyết sau. `W6-06`.

    uv run python scripts/confusable_probe.py --out plans/reports/probes/w606-td52-confusables.json

## Câu hỏi

Nợ đề xuất "dùng bảng confusables chuẩn của Unicode, đo lại dương tính giả trên
20.424 chunk". Probe này hỏi một câu rộng hơn, vì bảng chỉ là **một** cách:

1. Bảng hiện tại **bỏ sót** bao nhiêu, đo trên một tập ký tự thay-thế-được thật?
2. Luật **trộn hệ chữ** (`mixed_script_words`) bắt được bao nhiêu trong số đó?
3. Nó đánh nhầm bao nhiêu chunk trong **20.424 chunk corpus thật**?

Câu 3 là câu quyết định. Một bộ dò tiêm chỉ **gắn cờ** chứ không bỏ chunk
(`W4-12`), nhưng một cờ kêu ở 5% số chunk là một cờ không ai đọc nữa — và khi ấy
nó tệ hơn không có, vì nó trông giống một lớp phòng thủ.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

from rag_core.chunking.base import ChunkingConfig
from rag_core.chunking.fixed import split_recursive
from rag_core.generation.guardrails import (
    mixed_script_words,
    normalise_for_scan,
    scan_injection,
)

CORPUS = Path("data/corpus")
CHUNK_SIZE = 1000
#: Cùng bộ separator mà `ChunkSizing` mặc định dùng — probe phải chia y hệt
#: đường thật, nếu không thì con số 20.424 của `W4-12` không so được.
_SIZING = ChunkingConfig()
SEPARATORS = list(_SIZING.separators)
OVERLAP = _SIZING.chunk_overlap

#: Từ khoá Latin mà kẻ tấn công cần giữ nguyên **hình dáng** để người đọc không
#: nghi, nhưng phải phá **mã** để luật regex không khớp.
BAIT = "ignore"

#: Mọi ký tự Unicode mà NFKC **không** gập về ASCII nhưng trông giống một chữ
#: cái ASCII. Dò bằng cách quét: rẻ hơn và trung thực hơn một danh sách tôi tự
#: nhớ ra — chính cái danh sách ấy là thứ `TD-52` đang phàn nàn.
_LETTER_CATEGORIES = frozenset({"Lu", "Ll", "Lo", "Lt"})
"""⚠️⚠️ Bản đầu của probe **không** lọc theo category, và nó tự tạo ra 12 ca
"bỏ lọt" giả: `COMBINING LATIN SMALL LETTER E` (`Mn`) và `PARENTHESIZED LATIN
SMALL LETTER E` (`So`) đều mang chữ `E` trong tên nên lọt qua phép lọc theo tên,
nhưng cái thứ nhất vẽ ra một chữ e tí xíu **phía trên** chữ bên cạnh và cái thứ
hai vẽ ra `(e)`. Không cái nào lừa được mắt người, tức không cái nào là một phép
tấn công — mà cả hai vẫn nằm trong mẫu số và kéo tỉ lệ bắt được xuống 85%.

Bài học: một phép đo tự sinh mẫu thử thì **mẫu số cũng là một giả thuyết**, và
nó sai theo hướng làm hệ thống trông tệ hơn thực tế. Ở lượt trước tôi suýt ghi
con số 85% vào báo cáo."""


def _lookalikes() -> dict[str, list[str]]:
    """Với mỗi chữ cái ASCII, những **chữ cái** Unicode có tên gợi đúng chữ ấy."""
    out: dict[str, list[str]] = {c: [] for c in "abcdefghijklmnopqrstuvwxyz"}
    for code in range(0x80, 0x2500):
        ch = chr(code)
        if unicodedata.category(ch) not in _LETTER_CATEGORIES:
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        parts = name.split()
        # "CYRILLIC SMALL LETTER A", "GREEK SMALL LETTER ALPHA" → chỉ nhận dạng
        # "… LETTER X" một ký tự, tức những cái thật sự trông giống.
        if len(parts) < 3 or parts[-2] != "LETTER" or len(parts[-1]) != 1:
            continue
        target = parts[-1].lower()
        if target in out and unicodedata.normalize("NFKC", ch) != target:
            out[target].append(ch)
    return out


def _substitutions() -> list[tuple[str, str]]:
    """Mọi cách thay **một** chữ của `BAIT` bằng một ký tự nhìn giống."""
    table = _lookalikes()
    cases: list[tuple[str, str]] = []
    for index, letter in enumerate(BAIT):
        for swap in table.get(letter, ()):
            cases.append((f"{letter}->{swap!r}@{index}", BAIT[:index] + swap + BAIT[index + 1 :]))
    return cases


def _chunks() -> list[str]:
    out: list[str] = []
    for path in sorted(CORPUS.glob("*.txt")):
        out.extend(
            split_recursive(path.read_text(encoding="utf-8"), SEPARATORS, CHUNK_SIZE, OVERLAP)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cases = _substitutions()
    folded_ok = [name for name, text in cases if BAIT in normalise_for_scan(text)]
    mixed_ok = [name for name, text in cases if mixed_script_words(text)]
    caught = {*folded_ok, *mixed_ok}

    print(f"{len(cases)} cách thay một chữ của {BAIT!r} bằng ký tự nhìn giống")
    print(f"  bảng gập bắt được   : {len(folded_ok):>4} ({len(folded_ok) / len(cases):.1%})")
    print(f"  luật trộn hệ chữ    : {len(mixed_ok):>4} ({len(mixed_ok) / len(cases):.1%})")
    print(f"  hợp hai cơ chế      : {len(caught):>4} ({len(caught) / len(cases):.1%})")

    chunks = _chunks()
    start = time.perf_counter()
    hits = [(i, mixed_script_words(c)) for i, c in enumerate(chunks)]
    elapsed = time.perf_counter() - start
    flagged = [(i, words) for i, words in hits if words]
    whole = [i for i, c in enumerate(chunks) if scan_injection(c)]
    print(f"\n{len(chunks)} chunk corpus thật")
    print(f"  bị luật trộn hệ chữ gắn cờ: {len(flagged)} ({len(flagged) / len(chunks):.4%})")
    print(f"  chi phí quét              : {elapsed / len(chunks) * 1e6:.0f} µs/chunk")
    print(f"  TOÀN bộ 11 luật gắn cờ    : {len(whole)} ({len(whole) / len(chunks):.4%})")

    sample = [{"chunk": i, "words": sorted(set(words))[:5]} for i, words in flagged[:15]]
    for row in sample:
        print(f"    chunk {row['chunk']:>6}: {row['words']}")

    report: dict[str, Any] = {
        "task": "W6-06 / TD-52",
        "bait": BAIT,
        "substitutions": len(cases),
        "caught_by_fold_table": len(folded_ok),
        "caught_by_mixed_script": len(mixed_ok),
        "caught_by_either": len(caught),
        "missed_by_both": sorted(name for name, _ in cases if name not in caught),
        "corpus_chunks": len(chunks),
        "corpus_flagged": len(flagged),
        "corpus_flag_rate": round(len(flagged) / len(chunks), 6),
        "scan_us_per_chunk": round(elapsed / len(chunks) * 1e6, 1),
        "corpus_flagged_all_rules": len(whole),
        "corpus_flag_rate_all_rules": round(len(whole) / len(chunks), 6),
        "flagged_sample": sample,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nđã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
