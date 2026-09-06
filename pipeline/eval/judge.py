"""LLM judge — chấm nhãn, có cache nội dung, có trần chi phí. `W5-03`.

Mọi metric sinh (`W5-01`), độ chính xác trích dẫn (`W5-02`) và phần hiệu chỉnh
với người (`W5-04`) đều đi qua đúng lớp này. Bốn quyết định dưới đây quyết định
những con số ấy có nghĩa hay không.

## ⭐⭐ 1. Cache không phải để tiết kiệm tiền. Nó là thứ duy nhất làm eval tái lập được.

`TD-41` đã đo và ghi lại: DeepSeek ở `temperature=0` **không** xác định — cùng
prompt, cùng seed, hai lần chạy ra hai chuỗi khác nhau. Nghĩa là nếu judge gọi
model mỗi lần eval, thì chạy lại đúng một bundle trên đúng một golden set vẫn có
thể ra hai con số faithfulness khác nhau, và không có cách nào biết chênh lệch
đến từ hệ thống hay từ judge.

Cache địa chỉ theo nội dung biến điều đó thành: **lần chấm đầu tiên là phép đo,
mọi lần sau là phát lại**. Tiền tiết kiệm được chỉ là hệ quả.

Hệ quả thiết kế đi kèm: `frozen_cache=True` — mọi lượt tra trượt là **lỗi**, chứ
không phải một lời gọi mới. Đó là cách trả lời câu "tái lập lại con số trong báo
cáo của bạn đi" bằng một lệnh chứ không bằng một lời hứa: cùng cache + cùng run
⇒ cùng số, và nếu ai đó lặng lẽ sửa câu trả lời hay sửa rubric thì lệnh ấy đỏ.

## ⭐⭐ 2. Judge trả **nhãn**, không bao giờ trả điểm số

Hỏi một LLM "faithfulness bằng bao nhiêu, thang 0–1" thì nó sẽ trả `0.85`, và
con số đó có ba chữ số ý nghĩa mà không có gì đứng sau. Số học phải nằm trong mã
— nơi test được, nơi đổi quy ước thì mọi báo cáo cũ tính lại được.

Còn một lý do cứng hơn: `W5-04` phải tính Cohen's kappa giữa judge và người.
Kappa cần **nhãn rời rạc**. Nếu judge trả số thì phải chia khoảng, và cái khoảng
ấy sẽ được chọn *sau khi* đã nhìn dữ liệu — tức là chọn ngưỡng cho vừa kết quả
mình muốn.

## ⭐ 3. Đầu ra không đọc được KHÔNG phải là một phán quyết xấu

Cám dỗ là quy `UNPARSEABLE` thành "không được chống đỡ". Làm vậy là tính lỗi của
judge thành lỗi của hệ thống bị chấm — và nó luôn lệch về một phía (điểm thấp
hơn thực tế), nên trông giống một hệ thống tệ chứ không giống một phép đo hỏng.

Ở đây phán quyết không đọc được mang `label=None`, bị **loại** khỏi tử số lẫn
mẫu số, và được **đếm riêng**. Mọi hàm gộp phải công bố `n_unjudged`. Tỉ lệ ấy
cao thì kết luận đúng là "phép đo này không dùng được", không phải "điểm thấp".

## ⭐ 4. Quy tắc cấm preset áp cả cho bí danh của chính DeepSeek

Quy tắc cứng #1 của dự án cấm OpenRouter preset vì preset là con trỏ phía server.
Nhưng `DEEPSEEK_ALIASES` trong `rag_core.llm` ghi rõ rằng `deepseek-reasoner`
**cũng** là một con trỏ. Cấm cái này mà cho phép cái kia là áp quy tắc theo tên
nhà cung cấp chứ không theo lý do.

Nên `JudgeConfig` từ chối cả hai, trừ khi khai `allow_alias=True` một cách tường
minh — và kể cả lúc ấy, model **thực tế đã phục vụ** vẫn được ghi vào từng entry
cache và tổng hợp lại trong `JudgeStats.served_models`. Một lần chạy trộn hai
model là một dòng trong báo cáo, không phải một điều không ai biết.

`temperature` thì không có trong config: nó là hằng `0.0`. Một field mặc định 0
là một field ai đó đặt thành 0,7 được mà không có gì kêu.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_core.generation import PromptRegistry
from rag_core.llm import (
    DEEPSEEK_ALIASES,
    DEEPSEEK_PRICING,
    GLM_PRICING,
    MIN_REASONING,
    BudgetExceeded,
    ChatMessage,
    CostBudget,
    LLMProvider,
    ModelPricing,
)

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_PROMPT_DIR",
    "Judge",
    "JudgeCache",
    "JudgeConfig",
    "JudgeConfigError",
    "JudgeQuestion",
    "JudgeStats",
    "JudgeVerdict",
    "build_judge",
    "judge_registry",
]

logger = logging.getLogger(__name__)

JUDGE_PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
JUDGE_TEMPERATURE = 0.0
"""Hằng số, không phải mặc định. Xem điểm 4 ở docstring module."""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_PRESET = "@preset/"


class JudgeConfigError(ValueError):
    """Cấu hình judge vi phạm một quy tắc cứng — ném **trước** mọi lời gọi mạng."""


def judge_registry(root: Path | str = JUDGE_PROMPT_DIR) -> PromptRegistry:
    """Registry rubric của judge — cùng lớp với prompt serving của `W4-11`.

    Dùng lại chứ không viết lại, vì thứ cần ở đây đúng là thứ `W4-11` đã ép:
    sửa rubric mà quên tăng version thì loader từ chối nạp. Và version ấy nằm
    trong khoá cache dưới đây, nên **sửa rubric là mất cache** — đúng như phải
    thế: phán quyết cũ được sinh dưới một câu hỏi khác.
    """
    return PromptRegistry(root)


# --------------------------------------------------------------------- cấu hình


@dataclass(frozen=True)
class JudgeConfig:
    model: str = DEFAULT_JUDGE_MODEL
    base_url: str = DEEPSEEK_BASE_URL
    max_tokens: int = 512
    cap_usd: float = 1.0
    """Trần cho **một lần chạy**. `<= 0` là không trần, và phải khai tường minh."""
    cache_path: Path = Path(".cache/judge.sqlite3")
    frozen_cache: bool = False
    """Mọi lượt tra trượt là lỗi. Chế độ "tái lập lại đúng con số đã báo cáo"."""
    allow_alias: bool = False
    concurrency: int = 6
    seed: int = 20260904
    prompt_root: Path = JUDGE_PROMPT_DIR
    reasoning: bool = False
    """⚠️ Mặc định **tắt**, và đó là một quyết định do phép đo lật lại.

    Trực giác nói việc của judge đúng là suy luận, nên bản đầu để `True`. Probe
    `w5-03-judge-arms.json` (12 mẫu tự gán nhãn, k=3, cùng seed) nói khác:

    | nhánh | suy luận | không đọc được | ổn định | khớp nhãn tay | chi phí | trễ TB |
    |---|---|---|---|---|---|---|
    | A `deepseek-reasoner` | bật | 2/36 | 0,83 | **12/12** | $0,0230 | 1833 ms |
    | B `deepseek-v4-flash` | bật | 3/36 | 0,75 | **12/12** | $0,0115 | 1911 ms |
    | C `deepseek-v4-flash` | **tắt** | **0/36** | **1,00** | **12/12** | $0,0043 | 720 ms |

    Suy luận **không mua thêm một phán quyết đúng nào** trên tập này, mà đổi lại
    5,4× chi phí, 2,5× độ trễ và 5/72 lời gọi mất trắng. Và cả 5 lời gọi ấy có
    cùng một dấu vết: `completion_tokens == 512`, đúng bằng `max_tokens` — chuỗi
    suy luận ăn hết chỗ, `content` không bao giờ tới được phần JSON. Đúng cái bẫy
    `W4-06` đã trả tiền một lần để học.

    ⚠️ Giới hạn của kết luận: 12 mẫu ấy do tôi soạn và đều có đáp án dứt khoát.
    Nó chứng minh suy luận là thừa trên ca dễ; nó **không** nói gì về ca khó.
    `W5-04` chấm 50 mẫu lấy từ câu trả lời thật — đó là chỗ phải hỏi lại câu này.
    """
    json_mode: bool = True

    def __post_init__(self) -> None:
        if _PRESET in self.model:
            raise JudgeConfigError(
                f"Quy tắc cứng #1: không dùng OpenRouter preset trên đường eval "
                f"({self.model!r}). Ghim slug tường minh."
            )
        if not self.allow_alias and self.model in DEEPSEEK_ALIASES:
            raise JudgeConfigError(
                f"{self.model!r} là bí danh do server nắm — hiện trỏ tới "
                f"{DEEPSEEK_ALIASES[self.model]!r} và sẽ đổi khi nhà cung cấp ra bản mới. "
                "Đo bằng bí danh thì hai lần chạy cách nhau vài tháng không so được "
                "với nhau mà không có gì báo. Ghim slug thật, hoặc khai allow_alias=True "
                "và chấp nhận rằng báo cáo phải ghi kèm model thực tế đã phục vụ."
            )
        if self.max_tokens < 1:
            raise JudgeConfigError("max_tokens phải ≥ 1")
        if self.concurrency < 1:
            raise JudgeConfigError("concurrency phải ≥ 1")
        if self.family != "deepseek" and self.base_url == DEEPSEEK_BASE_URL:
            raise JudgeConfigError(
                f"{self.model!r} thuộc họ {self.family!r} nhưng base_url vẫn là endpoint "
                f"DeepSeek. Không chặn ở đây thì lỗi rơi xuống thành HTTP 404 giữa một "
                "lần chấm dở, sau khi đã tiêu tiền cho những câu trước nó."
            )

    @property
    def family(self) -> str:
        """Họ model, **suy ra từ slug** chứ không phải một field khai riêng.

        `W5-04` cần một judge khác họ để cross-check, và ba thứ đổi theo họ:
        bảng giá, tham số giảm suy luận (`MIN_REASONING`), và endpoint. Cám dỗ
        là thêm `family: str` vào config — nhưng thế là mở đường cho một cấu
        hình khai `family="glm"` với `model="deepseek-v4-flash"`, và khi đó
        `extra_body` gửi đi sai họ **mà không có lỗi nào**: DeepSeek nhận
        `reasoning_effort` rồi bỏ qua (đo ở `W3-04`). Phán quyết vẫn về, vẫn
        vào cache, chỉ là được sinh dưới một điều kiện khác lời khai.

        Suy ra từ slug thì trường hợp ấy không tồn tại được. Nó cũng là lý do
        `family` **không** cần vào khoá cache: `model` đã ở trong đó, và họ là
        hàm của model.
        """
        for prefix, name in (("deepseek-", "deepseek"), ("glm-", "glm")):
            if self.model.startswith(prefix):
                return name
        raise JudgeConfigError(
            f"chưa biết họ của {self.model!r}. Thêm bảng giá + mục MIN_REASONING "
            "trước khi chấm bằng model này — nếu không, chi phí sẽ báo $0 và tham "
            "số suy luận sẽ gửi sai họ mà không có gì kêu."
        )

    @property
    def pricing(self) -> ModelPricing:
        table = {"deepseek": DEEPSEEK_PRICING, "glm": GLM_PRICING}[self.family]
        return table.get(self.model, ModelPricing())

    @property
    def reasoning_body(self) -> dict[str, Any] | None:
        """`None` = để provider tự quyết (bật hết cỡ). Ngược lại: **thấp nhất
        provider cho phép** — với GLM đó không phải là tắt.

        Bất đối xứng này là thật và không khắc phục được: `glm-5.3-flash` trả
        HTTP 400 khi bị yêu cầu tắt suy luận, chỉ nhận `low/high/max`. Nên hai
        nhánh judge của `W5-04` **không** so được ở điều kiện suy luận giống
        nhau, và mọi bất đồng giữa chúng đều mang lẫn một phần khác biệt ấy.
        Ghi ra đây để báo cáo không kết luận quá tay.
        """
        return None if self.reasoning else MIN_REASONING[self.family]


def build_judge(config: JudgeConfig) -> Judge:
    """Dựng `Judge` với provider **suy ra từ họ model**, không phải từ tham số.

    ## Vì sao hàm này tồn tại

    `W5-04` (`calibration.py`) đã có đúng đoạn rẽ nhánh này; `W5-11` cần nó lần
    thứ hai ở `generation_metrics.py`, nơi provider trước giờ **hardcode
    DeepSeek**. Chép lần thứ hai thì hai bản sẽ lệch, và cách chúng lệch là im
    lặng: một bên gửi `reasoning_effort` sang DeepSeek — model **nhận rồi bỏ
    qua** (đo ở `W3-04`) — nên phán quyết vẫn về, vẫn vào cache, chỉ là được
    sinh dưới một điều kiện khác lời khai.

    Rẽ theo `config.family`, thứ tự nó tự suy ra từ slug, nên không có đường nào
    để một lời gọi khai một họ và dùng endpoint của họ khác.
    """
    from rag_core.llm import build_deepseek_provider, build_glm_provider
    from rag_core.settings import get_settings

    settings = get_settings()
    if config.family == "glm":
        key = settings.glm_api_key
        provider = build_glm_provider(
            config.model, api_key=key.get_secret_value() if key else "", base_url=config.base_url
        )
    else:
        key = settings.deepseek_api_key
        provider = build_deepseek_provider(
            config.model, api_key=key.get_secret_value() if key else "", base_url=config.base_url
        )
    return Judge(config, provider)


@dataclass(frozen=True)
class JudgeQuestion:
    """Một câu hỏi dành cho judge.

    `ref` cố ý **không** vào khoá cache: cùng một cặp (mệnh đề, ngữ cảnh) xuất
    hiện ở hai truy vấn khác nhau là cùng một phép chấm, và trả tiền hai lần cho
    nó vừa tốn vừa mở đường cho hai phán quyết khác nhau về cùng một thứ.
    """

    prompt_id: str
    labels: tuple[str, ...]
    variables: Mapping[str, str]
    ref: str = ""

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("phải khai tập nhãn — judge trả nhãn, không trả điểm")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"tập nhãn trùng: {self.labels}")


@dataclass(frozen=True)
class JudgeVerdict:
    ref: str
    label: str | None
    """`None` nghĩa là **không đọc được**, không phải nhãn xấu. Xem điểm 3."""
    reason: str
    served_model: str
    cached: bool
    cost_usd: float
    error: str = ""
    raw: str = ""

    @property
    def judged(self) -> bool:
        return self.label is not None


@dataclass
class JudgeStats:
    hits: int = 0
    misses: int = 0
    repairs: int = 0
    unparseable: int = 0
    budget_stopped: int = 0
    truncated: int = 0
    served_models: dict[str, int] = field(default_factory=dict)
    spent_usd: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "repairs": self.repairs,
            "unparseable": self.unparseable,
            "budget_stopped": self.budget_stopped,
            "served_models": dict(sorted(self.served_models.items())),
            "spent_usd": round(self.spent_usd, 6),
        }


# ------------------------------------------------------------------------ cache

_CACHE_TABLE = "judge_cache_v1"
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (
    key           TEXT PRIMARY KEY,
    label         TEXT,
    reason        TEXT NOT NULL,
    served_model  TEXT NOT NULL,
    raw           TEXT NOT NULL,
    cost_usd      REAL NOT NULL,
    created_at    REAL NOT NULL
);
"""


class JudgeCache:
    """Cache địa chỉ theo nội dung cho phán quyết của judge.

    ⚠️ Phán quyết **không đọc được không được ghi vào đây**. Nó là lỗi tạm thời
    của một lời gọi, không phải một kết quả; ghi lại thì lần chạy sau sẽ "tái
    lập" đúng cái hỏng ấy mà không tốn một lời gọi nào để phát hiện.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.DatabaseError:
            logger.warning("Cache judge %s hỏng, dựng lại từ đầu", self.path)
            self.path.unlink(missing_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT label, reason, served_model, raw, cost_usd FROM {_CACHE_TABLE} "
                "WHERE key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "label": row[0],
            "reason": row[1],
            "served_model": row[2],
            "raw": row[3],
            "cost_usd": row[4],
        }

    def put(
        self,
        key: str,
        *,
        label: str,
        reason: str,
        served_model: str,
        raw: str,
        cost_usd: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {_CACHE_TABLE} "
                "(key, label, reason, served_model, raw, cost_usd, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (key, label, reason, served_model, raw, cost_usd, time.time()),
            )

    def __len__(self) -> int:
        with self._connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {_CACHE_TABLE}").fetchone()[0])

    def digest(self) -> str:
        """Vân tay của **toàn bộ** phán quyết đang có, để báo cáo ghi lại một số.

        Hai lần chạy cùng digest ⇒ cùng tập phán quyết ⇒ cùng con số. Con số này
        đi vào report; ai muốn kiểm thì chạy lại với `frozen_cache=True`.
        """
        hasher = hashlib.sha256()
        with self._connect() as conn:
            for key, label in conn.execute(f"SELECT key, label FROM {_CACHE_TABLE} ORDER BY key"):
                hasher.update(f"{key}:{label}\n".encode())
        return hasher.hexdigest()


# ------------------------------------------------------------------------ judge


def _canonical_key(
    *,
    prompt_spec: str,
    prompt_sha256: str,
    model: str,
    max_tokens: int,
    seed: int,
    reasoning: bool,
    json_mode: bool,
    labels: Sequence[str],
    variables: Mapping[str, str],
) -> str:
    """Khoá cache. Mọi thứ **thay đổi được câu trả lời** phải nằm trong đây.

    Có cả `prompt_sha256` bên cạnh `prompt_spec`: version là lời khai của con
    người, hash là sự thật. `W4-11` đã ép hai cái phải khớp lúc nạp, nên đưa cả
    hai vào là dư thừa rẻ tiền — và nó chặn đúng trường hợp registry bị thay
    bằng một registry khác có cùng số version.
    """
    payload = {
        "prompt": prompt_spec,
        "prompt_sha256": prompt_sha256,
        "model": model,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": max_tokens,
        "seed": seed,
        "reasoning": reasoning,
        "json_mode": json_mode,
        "labels": list(labels),
        "variables": dict(variables),
    }
    # `sort_keys=True` chuẩn hoá thứ tự khoá **đệ quy**, kể cả `variables` lồng
    # bên trong — nên ở trên không cần sắp lại tay. Bản đầu có làm, và phép tiêm
    # J2 (bỏ chỗ sắp ấy) sống sót đúng vì nó là mã chết. Giữ lại một lớp bảo vệ
    # không có tác dụng còn tệ hơn không có: người đọc sau sẽ tưởng nó đang giữ
    # một bất biến nào đó.
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _render(template: str, variables: Mapping[str, str]) -> str:
    """Thay `{{ten}}`. Dùng `str.replace` chứ không `.format` vì rubric chứa
    chính dấu ngoặc nhọn của mẫu JSON mà nó yêu cầu judge trả về."""
    out = template
    for name, value in variables.items():
        out = out.replace("{{" + name + "}}", value)
    leftover = re.findall(r"\{\{(\w+)\}\}", out)
    if leftover:
        raise KeyError(f"rubric còn biến chưa điền: {sorted(set(leftover))}")
    return out


def _parse_verdict(text: str, labels: Sequence[str]) -> tuple[str | None, str]:
    """Trả `(label, reason)`; `label=None` nghĩa là không đọc được.

    Nhãn so khớp **không phân biệt hoa thường và bỏ khoảng trắng hai đầu** —
    model trả `"supported"` thay vì `"SUPPORTED"` là chuyện thường và không phải
    một phán quyết hỏng. Nhưng một nhãn *ngoài* tập khai báo thì có: nó nghĩa là
    judge đã trả lời một câu hỏi khác câu được hỏi.
    """
    cleaned = _FENCE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Đôi khi model bọc JSON trong một câu dẫn. Vớt đối tượng ngoài cùng.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None, "không tìm thấy JSON trong phản hồi"
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"JSON hỏng: {exc}"
    if not isinstance(data, dict):
        return None, "phản hồi không phải đối tượng JSON"
    raw_label = data.get("verdict")
    if not isinstance(raw_label, str):
        return None, "thiếu trường `verdict`"
    upper = {label.upper(): label for label in labels}
    label = upper.get(raw_label.strip().upper())
    if label is None:
        return None, f"nhãn ngoài tập khai báo: {raw_label!r}"
    reason = data.get("reason")
    return label, reason.strip() if isinstance(reason, str) else ""


_REPAIR = (
    "Phản hồi vừa rồi không đọc được bằng máy. Trả lại DUY NHẤT một đối tượng JSON "
    'dạng {"verdict": "...", "reason": "..."} — không giải thích, không khối mã, '
    "không chữ nào ngoài JSON."
)


class Judge:
    """Chấm nhãn qua LLM, có cache, có trần chi phí, an toàn nhiều luồng.

    `provider` được tiêm vào chứ không dựng bên trong: mọi test của lớp này chạy
    với một provider giả, không key, không mạng — cùng lý lẽ với
    `evaluate_run` của `retrieval_eval`.
    """

    def __init__(
        self,
        config: JudgeConfig,
        provider: LLMProvider | None = None,
        *,
        registry: PromptRegistry | None = None,
        cache: JudgeCache | None = None,
        budget: CostBudget | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.registry = registry or judge_registry(config.prompt_root)
        self.cache = cache if cache is not None else JudgeCache(config.cache_path)
        self.budget = budget or CostBudget(config.cap_usd, name="judge")
        self.stats = JudgeStats()
        self._lock = threading.Lock()
        self._stopped = False

    # ------------------------------------------------------------- nội bộ

    def _record(self, *, served_model: str, cost: float) -> None:
        with self._lock:
            self.stats.served_models[served_model] = (
                self.stats.served_models.get(served_model, 0) + 1
            )
            self.stats.spent_usd += cost

    def _estimate_usd(self, prompt_text: str) -> float:
        """Ước lượng **cao hơn thực tế** một cách có chủ đích.

        `len//3` cho tiếng Việt là ước lượng thừa (thực tế gần `len//3,5`), và
        `max_tokens` coi như dùng hết. Trần chi phí sai về phía chặn sớm thì mất
        vài lời gọi cuối; sai về phía nới thì mất tiền — và đó là thứ trần này
        tồn tại để ngăn.
        """
        return self.config.pricing.cost(len(prompt_text) // 3, self.config.max_tokens)

    def _call(self, messages: list[ChatMessage]) -> tuple[str, str, float, str]:
        if self.provider is None:  # pragma: no cover - lỗi cấu hình rõ ràng
            raise RuntimeError("Judge chưa có provider — chỉ tra cache được")
        response = self.provider.complete(
            messages,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=self.config.max_tokens,
            json_mode=self.config.json_mode,
            seed=self.config.seed,
            extra_body=self.config.reasoning_body,
        )
        return (
            response.text,
            response.model,
            response.usage.cost_usd,
            response.finish_reason or "",
        )

    # ------------------------------------------------------------ công khai

    def ask(self, question: JudgeQuestion) -> JudgeVerdict:
        prompt = self.registry.get(question.prompt_id)
        rendered = _render(prompt.text, question.variables)
        key = _canonical_key(
            prompt_spec=prompt.spec,
            prompt_sha256=prompt.sha256,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed,
            reasoning=self.config.reasoning,
            json_mode=self.config.json_mode,
            labels=question.labels,
            variables=question.variables,
        )

        hit = self.cache.get(key)
        if hit is not None:
            with self._lock:
                self.stats.hits += 1
            return JudgeVerdict(
                ref=question.ref,
                label=hit["label"],
                reason=hit["reason"],
                served_model=hit["served_model"],
                cached=True,
                cost_usd=0.0,
                raw=hit["raw"],
            )

        with self._lock:
            self.stats.misses += 1
        if self.config.frozen_cache:
            raise JudgeConfigError(
                f"frozen_cache: trượt cache ở {question.ref or key[:12]} — "
                "lần chạy này không tái lập được con số đã báo cáo. Hoặc dữ liệu vào "
                "đã đổi, hoặc rubric đã đổi, hoặc cache không phải cache của lần chạy ấy."
            )
        if self._stopped:
            return JudgeVerdict(
                ref=question.ref,
                label=None,
                reason="",
                served_model="",
                cached=False,
                cost_usd=0.0,
                error="budget",
            )

        messages = [ChatMessage(role="user", content=rendered)]
        try:
            self.budget.reserve(self._estimate_usd(rendered))
        except BudgetExceeded as exc:
            with self._lock:
                self._stopped = True
                self.stats.budget_stopped += 1
            logger.error("Judge dừng vì chạm trần chi phí: %s", exc)
            return JudgeVerdict(
                ref=question.ref,
                label=None,
                reason="",
                served_model="",
                cached=False,
                cost_usd=0.0,
                error="budget",
            )

        text, served, cost, finish = self._call(messages)
        self.budget.charge(cost)
        self._record(served_model=served, cost=cost)
        label, reason = _parse_verdict(text, question.labels)
        repair_blocked = False

        if label is None and finish == "length":
            # ⭐ Không tiêu một lời gọi sửa cho một nguyên nhân đã biết chắc.
            #
            # Prompt sửa nói "hãy trả JSON" — nhưng model **đang** trả JSON, nó
            # chỉ hết chỗ trước khi tới được đó. Gọi lại với đúng `max_tokens`
            # ấy sẽ cụt ở đúng chỗ ấy, và trả tiền lần thứ hai cho cùng một cái
            # hỏng. Probe `w5-03-judge-arms.json`: 5/5 phán quyết không đọc được
            # đều có `completion_tokens == max_tokens`, không sót cái nào.
            with self._lock:
                self.stats.truncated += 1
            logger.error(
                "Judge cụt ở max_tokens=%d cho %s — nới max_tokens hoặc đặt "
                "reasoning=False, đừng sửa rubric",
                self.config.max_tokens,
                question.ref or "?",
            )
        elif label is None:
            # Một lần sửa, không phải một vòng lặp: nếu nhắc một lần mà vẫn không
            # trả JSON thì vấn đề nằm ở rubric hoặc ở model, và nhắc thêm chỉ tiêu
            # thêm tiền cho cùng một câu trả lời hỏng.
            with self._lock:
                self.stats.repairs += 1
            messages = [*messages, ChatMessage(role="user", content=_REPAIR)]
            try:
                self.budget.reserve(self._estimate_usd(rendered))
            except BudgetExceeded:
                with self._lock:
                    self._stopped = True
                    self.stats.budget_stopped += 1
                repair_blocked = True
            else:
                text2, served2, cost2, finish = self._call(messages)
                self.budget.charge(cost2)
                self._record(served_model=served2, cost=cost2)
                cost += cost2
                label, reason = _parse_verdict(text2, question.labels)
                text, served = text2, served2

        if label is None:
            with self._lock:
                self.stats.unparseable += 1
            logger.warning("Judge trả phán quyết không đọc được cho %s: %s", question.ref, reason)
            # Phân biệt "judge nói năng lộn xộn" với "hết tiền nên không sửa được":
            # gộp hai cái vào một mã lỗi là mất đúng thông tin cần để biết nên
            # sửa rubric hay nên nới trần.
            if finish == "length":
                code = "truncated"
            elif repair_blocked:
                code = "unparseable+budget"
            else:
                code = "unparseable"
            # KHÔNG ghi cache. Xem docstring `JudgeCache`.
            return JudgeVerdict(
                ref=question.ref,
                label=None,
                reason=reason,
                served_model=served,
                cached=False,
                cost_usd=cost,
                error=code,
                raw=text,
            )

        self.cache.put(
            key,
            label=label,
            reason=reason,
            served_model=served,
            raw=text,
            cost_usd=cost,
        )
        return JudgeVerdict(
            ref=question.ref,
            label=label,
            reason=reason,
            served_model=served,
            cached=False,
            cost_usd=cost,
            raw=text,
        )

    def ask_many(self, questions: Sequence[JudgeQuestion]) -> list[JudgeVerdict]:
        """Chấm song song, **giữ nguyên thứ tự đầu vào**.

        Thứ tự quan trọng hơn vẻ ngoài: nơi gọi ghép phán quyết với mệnh đề theo
        chỉ số, và một danh sách trả về theo thứ tự hoàn thành sẽ gán nhãn của
        câu này cho câu khác — sai lặng lẽ, và sai khác nhau mỗi lần chạy.
        """
        if not questions:
            return []
        if self.config.concurrency == 1:
            return [self.ask(q) for q in questions]
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            return list(pool.map(self.ask, questions))
