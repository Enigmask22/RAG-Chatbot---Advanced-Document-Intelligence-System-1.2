"""Tầng LLM — dùng chung cho pipeline (sinh dữ liệu, judge) và serving (generator).

Giá tính theo bảng công bố của DeepSeek. Ghi thành hằng số ở đây để mọi báo cáo
chi phí dùng chung một nguồn, và để lúc giá đổi thì chỉ sửa một chỗ — nhưng giá
đã dùng vẫn được log kèm từng lần chạy, nên báo cáo cũ không bị viết lại.
"""

import logging
from typing import Any

from .base import (
    ChatMessage,
    LLMChunk,
    LLMError,
    LLMProvider,
    LLMResponse,
    ModelPricing,
    StreamingLLM,
)
from .budget import BudgetExceeded, CostBudget
from .openai_compat import OpenAICompatProvider, PermanentLLMError
from .router import AllRoutesFailed, CircuitBreaker, CircuitState, DailyBudget, LLMRouter, Route
from .tokenizer import HFTokenCounter

__all__ = [
    "DEEPSEEK_ALIASES",
    "DEEPSEEK_PRICING",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_GLM_MODEL",
    "GLM_BASE_URL",
    "GLM_PRICING",
    "MIN_REASONING",
    "AllRoutesFailed",
    "BudgetExceeded",
    "ChatMessage",
    "CircuitBreaker",
    "CircuitState",
    "CostBudget",
    "DailyBudget",
    "HFTokenCounter",
    "LLMChunk",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "ModelPricing",
    "OpenAICompatProvider",
    "PermanentLLMError",
    "Route",
    "StreamingLLM",
    "build_deepseek_provider",
    "build_glm_provider",
]

DEEPSEEK_PRICING: dict[str, ModelPricing] = {
    # Giá công bố cho `deepseek-chat`. `deepseek-v4-flash` dùng chung con số này
    # vì hiện `deepseek-chat` chính là bí danh trỏ tới nó — nếu DeepSeek tách giá
    # hai model thì phải sửa ở đây. Giá đã dùng luôn được ghi kèm từng lần chạy
    # nên báo cáo cũ không bị viết lại khi bảng này đổi.
    "deepseek-chat": ModelPricing(
        input_per_1m_usd=0.27,
        output_per_1m_usd=1.10,
        cached_input_per_1m_usd=0.07,
    ),
    "deepseek-v4-flash": ModelPricing(
        input_per_1m_usd=0.27,
        output_per_1m_usd=1.10,
        cached_input_per_1m_usd=0.07,
    ),
    "deepseek-reasoner": ModelPricing(
        input_per_1m_usd=0.55,
        output_per_1m_usd=2.19,
        cached_input_per_1m_usd=0.14,
    ),
}

#: Bí danh của DeepSeek: `deepseek-chat` và `deepseek-reasoner` đều được phục vụ
#: bởi model bên phải, và cái đó sẽ đổi khi DeepSeek ra phiên bản mới.
#:
#: Đây đúng là vấn đề mà quy tắc cứng #1 nói về OpenRouter preset, chỉ kín đáo
#: hơn: tên trông như một model cụ thể nhưng thật ra là con trỏ do server nắm.
#: Đo bằng bí danh thì con số tháng này không so được với tháng sau mà không có
#: gì báo. Mặc định của dự án vì vậy là slug thật.
DEEPSEEK_ALIASES: dict[str, str] = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
}

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

GLM_PRICING: dict[str, ModelPricing] = {
    # Giá công bố của Z.ai cho GLM-5.3-Flash. Rẻ hơn `deepseek-chat` 1,8× ở
    # input, 2,2× ở output, 2,3× ở phần cache trúng.
    #
    # ⚠️ Có một mức khuyến mại $0,075/1M input chạy tới 2026-09-09. **Không** ghi
    # nó vào đây: bảng này là giá niêm yết, và một báo cáo chi phí tính theo giá
    # khuyến mại sẽ đọc sai ngay khi khuyến mại hết mà không ai sửa lại con số.
    "glm-5.3-flash": ModelPricing(
        input_per_1m_usd=0.15,
        output_per_1m_usd=0.50,
        cached_input_per_1m_usd=0.03,
    ),
}

DEFAULT_GLM_MODEL = "glm-5.3-flash"

GLM_BASE_URL = "https://api.z.ai/api/paas/v4"
"""Endpoint quốc tế (Z.ai). Bản đại lục là `https://open.bigmodel.cn/api/paas/v4`.

Đo 2026-09-03: `api.z.ai` trả 200 với slug **`glm-5.3-flash`** (chữ thường).
"""


# ---------------------------------------------------------------- suy luận

MIN_REASONING: dict[str, dict[str, Any]] = {
    # `deepseek-v4-flash`, đo 2026-09-03, cùng prompt, `max_tokens=512`:
    #   khong dat                    -> reasoning 275, completion 328
    #   thinking={"type":"disabled"} -> reasoning   0, completion 138
    #   reasoning_effort="none"      -> reasoning   0, completion  87
    #   chat_template_kwargs=...     -> reasoning 159, completion 219  (NHAN roi BO QUA)
    "deepseek": {"thinking": {"type": "disabled"}},
    # `glm-5.3-flash`, cung prompt, cung ngay. Model nay **khong tat duoc** suy
    # luan -- API tra HTTP 400 ma`1210`: "This model always engages in thinking
    # and cannot be disabled; please use low, high, or max". Ba muc hop le, do
    # tren cung mot prompt:
    #   khong dat            -> reasoning 165, completion 243, content 401 ky tu
    #   reasoning_effort=low -> reasoning   0, completion  70, content 383
    #   reasoning_effort=high-> reasoning  40, completion 118, content 393
    #   reasoning_effort=max -> reasoning 180, completion 255, content 411
    # `low` cho `reasoning_content` **rong that** (0 ky tu), khong phai field bi
    # giau di -- da doc response tho de kiem. Do dai content khong doi dang ke,
    # nen 3,5x output do la tiet kiem sach.
    "glm": {"reasoning_effort": "low"},
    # vLLM/Qwen3: chat template cua Qwen3 doc `enable_thinking`. Day moi la cho
    # tham so ay CO tac dung -- DeepSeek nhan no roi bo qua.
    "vllm": {"chat_template_kwargs": {"enable_thinking": False}},
}
"""Tham số **giảm suy luận tới mức thấp nhất provider cho phép**, đã đo từng cái.

Tên là `MIN_REASONING` chứ không phải `NO_THINKING` vì GLM-5.3-Flash **không tắt
được**: nó chỉ nhận `low`/`high`/`max`. Gọi tên sai ở đây sẽ dẫn tới đọc sai một
báo cáo chi phí về sau.

⚠️ Hai trong bốn dòng trên là **tham số được nhận, không lỗi, và không có tác
dụng** nếu đặt nhầm nhà: `chat_template_kwargs` với DeepSeek (vẫn 159 token suy
luận), và `thinking={"type":...}` với GLM (HTTP 400). Không đo thì tưởng đã tắt.

⭐ Sống ở `rag_core` chứ không ở `pipeline`: `W3-04` đo nó cho job contextualize,
rồi `W4-06` cần **đúng** nó ở đường serving — và `serving` không được import
`pipeline` (`tests/unit/test_architecture_boundaries.py`). Chép sang là tạo ra
hai bảng số đo cho cùng một thứ, và cái thứ hai sẽ là cái không ai cập nhật khi
provider đổi hành vi.

⚠️ Lý do `W4-06` cần nó **không phải** để tiết kiệm tiền. Lần chạy thật đầu tiên
của `POST /chat` với `max_tokens=1024` trả về **0 ký tự**: 1024/1024 token
completion đi vào chuỗi suy luận, `finish_reason` = `empty`, hoá đơn $0,0015 cho
một câu trả lời không tồn tại. Ở job offline đây là lãng phí; ở đường request nó
là **endpoint hỏng**."""


def build_glm_provider(
    model: str = DEFAULT_GLM_MODEL,
    *,
    api_key: str,
    base_url: str = GLM_BASE_URL,
    **kwargs: object,
) -> OpenAICompatProvider:
    """GLM (Z.ai) nói giao thức OpenAI, kể cả phần `usage` — nên chỉ cần bảng giá.

    Đo được là nó trả **đúng** hai field mà `_parse` đã đọc sẵn từ `W1-10`:
    `prompt_tokens_details.cached_tokens` và
    `completion_tokens_details.reasoning_tokens`. Không phải provider nào cũng
    vậy (DeepSeek dùng `prompt_cache_hit_tokens`), nên đây là may chứ không phải
    mặc định — và nó có nghĩa là báo cáo chi phí của GLM chính xác ngay từ lượt
    chạy đầu, không cần đường parse riêng.
    """
    return OpenAICompatProvider(
        model,
        api_key=api_key,
        base_url=base_url,
        pricing=GLM_PRICING.get(model, ModelPricing()),
        **kwargs,  # type: ignore[arg-type]
    )


def build_deepseek_provider(
    model: str = DEFAULT_DEEPSEEK_MODEL,
    *,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    **kwargs: object,
) -> OpenAICompatProvider:
    """DeepSeek nói giao thức OpenAI, nên chỉ cần gắn đúng bảng giá."""
    if model in DEEPSEEK_ALIASES:
        logging.getLogger(__name__).warning(
            "%r là bí danh, hiện trỏ tới %r và sẽ đổi khi DeepSeek ra bản mới. "
            "Trên đường eval hãy ghim slug thật để hai lần đo còn so được với nhau.",
            model,
            DEEPSEEK_ALIASES[model],
        )
    return OpenAICompatProvider(
        model,
        api_key=api_key,
        base_url=base_url,
        pricing=DEEPSEEK_PRICING.get(model, ModelPricing()),
        **kwargs,  # type: ignore[arg-type]
    )
