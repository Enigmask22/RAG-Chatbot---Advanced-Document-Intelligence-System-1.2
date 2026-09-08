"""Demo công khai của RAG platform, chạy trên HF Spaces + ZeroGPU. `W6-02`.

## Đây KHÔNG phải một bản viết lại của hệ thống

`requirements.txt` cài `rag-platform` từ `git+…@<sha>`, và mọi việc thật ở file
này do `serving.core.chat.ChatService` làm — cùng lớp mà `POST /chat` của
production gọi. Truy hồi là `rag_core` nguyên vẹn, chạy trên `qdrant-client`
local mode (đo được: xếp hạng **trùng khớp hoàn toàn** với Qdrant server trên 30
truy vấn golden, `probes/w602-local-parity.json`). Bundle là `manifest.json`
đúng bản đã eval, và `QdrantRuntimeBuilder._check_identity` (`TD-38`) vẫn chạy
— nên nếu môi trường Space dựng ra một retriever khác với chuỗi đã ký, Space
**không lên được** thay vì lặng lẽ demo một hệ thống khác.

Ba thứ Space cố ý **không** có, và cả ba đều nói ra trên giao diện:

* **Postgres** (`sessions=None`) ⇒ không lưu lịch sử hội thoại giữa các lần tải
  trang. Đây cũng là lựa chọn về quyền riêng tư: một demo công khai không có lý
  do gì để giữ câu hỏi của người lạ.
* **Redis** ⇒ không có semantic cache. Hệ quả **tốn tiền**: hai người hỏi cùng
  một câu trả tiền hai lần. Đổi lại, mọi câu trả lời trên demo đều là một lượt
  sinh thật, không phải bản phát lại — và với một demo thì đó là thứ đáng xem.
* **Xác thực** ⇒ `guard.py` là hàng rào duy nhất. Đọc docstring của nó.

## ⚠️ Space không chạy đúng bộ phụ thuộc đã eval

Repo chạy Python **3.13.11** + torch **2.13**; ZeroGPU chỉ nhận Python 3.12.12
hoặc 3.10.13. `retriever_name` mà `TD-38` so **không** mã hoá phiên bản torch,
nên phép kiểm ấy xanh trong khi môi trường vẫn lệch. Điều đó có thể dịch vài
chữ số cuối của điểm cross-encoder fp16. Ghi ra ở đây vì không có phép kiểm nào
bắt được nó — xem `TD-87`.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr

from guard import Verdict, client_key_of, guard_from_env
from rag_core.bundle.store import read_pointer
from rag_core.settings import get_settings
from serving.api.app import build_llm, build_understanding
from serving.core.auth import Principal
from serving.core.chat import ChatService
from serving.core.registry import BundleRegistry
from serving.core.runtime import QdrantRuntimeBuilder
from serving.core.semantic_cache import embedder_of
from zerogpu import ZeroGpuRuntimeBuilder, materialise_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("space")

HERE = Path(__file__).parent
#: Trên Space cả hai nằm cạnh `app.py`; biến môi trường tồn tại để chạy thử
#: được trên máy dev mà **không** phải nhân đôi 239 MB index vào `space/`.
INDEX_DIR = Path(os.environ.get("SPACE_INDEX_DIR") or HERE / "index")
BUNDLE_ROOT = Path(os.environ.get("SPACE_BUNDLE_ROOT") or HERE / "bundles")

#: Người gọi. Corpus được index dưới `tenant_id="public"` (`TD-40`), và mọi
#: truy hồi đi qua `tenant_filter()` như trên production — không có đường tắt
#: "bỏ lọc tenant cho demo", vì đó đúng là hàng rào `W2-06` dựng lên.
DEMO_PRINCIPAL = Principal(tenant_id="public", key_id="space-demo", scopes=frozenset())

GUARD = guard_from_env()

EXAMPLES = [
    "Tăng trưởng GDP của Việt Nam được đóng góp bởi những yếu tố nào?",
    "What are the main climate risks identified for Vietnam's transport sector?",
    "Tỷ lệ nghèo đa chiều ở Việt Nam thay đổi thế nào trong giai đoạn gần đây?",
    "How does the World Bank assess Vietnam's business regulatory environment?",
]


# --------------------------------------------------------------------------
# Khởi động — chạy MỘT lần ở module scope. Xem `zerogpu.materialise_weights`.
# --------------------------------------------------------------------------


def _build() -> tuple[ChatService, dict[str, Any]]:
    from qdrant_client import QdrantClient

    settings = get_settings()
    started = time.perf_counter()

    client = QdrantClient(path=str(INDEX_DIR))
    opened = time.perf_counter()

    registry = BundleRegistry(
        root=BUNDLE_ROOT,
        # ⚠️⚠️ **Phải** tắt. `TD-72` làm nóng bằng một lượt truy hồi THẬT, tức
        # compute CUDA — và ở đây nó chạy ngoài `@spaces.GPU`, nơi không có GPU
        # thật. Nửa còn lại của cùng ý tưởng (`materialise_weights`) là nửa mà
        # nền tảng này cho phép. Xem docstring `zerogpu.py`.
        warmup=False,
        build_runtime=ZeroGpuRuntimeBuilder(
            QdrantRuntimeBuilder(
                url="local://khong-dung",
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                client=client,
                # ⚠️ **Không** bật drift. `TD-57` đã ghi đúng câu này: bật nó là
                # đúng kiểu nói dối mà `TD-38` sinh ra để chặn. Trên ZeroGPU
                # `torch.cuda.is_available()` trả True ngay ở module scope, nên
                # runtime dựng ra `…@cuda:L512:float16` — khớp bundle thật.
                allow_runtime_drift=False,
            )
        ),
    )
    # ⚠️ `read_pointer`, **không** `(root / "CURRENT").read_text()`. `W5-10` dựng
    # con trỏ này để xoá một bản sao thứ hai của cùng một sự thật (`AU-12`), và
    # tự đọc file ở đây là dựng lại đúng bản sao ấy — kèm toàn bộ luật nó mang
    # (thiếu file, chuỗi rỗng, xuống dòng thừa) viết lại bằng trí nhớ.
    version = read_pointer(BUNDLE_ROOT)
    if not version:
        raise SystemExit(f"{BUNDLE_ROOT}/CURRENT không trỏ vào bundle nào")
    registry.activate(version)
    activated = time.perf_counter()

    snapshot = registry.active
    packed = materialise_weights(embedder_of(snapshot.retriever), snapshot.reranker)
    ready = time.perf_counter()

    llm = build_llm(settings)
    service = ChatService(
        registry=registry,
        sessions=None,
        llm=llm,
        top_k=settings.chat_top_k,
        max_tokens=settings.chat_max_tokens,
        understanding=build_understanding(settings, llm),
        cache=None,
        sink=None,
    )
    boot = {
        "bundle": version,
        "retriever": snapshot.retriever.name,
        "mo_index_s": round(opened - started, 2),
        "kich_hoat_s": round(activated - opened, 2),
        "nap_trong_so_s": round(ready - activated, 2),
        "tong_khoi_dong_s": round(ready - started, 2),
        "trong_so_da_nap": packed,
        "co_khoa_sinh": llm is not None,
    }
    logger.info("khoi dong xong: %s", boot)
    return service, boot


SERVICE, BOOT = _build()


# --------------------------------------------------------------------------
# Trình bày
# --------------------------------------------------------------------------


def _sources_markdown(sources: list[dict[str, Any]], citations: dict[str, Any] | None) -> str:
    """Bảng nguồn. Nội dung chunk là dữ liệu corpus **không tin được**.

    ⚠️ `W6-01` đã ghi luật này cho trang HTML: không bao giờ `innerHTML`. Ở
    Gradio thì đường tương đương là **không nhúng chữ của chunk vào Markdown
    thô** — một chunk chứa `<img onerror=…>` hay `[x](javascript:…)` sẽ được
    Markdown dựng thành đúng thứ đó. Nên nội dung đi vào khối trích dẫn
    ``` ``` ``` (code fence), thứ Markdown không diễn giải bên trong. Cùng lý
    lẽ, cùng kết luận, cú pháp khác.
    """
    if not sources:
        return "_Lượt này không truy hồi tài liệu nào._"
    # ⭐ Ghép theo `chunk_id`, **không** theo số `[n]` — và `Citation` cố ý
    # không mang `n`. Cả điểm của `W4-09` là con số model viết ra có thể chỉ
    # sai chunk; `verify_citations` đối chiếu quote với **đúng chunk mà n trỏ
    # vào** rồi trả về `chunk_id` của kết quả. Ghép lại bằng `n` là quay về
    # tin con số vừa mới được đem đi kiểm.
    verified_chunks = set()
    if citations:
        verified_chunks = {
            c["chunk_id"] for c in citations.get("citations", []) if c.get("verified") is True
        }
    lines = []
    for src in sources:
        n = src["n"]
        # Không dùng emoji: nhãn xác minh là chữ, phần trang trí để CSS lo.
        badge = " · **trích dẫn đã xác minh**" if src.get("chunk_id") in verified_chunks else ""
        flags = src.get("flags") or []
        warn = f" · **cờ tiêm: {', '.join(flags)}**" if flags else ""
        title = src.get("title") or src.get("doc_id")
        url = src.get("source_url")
        head = f"**[{n}]** {title}" if not url else f"**[{n}]** [{title}]({url})"
        body = (src.get("content") or "").strip()
        if len(body) > 700:
            body = body[:700] + " …"
        lines.append(f"{head} — điểm {src['score']:.4f}{badge}{warn}\n\n```text\n{body}\n```")
    return "\n\n".join(lines)


def _stats_markdown(done: dict[str, Any], meta: dict[str, Any]) -> str:
    usage = done.get("usage") or {}
    bits = [
        f"bundle `{meta.get('bundle_version', '?')}`",
        f"model `{done.get('model', '?')}`",
        f"prepare {done.get('prepare_ms', '?')} ms",
        f"TTFB {done.get('ttfb_ms', '?')} ms",
        f"tổng {done.get('total_ms', '?')} ms",
    ]
    if usage.get("total_tokens"):
        bits.append(f"{usage['total_tokens']} token")
    if usage.get("cost_usd") is not None:
        bits.append(f"${usage['cost_usd']:.6f}")
    if done.get("language_mismatch"):
        bits.append("trả lời **sai ngôn ngữ** so với câu hỏi")
    return " · ".join(bits)


def _quota_html() -> str:
    """Chip trạng thái ở topbar. Toàn bộ đầu vào là số đếm của `GUARD` —
    không có chữ nào của người dùng hay của model đi qua đây, nên `gr.HTML`
    là an toàn. Đèn trạng thái là một chấm CSS, không phải emoji."""
    snap = GUARD.snapshot()
    state_cls, state_text = ("off", "tạm khoá") if snap["tripped"] else ("on", "đang mở")
    return (
        '<div class="quota">'
        f'<span class="dot {state_cls}"></span>'
        f"<span>sinh câu trả lời {state_text}</span>"
        '<span class="sep">·</span>'
        f"<span>{snap['used_today']}/{snap['daily_total']} câu hôm nay</span>"
        '<span class="sep">·</span>'
        f"<span>trần ~${snap['max_usd_per_day']}/ngày</span>"
        "</div>"
    )


async def answer(
    question: str,
    history: list[dict[str, str]],
    request: gr.Request,
) -> Any:
    """Một lượt. Yield `(chatbot, sources, stats, quota)`."""
    question = (question or "").strip()
    history = list(history or [])
    if not question:
        yield history, "", "", _quota_html()
        return

    history = [*history, {"role": "user", "content": question}]
    yield history, "_đang truy hồi…_", "", _quota_html()

    client = client_key_of(getattr(request, "headers", None), _host_of(request))
    # ⚠️ Xin phép TRƯỚC khi chạm model, và `commit` là nguyên tử — xem quyết
    # định 1 và ghi chú về cửa sổ đua ở `guard.py`.
    verdict: Verdict = GUARD.commit(client)

    if SERVICE.llm is None:
        history.append(
            {
                "role": "assistant",
                "content": (
                    "Bản demo chưa được cấu hình khoá sinh (`DEEPSEEK_API_KEY`), "
                    "nên chưa trả lời được. Phần truy hồi cũng cần lượt chuẩn bị "
                    "này nên tạm thời chưa chạy."
                ),
            }
        )
        yield history, "", "", _quota_html()
        return

    turn = await SERVICE.prepare(DEMO_PRINCIPAL, question=question, conversation_id=None)
    sources = turn.sources()
    sources_md = _sources_markdown(sources, None)

    if not verdict.allowed:
        # ⭐ Từ chối **sinh**, không từ chối **truy hồi**: truy hồi không tốn
        # tiền, và một người chạm trần vẫn xem được hệ thống tìm ra gì. Đó là
        # phần đáng xem nhất của một demo RAG.
        history.append({"role": "assistant", "content": verdict.reason})
        yield history, sources_md, "", _quota_html()
        return

    history.append({"role": "assistant", "content": ""})
    meta: dict[str, Any] = {}
    citations: dict[str, Any] | None = None
    text = ""
    async for event in SERVICE.stream_turn(turn):
        if event.event == "meta":
            meta = dict(event.data)
        elif event.event == "sources":
            sources_md = _sources_markdown(event.data.get("sources", sources), None)
        elif event.event == "delta":
            text += event.data.get("text", "")
            history[-1]["content"] = text
            yield history, sources_md, "", _quota_html()
        elif event.event == "citations":
            citations = dict(event.data)
            sources_md = _sources_markdown(sources, citations)
        elif event.event == "error":
            history[-1]["content"] = text + f"\n\nLỗi: {event.data.get('message', 'lỗi không rõ')}"
            yield history, sources_md, "", _quota_html()
        elif event.event == "done":
            yield history, sources_md, _stats_markdown(dict(event.data), meta), _quota_html()
            return
    yield history, sources_md, "", _quota_html()


def _host_of(request: gr.Request) -> str | None:
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client is not None else None


ABOUT = f"""
### Hệ thống này là gì

RAG trên **60 tài liệu World Bank về Việt Nam** (40 EN + 20 VI, 15.814 chunk),
[CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/) — bản quyền
thuộc World Bank, demo này chỉ phân phối lại.

| | |
|---|---|
| Bundle đang phục vụ | `{BOOT["bundle"]}` |
| Truy hồi | `{BOOT["retriever"]}` |
| Khởi động (Space này) | {BOOT["tong_khoi_dong_s"]} s |

Truy hồi **hybrid** (BGE-M3 dense + sparse, hợp nhất RRF) rồi xếp lại bằng
cross-encoder `bge-reranker-v2-m3` trên ZeroGPU; sinh bằng `deepseek-v4-flash`.
Mọi câu trả lời phải trích dẫn `[n]`, và **máy chủ tự đối chiếu** từng trích dẫn
với đúng chunk nó chỉ vào — nhãn *trích dẫn đã xác minh* ở bảng nguồn là phán
quyết của máy chủ, không phải lời của model.

Mã nguồn, số đo eval và báo cáo từng hạng mục:
**[github.com/Enigmask22/RAG-Chatbot](https://github.com/Enigmask22/RAG-Chatbot)**

### Bản demo này khác hệ thống đầy đủ ở đâu

Không có Postgres (không lưu lịch sử), không có Redis (không có cache — mỗi câu
là một lượt sinh thật), không có xác thực. Hạn mức chống lạm dụng là trần tổng
theo ngày cộng trần theo khách; trần theo khách **giả được** bằng một header và
điều đó được nói ra thay vì giấu.
"""

#: Toàn bộ "thiết kế" nằm ở đây, không ở component: chữ IBM Plex (sans cho
#: giao diện, mono cho số đo), một màu nhấn duy nhất, đèn trạng thái là chấm
#: CSS thay cho emoji, và footer mặc định của Gradio được ẩn đi. Màu chữ/viền
#: đi qua biến của Gradio (`--body-text-color…`, `--border-color-primary`) nên
#: theme sáng/tối đều tự đúng mà không cần hai bảng màu.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root { --app-accent: #0e9382; }

.gradio-container {
  font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif !important;
  max-width: 1220px !important;
  margin: 0 auto !important;
}
footer { display: none !important; }

#topbar {
  border-bottom: 1px solid var(--border-color-primary);
  padding: 4px 0 14px;
  margin-bottom: 8px;
  align-items: flex-end;
}
.brand-row { display: flex; align-items: center; gap: 10px; }
.brand-mark {
  width: 11px; height: 11px; display: inline-block;
  background: var(--app-accent); transform: rotate(45deg); border-radius: 2px;
}
.brand-name {
  font-size: 21px; font-weight: 650; letter-spacing: -0.02em;
  color: var(--body-text-color);
}
.brand-tag {
  font-size: 10.5px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--app-accent);
  border: 1px solid var(--app-accent); border-radius: 999px; padding: 2px 9px;
}
.brand-sub { margin-top: 5px; font-size: 13.5px; color: var(--body-text-color-subdued); }

.quota {
  display: flex; justify-content: flex-end; align-items: center; gap: 8px;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 12.5px; color: var(--body-text-color-subdued); padding-bottom: 3px;
}
.quota .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.quota .dot.on { background: #10b981; box-shadow: 0 0 0 3px rgba(16, 185, 129, .18); }
.quota .dot.off { background: #ef4444; box-shadow: 0 0 0 3px rgba(239, 68, 68, .18); }
.quota .sep { opacity: .5; }

.panel-label {
  font-size: 11px; font-weight: 600; letter-spacing: .14em;
  text-transform: uppercase; color: var(--body-text-color-subdued);
  margin: 2px 0 6px 2px;
}
#src-col {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px; padding: 14px 16px !important; align-self: stretch;
}
#sources { max-height: 660px; overflow-y: auto; }
#sources pre, #sources code { font-size: 12px; }

#stats, #stats * {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-size: 12px !important; color: var(--body-text-color-subdued) !important;
}

#examples button {
  border-radius: 999px; font-size: 13px;
  width: auto; max-width: calc(50% - 4px);
}
#examples button .gallery {
  white-space: nowrap !important; overflow: hidden; text-overflow: ellipsis;
}
#examples > .gallery { display: flex; flex-direction: row; flex-wrap: wrap; gap: 8px; }
#examples .label svg { display: none; }
#examples .label {
  font-size: 11px; font-weight: 600; letter-spacing: .14em;
  text-transform: uppercase; color: var(--body-text-color-subdued);
}
"""

BRAND_HTML = """
<div id="brand">
  <div class="brand-row">
    <span class="brand-mark"></span>
    <span class="brand-name">RAG Platform</span>
    <span class="brand-tag">demo công khai</span>
  </div>
  <div class="brand-sub">Hỏi đáp trên 60 báo cáo World Bank về Việt Nam —
  mọi trích dẫn được máy chủ đối chiếu với nguyên văn nguồn</div>
</div>
"""

with gr.Blocks(title="RAG Platform — demo", fill_height=True, css=CSS) as demo:
    with gr.Row(elem_id="topbar"):
        gr.HTML(BRAND_HTML)
        # Chip trạng thái: gr.HTML vì _quota_html chỉ chứa số đếm của GUARD —
        # không có chữ của người dùng hay của model. Mọi bồn chứa nội dung
        # không tin được vẫn là Markdown + sanitize bên dưới.
        quota = gr.HTML(_quota_html())
    with gr.Row():
        with gr.Column(scale=3):
            # ⚠️ `sanitize_html` khai TƯỜNG MINH dù mặc định đã là True. Đây
            # là nơi chữ của model đi ra màn hình, và một mặc định là thứ đổi
            # được ở phiên bản sau mà không ai đọc changelog — `W6-01` đã đặt
            # cùng luật này cho trang HTML dưới dạng "không bao giờ innerHTML".
            chatbot = gr.Chatbot(height=460, sanitize_html=True, show_label=False, elem_id="chat")
            box = gr.Textbox(
                placeholder="Hỏi bằng tiếng Việt hoặc tiếng Anh…",
                show_label=False,
                submit_btn=True,
                elem_id="ask",
            )
            stats = gr.Markdown("", elem_id="stats")
            gr.Examples(
                examples=EXAMPLES,
                inputs=box,
                cache_examples=False,
                label="Câu hỏi mẫu",
                elem_id="examples",
            )
        with gr.Column(scale=2, elem_id="src-col"):
            gr.HTML('<div class="panel-label">Nguồn đã đưa cho model</div>')
            # ⚠️⚠️ Đây là bồn chứa nguy hiểm nhất của trang: nội dung chunk
            # corpus, thứ mà chính khung `sources` gắn cờ tiêm. Khai tường minh.
            sources_view = gr.Markdown("_Chưa có lượt nào._", sanitize_html=True, elem_id="sources")
    with gr.Accordion("Về hệ thống này", open=False):
        gr.Markdown(ABOUT)

    box.submit(
        answer,
        inputs=[box, chatbot],
        outputs=[chatbot, sources_view, stats, quota],
    ).then(lambda: "", outputs=box)

if __name__ == "__main__":
    demo.queue(max_size=int(os.environ.get("DEMO_QUEUE_MAX", "12"))).launch()
