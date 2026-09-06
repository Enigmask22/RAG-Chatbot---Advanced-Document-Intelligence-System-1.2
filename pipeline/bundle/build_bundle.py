"""Đóng gói một cấu hình **đã đo** thành `RagBundle` (`W4-01`).

Công việc thật của file này không phải là chép trường từ chỗ này sang chỗ kia —
đó là phần dễ. Nó là **từ chối đóng gói một lời nói dối**.

Một bundle là một phát biểu: *"những con số eval này thuộc về đúng cấu hình
này"*. Phát biểu ấy sai một cách rất tự nhiên và rất im lặng:

* chạy eval trên `rag_bgem3`, rồi đóng gói với `configs/.../bgem3-contextual.yaml`
  vì đó là file mở sẵn trong editor;
* build lại index sau khi eval xong, rồi đóng gói — collection cùng tên, nội
  dung khác;
* đổi một tham số chunking, quên chạy lại eval, đóng gói bằng số cũ.

Cả ba đều cho ra một manifest hợp lệ, ký được, checksum khớp. Nên phép kiểm phải
nằm ở đây, ở chỗ *sinh*, và nó phải so **vân tay** chứ không so tên: tên
collection dùng lại được, vân tay thì không.

Chạy:

    python -m pipeline.bundle.build_bundle \\
        --index-config configs/indexing/bgem3-contextual.yaml \\
        --index-report plans/reports/probes/index-bgem3-contextual.json \\
        --eval-run plans/reports/runs/bgem3-ctx-rr-c50-retrieval.json \\
        --generator deepseek-chat@2026-09 \\
        --version 0.1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.indexing.config import IndexConfig, load_index_config
from rag_core.bundle import (
    BundleComponents,
    BundleValidationError,
    ChunkingComponent,
    EmbeddingComponent,
    EvalReport,
    GateRecord,
    GateStatus,
    GenerationComponent,
    IndexComponent,
    JudgeSpec,
    PromptComponent,
    RagBundle,
    RerankComponent,
    RetrievalComponent,
    save_bundle,
)

__all__ = ["build_bundle", "main"]

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("bundles")

#: Tham số reranker không nằm trong `branch_options` của báo cáo eval — chúng bị
#: nén vào chuỗi `retriever`. Đọc lại từ đó thay vì mặc định, vì mặc định ở đây
#: là đúng cái mà `test_bundle.py` cấm.
_RERANK_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise BundleValidationError(
            "không đọc được `git rev-parse HEAD`. Bundle phải truy ngược được về "
            "commit đã sinh ra nó, nên đây là lỗi cứng chứ không phải trường bỏ trống."
        ) from exc
    return out.stdout.strip()


def _parse_rerank(retriever_name: str, branch_options: dict[str, Any]) -> RerankComponent | None:
    """Bóc cấu hình reranker ra khỏi chuỗi `retriever` của báo cáo eval.

    Ví dụ chuỗi: ``reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:
    BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50``

    ⚠️ Phân tích chuỗi là cách làm dở, và nó ở đây vì báo cáo eval của `W2` ghi
    tham số reranker vào tên thay vì vào `branch_options` — sửa chỗ đó là sửa
    định dạng của 15 file báo cáo đã công bố. Đổi lại, hàm này **ném** khi không
    bóc được, chứ không lặng lẽ rơi về giá trị mặc định.
    """
    if not retriever_name.startswith("reranked["):
        return None
    tail = retriever_name.split("]:", 1)[1] if "]:" in retriever_name else ""
    parts = tail.split(":")
    model = parts[0].split("@")[0] if parts and parts[0] else _RERANK_DEFAULT_MODEL

    max_length: int | None = None
    candidates: int | None = None
    for piece in parts[1:]:
        if piece.startswith("L") and piece[1:].isdigit():
            max_length = int(piece[1:])
        elif piece.startswith("n") and piece[1:].isdigit():
            candidates = int(piece[1:])
    if max_length is None or candidates is None:
        raise BundleValidationError(
            f"không bóc được `max_length`/`candidates` từ tên nhánh {retriever_name!r}. "
            "Đoán giá trị ở đây sẽ tạo ra một bundle mô tả sai hệ thống đã đo."
        )
    # ⚠️ **Không** mặc định. Bản đầu viết `.get("rerank_top_n", 6)` — tức bịa ra
    # một giá trị ngay dưới docstring nói rằng đoán ở đây tạo ra bundle mô tả sai
    # hệ thống đã đo. Thiếu `rerank_top_n` nghĩa là lần eval **không cắt**, và
    # `None` là cách nói ra điều đó.
    raw_top_n = branch_options.get("rerank_top_n")
    top_n = int(raw_top_n) if raw_top_n is not None else None
    return RerankComponent(model=model, candidates=candidates, top_n=top_n, max_length=max_length)


def _require_retriever_name(eval_config: dict[str, Any]) -> str:
    """Tên retriever là **bắt buộc** ở phía sinh, dù schema cho phép `None`.

    Hai chiều không đối xứng: schema phải chấp nhận `None` để bundle sinh trước
    `TD-38` còn nạp được, nhưng một bundle **mới** thiếu trường này là một bundle
    không tự kiểm được — và nó sẽ nạp xanh trên mọi cấu hình sai. Nên chỗ duy
    nhất được phép để trống là quá khứ.
    """
    name = str(eval_config.get("retriever", "")).strip()
    if not name:
        raise BundleValidationError(
            "báo cáo eval không có trường `config.retriever`, nên không đóng gói được "
            "danh tính của hệ thống đã đo (`TD-38`). Chạy lại eval bằng "
            "`pipeline.eval.retrieval` phiên bản hiện tại."
        )
    return name


def _check_provenance(
    config: IndexConfig, index_report: dict[str, Any], eval_run: dict[str, Any]
) -> None:
    """Ba nguồn phải nói về **cùng một** index. So vân tay, không so tên.

    Tên collection dùng lại được — `rag_bgem3_ctx` hôm nay và `rag_bgem3_ctx`
    sau khi build lại với `chunk_size` khác là hai index khác nhau mang cùng một
    tên. Vân tay thì không dùng lại được.
    """
    eval_config = eval_run.get("config", {})
    expected = config.fingerprint

    mismatches: list[str] = []
    for source, value in (
        ("báo cáo build index", index_report.get("fingerprint")),
        ("lượt chạy eval", eval_config.get("index_fingerprint")),
    ):
        status = config.fingerprint_status(value) if isinstance(value, str) else "mismatch"
        if status == "legacy":
            # `TD-82`: artifact sinh trên Windows trước bản vá. Cùng một config —
            # từ chối ở đây là từ chối đóng gói những số đo hợp lệ chỉ vì công
            # thức băm đã được sửa sau khi chúng được ghi.
            logger.warning(
                "%s khai vân tay theo công thức trước TD-82 (%s); chấp nhận vì "
                "nó mô tả đúng config `%s`. Bundle mới ghi vân tay hiện tại.",
                source,
                value,
                config.name,
            )
        elif status == "mismatch":
            mismatches.append(f"  {source}: {value}")
    if mismatches:
        raise BundleValidationError(
            f"vân tay index không khớp với `{config.name}` ({expected}):\n"
            + "\n".join(mismatches)
            + "\nSố eval thuộc về một index khác với config đang đóng gói. Chạy "
            "lại eval trên index này thay vì đóng gói chéo."
        )

    if eval_config.get("collection") != config.collection:
        raise BundleValidationError(
            f"eval chạy trên collection {eval_config.get('collection')!r} nhưng "
            f"config nói {config.collection!r}."
        )


# ---------------------------------------------------------------- tầng sinh


#: Metric của tầng sinh do **judge** chấm. Chỉ những metric này mới có
#: `unjudged_rate`: một metric tất định (`citation_validity`, `context_recall@5`)
#: không có judge để hỏng, nên gán cho nó tỉ lệ `0,0` là khai một phép kiểm
#: chưa từng chạy.
_JUDGED_METRICS = ("faithfulness", "answer_relevancy", "uncited_grounding", "misattribution")

#: Ánh xạ tên trong báo cáo `W5-01` → tên trong bundle. Tên bundle là tên mà
#: `configs/eval/gate.yaml` nói tới, nên đổi tên ở đây là đổi hợp đồng với gate.
_GENERATION_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "citation_coverage",
    "misattribution",
    "uncited_grounding",
    "context_precision@5",
    "context_recall@5",
)

#: Các trường của `BundleComponents` mô tả **đường truy hồi**. Đây là phần mà
#: một lần chạy tầng sinh phụ thuộc vào; `prompt`/`generation` thì không, vì
#: chính chúng là thứ bundle mới đang bổ sung.
_RETRIEVAL_SIDE = ("chunking", "embedding", "index", "retrieval", "rerank", "retriever_name")


def _unjudged_rate(block: Mapping[str, Any]) -> float:
    """`n_unjudged / số câu đã hỏi`. Mẫu số **không** phải số câu chấm được.

    Dùng mẫu số "chấm được" thì tỉ lệ tự nhỏ đi đúng ở những lần chạy hỏng nặng
    nhất — mất càng nhiều phán quyết, mẫu số càng bé, tỉ lệ trông càng đẹp.
    """
    unjudged = int(block.get("n_unjudged", 0))
    asked = int(block.get("n", 0)) + unjudged + int(block.get("n_not_a_claim", 0))
    return unjudged / asked if asked else 0.0


def _measured_on(generation_report: Mapping[str, Any], root: Path) -> RagBundle:
    """Bundle mà lần chạy tầng sinh đã thật sự được phục vụ bởi.

    ⭐⭐ Đây là chỗ dễ nói dối nhất trong cả file, và nó không có phép kiểm nào
    sẵn có: metric truy hồi đo **offline** trên một index, còn metric tầng sinh
    đo **qua HTTP** trên một server đang chạy một bundle. Hai nguồn ấy có thể
    thuộc về hai hệ thống khác nhau mà vẫn ghép thành một manifest hợp lệ.

    `answer_run` có ghi `bundle_versions`, nên phép kiểm là: nạp đúng bundle ấy
    lên và đòi phần **đường truy hồi** của nó trùng khít với bundle đang dựng.
    """
    from rag_core.bundle import bundle_dir_name, load_bundle

    versions = list(generation_report.get("bundle_versions") or [])
    if len(versions) != 1:
        raise BundleValidationError(
            f"lần chạy tầng sinh phục vụ bởi {versions or 'không bundle nào'} — cần đúng "
            "một. Trộn nhiều bundle trong một lần chạy thì con số không thuộc về bundle nào."
        )
    path = root / bundle_dir_name(str(versions[0])) / "manifest.json"
    if not path.is_file():
        raise BundleValidationError(
            f"lần chạy tầng sinh nói nó chạy trên bundle {versions[0]!r} nhưng không "
            f"tìm thấy {path}. Không đối chiếu được thì không đóng gói."
        )
    return load_bundle(path)


def _check_generation_provenance(components: BundleComponents, measured: RagBundle) -> None:
    differing = [
        field_name
        for field_name in _RETRIEVAL_SIDE
        if getattr(components, field_name) != getattr(measured.components, field_name)
    ]
    if differing:
        raise BundleValidationError(
            f"metric tầng sinh đo trên bundle {measured.bundle_version}, nhưng đường "
            f"truy hồi của bundle đang dựng khác ở {differing}. Gắn số của hệ thống này "
            "lên hệ thống khác là đúng loại lời nói dối mà module này tồn tại để chặn."
        )


def _judge_spec(
    generation_report: Mapping[str, Any], calibration: Mapping[str, Any] | None
) -> JudgeSpec:
    judge = dict(generation_report.get("judge") or {})
    if not judge.get("model"):
        raise BundleValidationError("báo cáo tầng sinh không nêu `judge.model`.")

    kappa: float | None = None
    if calibration is not None:
        cal_rubric = str(calibration.get("rubric", ""))
        rubrics = [str(r) for r in judge.get("rubrics", [])]
        if cal_rubric and cal_rubric not in rubrics:
            raise BundleValidationError(
                f"hiệu chỉnh đo trên rubric {cal_rubric!r} nhưng lần chạy dùng {rubrics}. "
                "κ của một rubric không nói gì về rubric khác (`W5-01` đo được rubric "
                "v1→v2 đổi một metric gấp đôi)."
            )
        if calibration.get("judge_model") and calibration["judge_model"] != judge["model"]:
            raise BundleValidationError(
                f"hiệu chỉnh đo trên judge {calibration['judge_model']!r} nhưng lần chạy "
                f"dùng {judge['model']!r}."
            )
        arm = ((calibration.get("agreement") or {}).get("judge_vs_human") or {}).get("population")
        if arm is None or arm.get("kappa") is None:
            raise BundleValidationError(
                "file hiệu chỉnh không có `agreement.judge_vs_human.population.kappa`."
            )
        kappa = float(arm["kappa"])

    return JudgeSpec(
        model=str(judge["model"]),
        temperature=0.0,
        kappa_vs_human=kappa,
        rubrics=tuple(str(r) for r in judge.get("rubrics", [])),
        reasoning=bool(judge["reasoning"]) if "reasoning" in judge else None,
        cache_digest=judge.get("cache_digest"),
    )


def _prompt_component(generation_report: Mapping[str, Any]) -> PromptComponent:
    from rag_core.generation import PromptRegistry

    specs = list(generation_report.get("prompt_specs") or [])
    if len(specs) != 1:
        raise BundleValidationError(
            f"lần chạy dùng {specs or 'không'} prompt — cần đúng một để bundle nêu được "
            "prompt nào đã sinh ra các con số này."
        )
    prompt_id, _, version = str(specs[0]).partition("@")
    prompt = PromptRegistry().get(prompt_id)
    if prompt.spec != specs[0]:
        raise BundleValidationError(
            f"registry hiện có {prompt.spec!r} nhưng lần chạy dùng {specs[0]!r}. "
            "Prompt đã đổi kể từ lần đo; đóng gói bây giờ sẽ mô tả sai."
        )
    return PromptComponent(id=prompt_id, version=int(version.lstrip("v")), hash=prompt.sha256)


def _generation_eval(
    generation_report: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    metrics_block = dict(generation_report.get("metrics") or {})
    values: dict[str, float] = {}
    for name in _GENERATION_METRICS:
        block = metrics_block.get(name)
        if block is None or block.get("value") is None:
            continue
        values[name] = float(block["value"])

    accuracy = dict(generation_report.get("citation_accuracy") or {})
    gate_metric = str(accuracy.get("gate_metric", "quote_level"))
    if gate_metric in accuracy:
        # ⭐ Bundle mang **một** con số tên `citation_accuracy`, và nó là con số
        # mà chính `W5-02` đã khai là gate_metric. Đóng gói cả hai cấp rồi để
        # gate chọn sau là mở đường cho việc chọn cấp nào cho vừa kết quả.
        values["citation_accuracy"] = float(accuracy[gate_metric]["value"])

    refusal = dict(generation_report.get("refusal") or {})
    if "refusal_accuracy" in refusal:
        values["refusal_accuracy"] = float(refusal["refusal_accuracy"]["value"])

    rates = {
        name: _unjudged_rate(metrics_block[name])
        for name in _JUDGED_METRICS
        if name in metrics_block
    }
    return values, rates


def build_bundle(
    *,
    config: IndexConfig,
    index_report: dict[str, Any],
    eval_run: dict[str, Any],
    version: str,
    generator: str | None = None,
    notes: str | None = None,
    generation_report: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    gen_max_tokens: int | None = None,
    gen_temperature: float | None = None,
    root: Path = DEFAULT_ROOT,
) -> RagBundle:
    """`generation_report` là báo cáo của `pipeline.eval.generation_metrics` (`W5-01`).

    Khi có nó, bundle mọc thêm `components.prompt`, `components.generation`,
    `eval.judge`, `eval.generation_metrics` và `eval.unjudged_rate` — và
    `evaluated_with_generator` được **đọc từ lần chạy** thay vì gõ tay. Đó là
    cách trả một lỗi có thật: cả hai bundle đang có trên đĩa đều ghi
    `deepseek-chat@2026-09`, mà `deepseek-chat` là bí danh (`W5-03` đo được nó
    được phục vụ bởi `deepseek-v4-flash`). Trường sinh ra để bảo đảm so
    like-for-like đang mang một danh tính không ổn định, và nó không ổn định vì
    nó được gõ tay.
    """
    _check_provenance(config, index_report, eval_run)

    eval_config: dict[str, Any] = eval_run.get("config", {})
    # ⚠️ Giữ **bản gốc**: `_parse_rerank` cần `rerank_top_n`, mà mấy dòng dưới lại
    # `pop` nó ra khỏi `branch_options` (đúng — nó thuộc khối `rerank`, không phải
    # option của nhánh nền). Bản đầu truyền nhầm `eval_config` cho `_parse_rerank`
    # thay vì dict này, nên `rerank_top_n` **không bao giờ** được đọc: dù lần eval
    # có nêu hay không, bundle luôn ghi giá trị bịa.
    raw_options: dict[str, Any] = dict(eval_config.get("branch_options", {}))
    branch_options: dict[str, Any] = dict(raw_options)
    # `base` mô tả nhánh nền của reranked; nó là `mode`, không phải một option.
    base_mode = str(branch_options.pop("base", eval_config.get("retrieval_mode", "dense")))
    branch_options.pop("rerank_candidates", None)
    branch_options.pop("rerank_top_n", None)

    components = BundleComponents(
        chunking=ChunkingComponent(
            strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            contextual=config.contextual.enabled,
            chunking_fingerprint=config.chunking_fingerprint,
        ),
        embedding=EmbeddingComponent(
            model=config.embedding_model,
            dim=int(index_report["embedding_dim"]),
            normalize=config.embedding_normalize,
        ),
        index=IndexComponent(
            backend="qdrant",
            collection=config.collection,
            fingerprint=config.fingerprint,
            n_chunks=int(index_report["collection_count"]),
            n_documents=int(index_report["n_documents_indexed"]),
        ),
        retrieval=RetrievalComponent(
            mode=base_mode,
            top_k=int(eval_config.get("top_k", 20)),
            options=branch_options,
        ),
        rerank=_parse_rerank(str(eval_config.get("retriever", "")), raw_options),
        # ⭐ `TD-38`: chép nguyên tên retriever của lần eval. Không suy lại từ các
        # trường ở trên — suy lại là dựng một *bản sao thứ hai* của quy ước đặt
        # tên, và hai bản sao sẽ lệch nhau ở đúng lúc không ai nhìn. Đây là chuỗi
        # mà `QdrantRuntimeBuilder` sẽ so với runtime nó vừa dựng.
        retriever_name=_require_retriever_name(eval_config),
        # Tầng sinh chưa được dựng (`W4-08`/`W4-11`). Bundle khai thiếu thay vì
        # bịa — xem `BundleComponents.prompt`.
        prompt=None,
        generation=None,
    )

    generation_values: dict[str, float] = {}
    unjudged: dict[str, float] = {}
    judge: JudgeSpec | None = None
    end_to_end: float | None = None
    if generation_report is not None:
        measured = _measured_on(generation_report, root)
        _check_generation_provenance(components, measured)
        if gen_max_tokens is None or gen_temperature is None:
            raise BundleValidationError(
                "`--generation-run` cần cả `--gen-max-tokens` và `--gen-temperature`. "
                "Artifact answer run chưa ghi lại hai tham số này (`TD-69`), nên chúng "
                "phải được **khai**; mặc định ở đây sẽ khai hộ một điều chưa ai kiểm."
            )
        served = list(generation_report.get("models") or [])
        components = components.model_copy(
            update={
                "prompt": _prompt_component(generation_report),
                "generation": GenerationComponent(
                    primary=str(served[0]) if served else str(generator or ""),
                    max_tokens=gen_max_tokens,
                    temperature=gen_temperature,
                ),
            }
        )
        judge = _judge_spec(generation_report, calibration)
        generation_values, unjudged = _generation_eval(generation_report)
        end_to_end = (generation_report.get("latency_ms") or {}).get("p95")
        if served:
            generator = str(served[0])

    if not generator:
        raise BundleValidationError(
            "`evaluated_with_generator` bắt buộc — gate chỉ so được like-for-like."
        )

    latency = eval_run.get("latency_ms", {})
    report = EvalReport(
        golden_set=Path(eval_config.get("golden", "golden_v1")).stem,
        n_queries=int(eval_run["n_scored"]),
        evaluated_with_generator=generator,
        judge=judge,
        retrieval_metrics={str(k): float(v) for k, v in eval_run.get("overall", {}).items()},
        generation_metrics=generation_values,
        unjudged_rate=unjudged,
        p95_latency_ms=float(latency["p95"]) if "p95" in latency else None,
        p95_end_to_end_ms=float(end_to_end) if end_to_end is not None else None,
    )

    return RagBundle(
        bundle_version=version,
        created_at=datetime.now(UTC),
        git_sha=_git_sha(),
        components=components,
        eval=report,
        gate=GateRecord(status=GateStatus.NOT_RUN),
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--index-report", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--version", required=True, help="semver, ví dụ `0.1.0`.")
    parser.add_argument(
        "--generator",
        default=None,
        help=(
            "`evaluated_with_generator`. Bắt buộc TRỪ KHI có `--generation-run` — "
            "khi ấy nó được đọc từ model thực tế đã phục vụ lần chạy, thay vì gõ tay."
        ),
    )
    parser.add_argument(
        "--generation-run",
        type=Path,
        default=None,
        help="Báo cáo `generation_metrics` của `W5-01` (`*-generation.json`).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Báo cáo `W5-04` — nguồn của `judge.kappa_vs_human`. Không gõ tay κ.",
    )
    parser.add_argument("--gen-max-tokens", type=int, default=None)
    parser.add_argument("--gen-temperature", type=float, default=None)
    parser.add_argument("--notes")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    def _read(path: Path | None) -> dict[str, Any] | None:
        return None if path is None else json.loads(path.read_text(encoding="utf-8"))

    bundle = build_bundle(
        config=load_index_config(args.index_config),
        index_report=json.loads(args.index_report.read_text(encoding="utf-8")),
        eval_run=json.loads(args.eval_run.read_text(encoding="utf-8")),
        version=args.version,
        generator=args.generator,
        notes=args.notes,
        generation_report=_read(args.generation_run),
        calibration=_read(args.calibration),
        gen_max_tokens=args.gen_max_tokens,
        gen_temperature=args.gen_temperature,
        root=args.root,
    )
    path = save_bundle(bundle, args.root, overwrite=args.overwrite)
    logger.info("đã ghi bundle %s → %s", bundle.bundle_version, path)
    logger.info("serves_generation=%s", bundle.serves_generation)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
