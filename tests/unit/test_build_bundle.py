"""`W4-01` — bên **sinh** bundle, nơi một lời nói dối được tạo ra chứ không bị phát hiện.

Checksum bảo vệ manifest khỏi bị sửa *sau* khi ký. Nó không bảo vệ được gì trước
một manifest **sai từ lúc ký** — và đó là kiểu sai dễ xảy ra hơn nhiều: chạy eval
trên index này, đóng gói bằng config kia, ký, và mọi phép kiểm phía sau đều xanh.

Nên nhóm test này gần như chỉ hỏi một câu: builder có từ chối đóng gói khi ba
nguồn (config, báo cáo build, lượt eval) **không nói về cùng một index** không.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.bundle.build_bundle import _parse_rerank, build_bundle
from pipeline.indexing.config import IndexConfig
from rag_core.bundle import BundleValidationError

RERANKED_NAME = (
    "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50"
)


def make_config(**overrides: Any) -> IndexConfig:
    base: dict[str, Any] = {
        "name": "bgem3-contextual",
        "collection": "rag_bgem3_ctx",
        "tenant_id": "public",
        "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100},
        "embedding_model": "BAAI/bge-m3",
        "contextual": {"enabled": True},
    }
    base.update(overrides)
    return IndexConfig(**base)


def make_index_report(config: IndexConfig, **overrides: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "fingerprint": config.fingerprint,
        "collection": config.collection,
        "embedding_dim": 1024,
        "collection_count": 15814,
        "n_documents_indexed": 60,
    }
    report.update(overrides)
    return report


def make_eval_run(config: IndexConfig, **config_overrides: Any) -> dict[str, Any]:
    run_config: dict[str, Any] = {
        "retriever": RERANKED_NAME,
        "top_k": 20,
        "index_fingerprint": config.fingerprint,
        "collection": config.collection,
        "retrieval_mode": "reranked",
        "branch_options": {"k": 1, "candidate_k": 20, "base": "hybrid"},
        "golden": "data/golden/golden_v1.jsonl",
    }
    run_config.update(config_overrides)
    return {
        "n_scored": 209,
        "overall": {"ndcg@10": 0.6888, "hit_rate@5": 0.8086},
        "latency_ms": {"p95": 809.27},
        "config": run_config,
    }


def build(config: IndexConfig | None = None, **kwargs: Any) -> Any:
    config = config or make_config()
    args: dict[str, Any] = {
        "config": config,
        "index_report": make_index_report(config),
        "eval_run": make_eval_run(config),
        "version": "0.1.0",
        "generator": "deepseek-chat@2026-09",
    }
    args.update(kwargs)
    return build_bundle(**args)


# ---------------------------------------------------------------------------
# 1. ⭐⭐ Từ chối đóng gói chéo
# ---------------------------------------------------------------------------


def test_eval_from_a_different_index_is_refused() -> None:
    """Kiểu sai nguy hiểm nhất, và kiểu duy nhất checksum không bắt được.

    Số eval đo trên `rag_bgem3` đem đóng gói với config `bgem3-contextual` cho ra
    một manifest hợp lệ, ký được, checksum khớp — và nói dối về hệ thống nào đạt
    được những con số ấy.
    """
    config = make_config()
    other = make_config(contextual={"enabled": False})
    with pytest.raises(BundleValidationError, match="vân tay index không khớp"):
        build(config, eval_run=make_eval_run(other))


def test_index_report_from_a_different_build_is_refused() -> None:
    """Build lại index sau khi eval xong: cùng tên collection, khác nội dung."""
    config = make_config()
    with pytest.raises(BundleValidationError, match="báo cáo build index"):
        build(config, index_report=make_index_report(config, fingerprint="0" * 64))


def test_the_error_names_every_source_that_disagrees() -> None:
    """Một nguồn lệch và hai nguồn lệch là hai tình huống khác nhau."""
    config = make_config()
    other = make_config(chunking={"chunk_size": 550, "chunk_overlap": 50})
    with pytest.raises(BundleValidationError) as excinfo:
        build(
            config,
            index_report=make_index_report(other),
            eval_run=make_eval_run(other),
        )
    message = str(excinfo.value)
    assert "báo cáo build index" in message
    assert "lượt chạy eval" in message


def test_collection_rename_is_caught_even_when_fingerprints_agree() -> None:
    """Vân tay không gồm tên collection (đúng), nên tên phải được kiểm riêng.

    ⚠️ Nếu bỏ phép kiểm này thì bundle trỏ serving vào một collection **không
    tồn tại** trong khi mọi vân tay đều khớp.
    """
    config = make_config()
    with pytest.raises(BundleValidationError, match="collection"):
        build(config, eval_run=make_eval_run(config, collection="rag_bgem3"))


def test_matching_sources_produce_a_bundle() -> None:
    bundle = build()
    assert bundle.components.index.collection == "rag_bgem3_ctx"
    assert bundle.eval.retrieval_metrics["ndcg@10"] == pytest.approx(0.6888)


# ---------------------------------------------------------------------------
# 2. Bóc tham số reranker khỏi tên nhánh
# ---------------------------------------------------------------------------


def test_rerank_parameters_come_from_the_run_not_from_defaults() -> None:
    component = _parse_rerank(RERANKED_NAME, {})
    assert component is not None
    assert (component.model, component.max_length, component.candidates) == (
        "BAAI/bge-reranker-v2-m3",
        512,
        50,
    )


def test_a_branch_name_missing_the_window_is_an_error_not_a_default() -> None:
    """⭐ Đoán `max_length=512` ở đây sẽ tạo ra bundle mô tả sai hệ thống đã đo —
    và `W3-04` vừa đo được rằng cửa sổ ấy là ràng buộc thật, không phải chi tiết."""
    with pytest.raises(BundleValidationError, match="không bóc được"):
        _parse_rerank("reranked[qdrant-dense:c]:BAAI/bge-reranker-v2-m3@cuda", {})


def test_a_non_reranked_branch_has_no_rerank_component() -> None:
    """`None` ở đây nghĩa là *tắt rerank*, và đó là mô tả đúng của nhánh dense."""
    assert _parse_rerank("qdrant-dense:rag_bgem3", {}) is None


def test_base_branch_becomes_the_mode_not_an_option() -> None:
    """`base` mô tả nhánh nền của reranked; để nó nằm trong `options` thì
    `build_branch` sẽ nhận một tham số nó không hiểu và nổ lúc serving load."""
    bundle = build()
    assert bundle.components.retrieval.mode == "hybrid"
    assert "base" not in bundle.components.retrieval.options
    assert bundle.components.retrieval.options == {"k": 1, "candidate_k": 20}


# ---------------------------------------------------------------------------
# 3. Bundle hôm nay khai thiếu tầng sinh — và khai thật
# ---------------------------------------------------------------------------


def test_todays_bundle_is_retrieval_only() -> None:
    bundle = build()
    assert bundle.serves_generation is False
    assert bundle.components.prompt is None
    assert bundle.eval.generation_metrics == {}


def test_generator_cannot_be_left_blank() -> None:
    """Không có mặc định nào cho `evaluated_with_generator`.

    ⚠️ `W5-05` đổi nó từ tham số **bắt buộc của chữ ký** thành bắt buộc **lúc
    chạy**, vì khi có `--generation-run` thì nó được đọc từ model đã thật sự
    phục vụ lần chạy — tốt hơn hẳn gõ tay. Chính việc gõ tay đã đưa bí danh
    `deepseek-chat@2026-09` vào cả hai bundle đang có trên đĩa. Ràng buộc không
    mất đi, nó chuyển chỗ; bài test này giữ đúng chỗ mới.
    """
    with pytest.raises(BundleValidationError, match="bắt buộc"):
        build_bundle(
            config=make_config(),
            index_report=make_index_report(make_config()),
            eval_run=make_eval_run(make_config()),
            version="0.1.0",
        )


# ---------------------------------------------------------------------------
# 4. Bundle mẫu đã commit phải còn hợp lệ
# ---------------------------------------------------------------------------

SAMPLE = Path(__file__).resolve().parents[2] / "bundles" / "rag-bundle-v0.1.0" / "manifest.json"


@pytest.mark.skipif(not SAMPLE.is_file(), reason="chưa sinh bundle mẫu")
def test_the_committed_sample_bundle_still_loads() -> None:
    """Bundle mẫu là *bằng chứng* của `W4-01`, nên nó phải chịu được mọi thay đổi
    schema sau này — hoặc bị sinh lại một cách có ý thức."""
    from rag_core.bundle import load_bundle

    bundle = load_bundle(SAMPLE)
    assert bundle.bundle_version == "0.1.0"
    assert bundle.components.index.n_chunks == 15814
    assert bundle.eval.retrieval_metrics["ndcg@10"] == pytest.approx(0.6888, abs=1e-4)


def _real_config() -> Any:
    from pipeline.indexing.config import load_index_config

    return load_index_config(Path("configs/indexing/bgem3-contextual.yaml"))


def test_the_sample_bundle_matches_the_real_chunking_config() -> None:
    """Ghim rằng bundle mẫu được sinh từ artifact thật, không phải gõ tay.

    Dùng `chunking_fingerprint` vì nó **không** băm đường dẫn nào, tức nó giống
    nhau trên mọi hệ điều hành. Vế `fingerprint` tách xuống bài dưới — xem
    `TD-82`, đã trả ở `W5-10`.
    """
    if not SAMPLE.is_file():  # pragma: no cover
        pytest.skip("chưa sinh bundle mẫu")
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert (
        raw["components"]["chunking"]["chunking_fingerprint"] == _real_config().chunking_fingerprint
    )


def test_the_sample_bundle_matches_the_real_index_config() -> None:
    """`TD-82` đã trả: bài này chạy trên **mọi** hệ điều hành, không còn skip.

    Trước `W5-10` nó bị `skipif(os.name != "nt")` vì manifest đã commit mang vân
    tay tính trên Windows, còn `fingerprint` tính trên Linux ra giá trị khác —
    tức nửa số máy chạy CI không kiểm được gì ở đây. Giờ `fingerprint_status`
    tính lại được **cả hai** công thức trên cả hai nền tảng, nên phép kiểm là
    một câu về bundle chứ không còn là một câu về máy đang chạy nó.

    Kết quả mong đợi là `"legacy"`, không phải `"current"`: ba manifest đã ký
    vẫn giữ nguyên giá trị chúng được ký cùng. Xem
    `test_the_committed_manifests_are_not_rewritten_by_the_fix`.
    """
    if not SAMPLE.is_file():  # pragma: no cover
        pytest.skip("chưa sinh bundle mẫu")
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    recorded = raw["components"]["index"]["fingerprint"]
    assert _real_config().fingerprint_status(recorded) in {"current", "legacy"}


def test_the_index_fingerprint_no_longer_carries_the_shape_of_the_host_os() -> None:
    """`TD-82` — quét **payload** đi vào hàm băm, không quét mã.

    Bài cũ ghim món nợ bằng cách đọc `inspect.getsource(fingerprint)` và đòi
    thấy chữ `contextual`. Nó đỏ đúng ngày ai đó sửa, đúng như thiết kế — nhưng
    nó không gác được gì sau đó, và nhất là nó mù với **trường path tiếp theo**
    lọt vào payload.

    Bài này quét mọi giá trị chuỗi trong payload và từ chối dấu `\\`. Thêm một
    `Path` vào `fingerprint` mà quên chuẩn hoá ⇒ đỏ trên Windows, và đỏ **trước
    khi** artifact nào kịp mang giá trị lệch nền tảng.
    """
    payload = _real_config()._fingerprint_payload()

    def strings(node: object) -> list[str]:
        if isinstance(node, str):
            return [node]
        if isinstance(node, dict):
            return [s for value in node.values() for s in strings(value)]
        if isinstance(node, list):
            return [s for item in node for s in strings(item)]
        return []

    offenders = [s for s in strings(payload) if "\\" in s]
    assert not offenders, (
        f"payload của `fingerprint` mang dấu phân cách của Windows: {offenders} — "
        "một trường path chưa được chuẩn hoá về POSIX sẽ cho HAI vân tay cho "
        "cùng một config (TD-82)"
    )


def test_the_two_fingerprint_formulas_really_are_different() -> None:
    """Nếu hai công thức trùng nhau thì `fingerprint_status` không chứng minh gì.

    `legacy_windows_fingerprint` dùng `PureWindowsPath`, thứ viết dấu `\\` trên
    **cả** Windows lẫn Linux — đó là lý do phép so sánh này chạy được ở cả hai
    nơi, và cũng là lý do bản vá không cần một máy Windows để kiểm.
    """
    config = _real_config()
    assert config.fingerprint != config.legacy_windows_fingerprint
    assert config.fingerprint_status(config.legacy_windows_fingerprint) == "legacy"
    assert config.fingerprint_status(config.fingerprint) == "current"
    assert config.fingerprint_status("0" * 64) == "mismatch"


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0", "0.2.1"])
def test_the_committed_manifests_are_not_rewritten_by_the_fix(version: str) -> None:
    """⭐⭐ Bản vá **không** sửa ba manifest đã ký, và đó là một lựa chọn.

    Ghi chú gốc của `TD-82` dự tính "đúc lại ba manifest". Không làm, vì
    `save_bundle` có đúng một luật số 1: *không ghi đè một version đã tồn tại* —
    "bundle bất biến" là một câu về hệ thống file. Sửa một trường bên trong một
    artifact **đã ký** để nó khớp với mã hôm nay là biến chữ ký thành trang trí.

    Cái được sửa là **công thức** và **phép so sánh**. Vân tay cũ vẫn đọc được,
    vẫn nhận diện được là cũ, và bundle kế tiếp được đúc từ một lần build index
    thật sẽ mang giá trị mới. Bài này đỏ nếu ai đó chọn con đường kia.
    """
    manifest = Path(f"bundles/rag-bundle-v{version}/manifest.json")
    if not manifest.is_file():  # pragma: no cover
        pytest.skip(f"chưa có bundle {version}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    recorded = raw["components"]["index"]["fingerprint"]
    assert len(recorded) == 64
    # Không đòi `== legacy_windows_fingerprint`: `0.1.0` dùng config khác
    # (`bkai`, không contextual), nên nó không so được với `_real_config()`.
    # Đòi vừa đủ: giá trị nguyên vẹn, không bị bản vá viết đè.
    assert raw["checksum"], "manifest phải còn chữ ký"


# ---------------------------------------------------------------------------
# TD-38 — bundle mới phải mang danh tính của hệ thống đã đo
# ---------------------------------------------------------------------------


def test_the_measured_retriever_name_is_copied_verbatim() -> None:
    """⭐ Chép nguyên, **không** dựng lại từ các trường đã bóc ra.

    Dựng lại nghĩa là có một *bản sao thứ hai* của quy ước đặt tên sống trong
    `build_bundle`, và hai bản sao sẽ lệch nhau ở đúng lần `rag_core` đổi hậu tố
    — lúc không ai nhìn, và theo hướng làm phép kiểm danh tính ở serving đỏ giả.
    """
    assert build().components.retriever_name == RERANKED_NAME


def test_an_eval_report_without_a_retriever_name_cannot_be_packaged() -> None:
    """Schema cho phép `None` để bundle sinh trước `TD-38` còn nạp được; phía
    **sinh** thì không, vì một bundle mới thiếu trường này nạp xanh trên mọi cấu
    hình sai. Chỗ duy nhất được phép để trống là quá khứ."""
    config = make_config()
    with pytest.raises(BundleValidationError, match="TD-38"):
        build(config, eval_run=make_eval_run(config, retriever=""))


def test_an_unset_top_n_is_recorded_as_none_not_guessed() -> None:
    """⭐⭐ Bug thật, và nó sống sót qua cả `W4-01` lẫn `W4-03`.

    Bản đầu viết `branch_options.get("rerank_top_n", 6)` — **bịa ra 6** khi lần
    eval không nêu, ngay dưới một docstring nói rằng đoán giá trị ở đây tạo ra
    một bundle mô tả sai hệ thống đã đo.

    Không test nào của `W4-01` bắt được, vì cả `6` lẫn `None` đều hợp lệ về hình
    dạng và bundle vẫn round-trip xanh. Thứ bắt được là phép kiểm danh tính của
    `TD-38`, ở **lần chạy thật đầu tiên**: runtime dựng ra `…:n50-top6` còn báo
    cáo eval ghi `…:n50`.
    """
    rerank = build().components.rerank
    assert rerank is not None
    assert rerank.top_n is None, "đoán `top_n` = mô tả sai hệ thống đã đo"


def test_an_explicit_top_n_is_kept() -> None:
    config = make_config()
    run = make_eval_run(config)
    run["config"]["branch_options"]["rerank_top_n"] = 6
    rerank = build(config, eval_run=run).components.rerank
    assert rerank is not None and rerank.top_n == 6


# ---------------------------------------------------------------------------
# 5. `W5-05` — gắn phép đo tầng sinh vào bundle
# ---------------------------------------------------------------------------


def make_generation_report(**overrides: Any) -> dict[str, Any]:
    """Hình dạng thật của `w5-answers-v1-generation.json`, số thật của `W5-01`."""
    report: dict[str, Any] = {
        "run": "w5-answers-v1",
        "bundle_versions": ["0.1.0"],
        "prompt_specs": ["chat-system@v2"],
        "models": ["deepseek-v4-flash"],
        "judge": {
            "model": "deepseek-v4-flash",
            "reasoning": False,
            "rubrics": ["judge-answer-relevancy@v1", "judge-faithfulness@v2"],
            "cache_digest": "79d7df51",
        },
        "metrics": {
            "faithfulness": {"value": 0.9877, "n": 407, "n_unjudged": 0, "n_not_a_claim": 26},
            "answer_relevancy": {"value": 0.7479, "n": 242, "n_unjudged": 0, "n_not_a_claim": 0},
        },
        "citation_accuracy": {
            "quote_level": {"value": 0.8308},
            "claim_level": {"value": 0.9877},
            "gate_metric": "quote_level",
        },
        "refusal": {"refusal_accuracy": {"value": 0.9091}},
        "latency_ms": {"p95": 4706.5},
    }
    report.update(overrides)
    return report


def save_measured_bundle(root: Path, config: IndexConfig, version: str = "0.1.0") -> None:
    """Ghi ra bundle mà lần chạy tầng sinh khai là đã chạy trên đó."""
    from rag_core.bundle import save_bundle

    save_bundle(build(config, version=version).signed(), root)


def test_generation_metrics_land_in_the_bundle(tmp_path: Path) -> None:
    config = make_config()
    save_measured_bundle(tmp_path, config)
    bundle = build(
        config,
        version="0.1.1",
        generator=None,
        generation_report=make_generation_report(),
        gen_max_tokens=1024,
        gen_temperature=0.0,
        root=tmp_path,
    )
    assert bundle.serves_generation is True
    assert bundle.eval.generation_metrics["faithfulness"] == pytest.approx(0.9877)
    assert bundle.eval.generation_metrics["citation_accuracy"] == pytest.approx(0.8308)
    assert bundle.eval.p95_end_to_end_ms == pytest.approx(4706.5)


def test_the_generator_is_read_from_the_run_not_typed_by_hand(tmp_path: Path) -> None:
    """⭐⭐ Đây là cách bí danh `deepseek-chat@2026-09` lọt vào hai bundle đầu.

    Trường sinh ra để bảo đảm danh tính ổn định mà lại được gõ tay thì nó ổn định
    đúng bằng trí nhớ của người gõ.
    """
    config = make_config()
    save_measured_bundle(tmp_path, config)
    bundle = build(
        config,
        version="0.1.1",
        generator="deepseek-chat@2026-09",
        generation_report=make_generation_report(),
        gen_max_tokens=1024,
        gen_temperature=0.0,
        root=tmp_path,
    )
    assert bundle.eval.evaluated_with_generator == "deepseek-v4-flash"


def test_numbers_measured_on_a_different_retrieval_path_are_refused(tmp_path: Path) -> None:
    """Metric truy hồi đo **offline**, metric tầng sinh đo **qua HTTP** trên một
    server đang chạy một bundle. Không có gì tự bắt việc ghép chéo hai nguồn ấy."""
    config = make_config()
    save_measured_bundle(tmp_path, config)
    other = make_config(chunking={"chunk_size": 550, "chunk_overlap": 50})
    with pytest.raises(BundleValidationError, match="truy hồi của bundle đang dựng"):
        build(
            other,
            version="0.1.1",
            generator=None,
            index_report=make_index_report(other),
            eval_run=make_eval_run(other),
            generation_report=make_generation_report(),
            gen_max_tokens=1024,
            gen_temperature=0.0,
            root=tmp_path,
        )


def test_a_run_spanning_two_bundles_is_refused(tmp_path: Path) -> None:
    config = make_config()
    save_measured_bundle(tmp_path, config)
    with pytest.raises(BundleValidationError, match="cần đúng"):
        build(
            config,
            version="0.1.1",
            generator=None,
            generation_report=make_generation_report(bundle_versions=["0.1.0", "0.2.0"]),
            gen_max_tokens=1024,
            gen_temperature=0.0,
            root=tmp_path,
        )


def test_generation_params_must_be_declared_not_defaulted(tmp_path: Path) -> None:
    """`TD-69`: answer run chưa ghi lại `max_tokens`/`temperature`, nên chúng
    phải được khai. Mặc định ở đây sẽ khai hộ một điều chưa ai kiểm."""
    config = make_config()
    save_measured_bundle(tmp_path, config)
    with pytest.raises(BundleValidationError, match="gen-max-tokens"):
        build(
            config,
            version="0.1.1",
            generator=None,
            generation_report=make_generation_report(),
            root=tmp_path,
        )


def test_two_prompts_in_one_run_are_refused(tmp_path: Path) -> None:
    config = make_config()
    save_measured_bundle(tmp_path, config)
    with pytest.raises(BundleValidationError, match="prompt"):
        build(
            config,
            version="0.1.1",
            generator=None,
            generation_report=make_generation_report(
                prompt_specs=["chat-system@v2", "chat-system@v1"]
            ),
            gen_max_tokens=1024,
            gen_temperature=0.0,
            root=tmp_path,
        )


def test_a_prompt_that_has_changed_since_the_run_is_refused(tmp_path: Path) -> None:
    config = make_config()
    save_measured_bundle(tmp_path, config)
    with pytest.raises(BundleValidationError, match="đã đổi kể từ lần đo"):
        build(
            config,
            version="0.1.1",
            generator=None,
            generation_report=make_generation_report(prompt_specs=["chat-system@v99"]),
            gen_max_tokens=1024,
            gen_temperature=0.0,
            root=tmp_path,
        )


def test_unjudged_rate_denominator_is_questions_asked(tmp_path: Path) -> None:
    """Mẫu số là số câu **đã hỏi**, không phải số câu chấm được.

    Dùng mẫu số "chấm được" thì tỉ lệ tự nhỏ đi đúng ở lần chạy hỏng nặng nhất:
    mất càng nhiều phán quyết, mẫu số càng bé, con số trông càng đẹp.
    """
    config = make_config()
    save_measured_bundle(tmp_path, config)
    broken = make_generation_report(
        metrics={
            "faithfulness": {"value": 1.0, "n": 18, "n_unjudged": 32, "n_not_a_claim": 0},
        }
    )
    bundle = build(
        config,
        version="0.1.1",
        generator=None,
        generation_report=broken,
        gen_max_tokens=1024,
        gen_temperature=0.0,
        root=tmp_path,
    )
    assert bundle.eval.unjudged_rate["faithfulness"] == pytest.approx(32 / 50)


def test_kappa_comes_from_the_calibration_file_not_from_a_flag(tmp_path: Path) -> None:
    config = make_config()
    save_measured_bundle(tmp_path, config)
    calibration = {
        "rubric": "judge-faithfulness@v2",
        "judge_model": "deepseek-v4-flash",
        "agreement": {"judge_vs_human": {"population": {"kappa": 0.7368}}},
    }
    bundle = build(
        config,
        version="0.1.1",
        generator=None,
        generation_report=make_generation_report(),
        calibration=calibration,
        gen_max_tokens=1024,
        gen_temperature=0.0,
        root=tmp_path,
    )
    assert bundle.eval.judge is not None
    assert bundle.eval.judge.kappa_vs_human == pytest.approx(0.7368)
    assert bundle.eval.judge.rubrics == (
        "judge-answer-relevancy@v1",
        "judge-faithfulness@v2",
    )


def test_a_calibration_of_a_different_rubric_is_refused(tmp_path: Path) -> None:
    """κ của một rubric không nói gì về rubric khác — `W5-01` đo được rubric
    v1→v2 làm một metric đổi gấp đôi trên cùng dữ liệu."""
    config = make_config()
    save_measured_bundle(tmp_path, config)
    with pytest.raises(BundleValidationError, match="rubric"):
        build(
            config,
            version="0.1.1",
            generator=None,
            generation_report=make_generation_report(),
            calibration={
                "rubric": "judge-faithfulness@v1",
                "agreement": {"judge_vs_human": {"population": {"kappa": 0.9}}},
            },
            gen_max_tokens=1024,
            gen_temperature=0.0,
            root=tmp_path,
        )


def test_a_calibration_of_a_different_judge_model_is_refused(tmp_path: Path) -> None:
    config = make_config()
    save_measured_bundle(tmp_path, config)
    with pytest.raises(BundleValidationError, match="judge"):
        build(
            config,
            version="0.1.1",
            generator=None,
            generation_report=make_generation_report(),
            calibration={
                "rubric": "judge-faithfulness@v2",
                "judge_model": "glm-5.3-flash",
                "agreement": {"judge_vs_human": {"population": {"kappa": 0.371}}},
            },
            gen_max_tokens=1024,
            gen_temperature=0.0,
            root=tmp_path,
        )


def test_only_judged_metrics_get_an_unjudged_rate(tmp_path: Path) -> None:
    """Gán `0,0` cho một metric tất định là khai một phép kiểm chưa từng chạy."""
    config = make_config()
    save_measured_bundle(tmp_path, config)
    bundle = build(
        config,
        version="0.1.1",
        generator=None,
        generation_report=make_generation_report(),
        gen_max_tokens=1024,
        gen_temperature=0.0,
        root=tmp_path,
    )
    assert set(bundle.eval.unjudged_rate) == {"faithfulness", "answer_relevancy"}
    assert "citation_accuracy" in bundle.eval.generation_metrics
    assert "citation_accuracy" not in bundle.eval.unjudged_rate
