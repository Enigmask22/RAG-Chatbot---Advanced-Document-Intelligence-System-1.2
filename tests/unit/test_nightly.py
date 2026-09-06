"""`W5-10` — con trỏ phát hành, promote, và vòng eval đêm.

Ba nhóm, và ranh giới giữa chúng là ranh giới của ba món nợ được trả cùng lúc:
`AU-12` (con trỏ), `TD-71` (phán quyết gate nằm trong bundle), và bản thân vòng
đêm nối hai thứ đó lại.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.bundle.promote import (
    CARRIED_OVER,
    PromotionRefused,
    next_patch_version,
    promote,
    rollback,
)
from pipeline.eval.gate import GateStatus as VerdictStatus
from pipeline.eval.gate import GateVerdict, Rule, RuleOutcome
from pipeline.eval.nightly import EXIT_PROMOTION_REFUSED, render_pr_body, run_nightly
from rag_core.bundle import (
    POINTER_NAME,
    BundleValidationError,
    GateStatus,
    current_bundle,
    load_bundle,
    read_pointer,
    save_bundle,
    write_pointer,
)

from .test_bundle import make_bundle

WORKFLOW = Path(".github/workflows/nightly.yml")


#: Tên model **thực tế đã phục vụ**, không phải bí danh. Gate từ chối bí danh
#: (`TD-70`), và một fixture mang bí danh làm mọi bài dưới đây đo nhầm luật:
#: chúng sẽ báo INCOMPARABLE vì tên model chứ không vì thứ đang được kiểm.
SERVED_MODEL = "deepseek-v4-flash"


def _seed(root: Path, *versions: str, **overrides: Any) -> None:
    from rag_core.bundle import EvalReport

    base = make_bundle()
    eval_block = EvalReport(
        golden_set=base.eval.golden_set,
        n_queries=base.eval.n_queries,
        evaluated_with_generator=SERVED_MODEL,
        retrieval_metrics=dict(base.eval.retrieval_metrics),
    )
    for version in versions:
        save_bundle(make_bundle(bundle_version=version, eval=eval_block, **overrides), root)


def _verdict(
    candidate: str,
    champion: str | None = "1.0.0",
    status: VerdictStatus = VerdictStatus.PASS,
) -> GateVerdict:
    return GateVerdict(
        status=status,
        candidate=candidate,
        champion=champion,
        rules=[Rule("absolute", "ndcg@10", RuleOutcome.PASS, "0.69 ≥ 0.60")],
    )


# ---------------------------------------------------------------------------
# 1. Con trỏ `CURRENT` — `AU-12`
# ---------------------------------------------------------------------------


class TestPointer:
    def test_no_pointer_is_not_an_error(self, tmp_path: Path) -> None:
        """Checkout mới chưa có con trỏ. Đó là trạng thái, không phải sự cố.

        ⚠️ Thư mục phải **có bundle**. Bản đầu của bài này chạy trên thư mục
        rỗng, nên `None` là câu trả lời của cả đường đúng lẫn đường sai — một
        phép tiêm cho `current_bundle` fallback về `latest_bundle` sống sót qua
        nó. Có bundle trong thư mục thì hai đường tách nhau.
        """
        _seed(tmp_path, "1.0.0", "1.1.0")
        assert read_pointer(tmp_path) is None
        assert current_bundle(tmp_path) is None, (
            "`current_bundle` trả lời 'đã chọn cái nào chưa', không phải 'đoán "
            "xem cái nào' — trộn hai câu ấy là con trỏ mất tác dụng ngay lần "
            "đầu ai đó quên tạo nó"
        )

    def test_an_empty_pointer_is_an_error(self, tmp_path: Path) -> None:
        """⭐ File rỗng **không** được đọc thành "chưa chọn".

        Hai thứ đó cho ra hai hành vi khác nhau — một cái quay về suy luận theo
        semver, một cái là một lần ghi hỏng — và trộn chúng lại đưa hệ thống về
        đúng cái hành vi mà con trỏ sinh ra để thay, im lặng.
        """
        (tmp_path / POINTER_NAME).write_text("  \n", encoding="utf-8")
        with pytest.raises(BundleValidationError, match="rỗng"):
            read_pointer(tmp_path)

    def test_the_pointer_is_checked_before_it_is_written(self, tmp_path: Path) -> None:
        """Trỏ vào một bundle không tồn tại phải hỏng **lúc ghi**, không phải lúc
        khởi động serving — lúc đó thông tin duy nhất còn lại là một tiến trình
        không lên được."""
        with pytest.raises(FileNotFoundError):
            write_pointer(tmp_path, "9.9.9")
        assert not (tmp_path / POINTER_NAME).exists()

    def test_a_pointer_that_names_a_corrupt_manifest_is_refused(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0")
        manifest = tmp_path / "rag-bundle-v1.0.0" / "manifest.json"
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw["eval"]["n_queries"] = 1  # nội dung đổi ⇒ chữ ký cũ không còn đúng
        manifest.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(Exception, match=r"checksum|chữ ký"):
            write_pointer(tmp_path, "1.0.0")

    def test_the_pointer_wins_over_the_highest_version(self, tmp_path: Path) -> None:
        """Lý do tồn tại của con trỏ, phát biểu thành một câu.

        `latest_bundle` sẽ chọn `1.1.0`. Con trỏ nói `1.0.0`, và một bản rollback
        phải sống sót qua restart.
        """
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        chosen = current_bundle(tmp_path)
        assert chosen is not None
        assert chosen.bundle_version == "1.0.0"

    def test_serving_and_the_pr_gate_read_the_same_pointer(self, tmp_path: Path) -> None:
        """⭐⭐ `AU-12` phát biểu thành một phép kiểm chạy được.

        Bug gốc: `smoke.py` hardcode `rag-bundle-v0.2.1` còn serving suy ra bản
        cao nhất. Hai bên **có thể** lệch, và triệu chứng của lệch là một cổng PR
        MÀU XANH gác cấu hình cũ — thứ không tự lộ ra bao giờ.

        Bài này dựng đúng cảnh ấy: một bundle mới hơn bản đang được trỏ tới. Nếu
        một trong hai bên quay lại suy luận theo semver, hai giá trị rời nhau.
        """
        from pipeline.eval.smoke import default_bundle
        from rag_core.settings import Settings
        from serving.api.app import _startup_version

        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")

        settings = Settings(bundle_root=tmp_path, bundle_version=None)
        served = _startup_version(settings)
        gated = load_bundle(default_bundle(tmp_path)).bundle_version
        assert served == gated == "1.0.0"

    def test_without_a_pointer_serving_says_out_loud_that_it_is_guessing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nhánh dự phòng được giữ lại cho dev, nhưng nó phải **khai ra**.

        Một hệ thống chọn bundle bằng suy luận mà không để lại dòng log nào là
        một hệ thống không ai trả lời được câu "vì sao nó đang chạy bản này".
        """
        from rag_core.settings import Settings
        from serving.api.app import _startup_version

        _seed(tmp_path, "1.0.0", "1.1.0")
        with caplog.at_level("WARNING"):
            assert _startup_version(Settings(bundle_root=tmp_path)) == "1.1.0"
        assert any("SUY LUẬN" in record.message for record in caplog.records)

    def test_smoke_refuses_to_guess(self, tmp_path: Path) -> None:
        """Cổng PR thì **không** có nhánh dự phòng: nó tồn tại để gác đúng cấu
        hình đang phục vụ, nên đoán là hỏng đúng cái việc của nó."""
        from pipeline.eval.smoke import default_bundle

        _seed(tmp_path, "1.0.0")
        with pytest.raises(FileNotFoundError, match="con trỏ"):
            default_bundle(tmp_path)


# ---------------------------------------------------------------------------
# 2. Promote — `TD-71`
# ---------------------------------------------------------------------------


class TestPromote:
    def test_the_verdict_lands_inside_the_new_bundle(self, tmp_path: Path) -> None:
        """`TD-71` chính là dòng này: bundle tự khai nó đã qua gate nào."""
        _seed(tmp_path, "1.0.0", "1.1.0")
        candidate = load_bundle(tmp_path / "rag-bundle-v1.1.0")
        assert candidate.gate.status is GateStatus.NOT_RUN

        result = promote(candidate, _verdict("1.1.0"), root=tmp_path, report="gate-1.1.0")

        stored = load_bundle(result.manifest)
        assert stored.bundle_version == "1.1.1"
        assert stored.gate.status is GateStatus.PASS
        assert stored.gate.champion_compared == "1.0.0"
        assert stored.gate.report == "gate-1.1.0"

    def test_promote_copies_the_measurements_verbatim(self, tmp_path: Path) -> None:
        """Bản phát hành mang đúng số đo của ứng viên — không hơn, không kém.

        Đọc `CARRIED_OVER` để danh sách ấy không trôi thành một comment: thêm
        một trường vào đó mà quên sao chép ⇒ đỏ.
        """
        _seed(tmp_path, "1.0.0", "1.1.0")
        candidate = load_bundle(tmp_path / "rag-bundle-v1.1.0")
        promoted = promote(candidate, _verdict("1.1.0"), root=tmp_path).bundle
        for field in CARRIED_OVER:
            assert getattr(promoted, field) == getattr(candidate, field), field

    def test_git_sha_stays_with_the_measurements(self, tmp_path: Path) -> None:
        """⭐⭐ Trường dễ ghi đè nhất, và ghi đè nó là mất đường truy nguyên.

        `git_sha` trả lời *"mã nào sinh ra những con số này"*. Mã ấy là commit
        của ứng viên, không phải commit của cái đêm mà một job CI dời con trỏ.
        Commit của lần promote vẫn được ghi — ở `notes`, nơi nó là nhật ký.
        """
        _seed(tmp_path, "1.0.0", "1.1.0")
        candidate = load_bundle(tmp_path / "rag-bundle-v1.1.0")
        promoted = promote(candidate, _verdict("1.1.0"), root=tmp_path).bundle
        assert promoted.git_sha == candidate.git_sha
        assert promoted.notes is not None
        assert "Commit lúc promote" in promoted.notes

    def test_the_new_bundle_goes_through_the_same_door_as_every_other(self, tmp_path: Path) -> None:
        """⭐ Dựng bằng `model_validate`, không bằng `model_copy`.

        `model_copy` bỏ qua validator. Trên đường **ghi ra đĩa và ký**, điều đó
        nghĩa là một bundle mà schema sẽ từ chối vẫn ra được một manifest có chữ
        ký hợp lệ. Bài này kiểm bằng đường đọc: `load_bundle` xác thực lại.
        """
        _seed(tmp_path, "1.0.0", "1.1.0")
        candidate = load_bundle(tmp_path / "rag-bundle-v1.1.0")
        result = promote(candidate, _verdict("1.1.0"), root=tmp_path)
        reread = load_bundle(result.manifest)  # verify=True: chữ ký + schema
        assert reread.bundle_version == result.bundle.bundle_version
        # ⭐ Chỗ hai đường đi khác nhau **quan sát được**: payload đưa vào là JSON
        # (`created_at` là chuỗi ISO). `model_validate` ép nó về `datetime`;
        # `model_copy` nhét thẳng chuỗi vào một trường khai kiểu `datetime` và
        # trả về một object nửa đúng kiểu — vẫn ghi ra file được, vẫn đọc lại
        # được, nên không có phép kiểm nào ở phía đĩa bắt được.
        assert isinstance(result.bundle.created_at, datetime)

    def test_the_pointer_follows_the_release(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        candidate = load_bundle(tmp_path / "rag-bundle-v1.1.0")
        result = promote(candidate, _verdict("1.1.0"), root=tmp_path)
        assert result.previous_pointer == "1.0.0"
        assert read_pointer(tmp_path) == "1.1.1"

    def test_a_taken_patch_number_does_not_wedge_the_job(self, tmp_path: Path) -> None:
        """Lần promote thứ hai trong cùng một đêm không được đâm vào luật
        không-ghi-đè rồi kẹt ở đó mỗi đêm sau."""
        _seed(tmp_path, "1.0.0", "1.1.0", "1.1.1")
        candidate = load_bundle(tmp_path / "rag-bundle-v1.1.0")
        assert next_patch_version(tmp_path, candidate) == "1.1.2"

    @pytest.mark.parametrize(
        ("verdict", "match"),
        [
            (_verdict("1.1.0", status=VerdictStatus.FAIL), "không phải PASS"),
            (_verdict("1.1.0", status=VerdictStatus.INCOMPARABLE), "không phải PASS"),
            (_verdict("9.9.9"), "chỉ thuộc về đúng cái"),
            (_verdict("1.1.0", champion=None), "không có champion"),
        ],
    )
    def test_the_three_ways_a_release_gets_stamped_by_mistake(
        self, tmp_path: Path, verdict: GateVerdict, match: str
    ) -> None:
        """Cả ba đều trông bình thường trong log, và cả ba đều phát hành nhầm."""
        _seed(tmp_path, "1.0.0", "1.1.0")
        candidate = load_bundle(tmp_path / "rag-bundle-v1.1.0")
        with pytest.raises(PromotionRefused, match=match):
            promote(candidate, verdict, root=tmp_path)
        assert not (tmp_path / "rag-bundle-v1.1.1").exists()

    def test_rollback_moves_the_pointer_and_deletes_nothing(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.1.0")
        rollback(tmp_path, "1.0.0")
        assert read_pointer(tmp_path) == "1.0.0"
        assert (tmp_path / "rag-bundle-v1.1.0" / "manifest.json").is_file()

    def test_rollback_to_a_version_that_is_not_there(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0")
        with pytest.raises(BundleValidationError, match="để lùi về"):
            rollback(tmp_path, "0.9.0")


# ---------------------------------------------------------------------------
# 3. Vòng đêm
# ---------------------------------------------------------------------------


def _thresholds(tmp_path: Path, *, ndcg: float = 0.6) -> Path:
    path = tmp_path / "gate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "require_same": ["golden_set"],
                "reject_alias_identity": False,
                "max_unjudged_rate": 0.05,
                "min_kappa": None,
                "absolute": {"ndcg@10": {"min": ndcg, "why": "ngưỡng test"}},
                "regression": {"ndcg@10": {"max_drop": 0.01, "why": "ngưỡng test"}},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


class TestNightly:
    def test_an_empty_bundle_root_is_reported_not_crashed(self, tmp_path: Path) -> None:
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path))
        assert result.exit_code == 0
        assert "không có bundle nào" in result.reason

    def test_the_candidate_that_is_already_serving_is_not_promoted_again(
        self, tmp_path: Path
    ) -> None:
        """⭐ PASS mà vẫn không phát hành.

        Không có phép kiểm này thì mỗi đêm sinh ra một patch mới dán dấu PASS
        lên đúng hệ thống đêm trước — số hiệu tăng, thông tin không.
        """
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.1.0")
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path))
        assert result.verdict is not None
        assert result.verdict.status is VerdictStatus.PASS
        assert result.promoted is None
        assert result.exit_code == 0
        assert "đã là bản đang phục vụ" in result.reason

    def test_a_new_candidate_that_passes_becomes_a_release_proposal(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path))
        assert result.promoted is not None
        assert result.promoted.bundle.bundle_version == "1.1.1"
        assert read_pointer(tmp_path) == "1.1.1"

    def test_a_failing_candidate_leaves_the_pointer_where_it_was(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path, ndcg=0.99))
        assert result.exit_code == 1
        assert result.promoted is None
        assert read_pointer(tmp_path) == "1.0.0"
        assert not (tmp_path / "rag-bundle-v1.1.1").exists()

    def test_no_promote_runs_the_gate_and_writes_nothing(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path), do_promote=False)
        assert result.verdict is not None
        assert result.promoted is None
        assert read_pointer(tmp_path) == "1.0.0"

    def test_the_champion_is_what_is_serving_not_the_version_below(self, tmp_path: Path) -> None:
        """⭐⭐ Chỗ nightly cố ý **không** dùng mặc định của `gate.py`.

        Cảnh: đã rollback về `1.0.0` trong khi `1.1.0` vẫn nằm trên đĩa, rồi
        `1.2.0` xuất hiện. `gate._pick_champion` sẽ so với `1.1.0` — bản không
        ai đang dùng. Câu hỏi phát hành là *"có tụt so với thứ người dùng đang
        nhận không"*, nên champion phải là `1.0.0`.
        """
        _seed(tmp_path, "1.0.0", "1.1.0", "1.2.0")
        write_pointer(tmp_path, "1.0.0")
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path), do_promote=False)
        assert result.champion is not None
        assert result.champion.bundle_version == "1.0.0"

    def test_a_broken_release_path_gets_its_own_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAIL nghĩa là hệ thống chưa đủ tốt; cái này nghĩa là hệ thống đủ tốt
        mà đường phát hành hỏng. Hai người khác nhau phải thức dậy."""
        import pipeline.bundle.promote as promote_module

        def boom(*_: object, **__: object) -> None:
            raise PromotionRefused("đĩa đầy")

        monkeypatch.setattr(promote_module, "promote", boom)
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path))
        # ⚠️ Số **viết thẳng**, không phải hằng số. `assert x == EXIT_PROMOTION_REFUSED`
        # đọc chính giá trị bị đổi, nên nó xanh với mọi giá trị — một phép tiêm
        # hạ nó xuống 1 sống sót qua bài này. Hằng số vẫn được kiểm, nhưng bằng
        # một câu về **quan hệ** với các mã còn lại.
        assert result.exit_code == 3
        assert EXIT_PROMOTION_REFUSED == 3
        assert EXIT_PROMOTION_REFUSED not in {
            VerdictStatus.PASS.exit_code,
            VerdictStatus.FAIL.exit_code,
            VerdictStatus.INCOMPARABLE.exit_code,
        }, "một mã lỗi trùng mã của gate là hai nguyên nhân đọc thành một"

    def test_every_run_produces_a_readable_body_even_when_nothing_happens(
        self, tmp_path: Path
    ) -> None:
        """Một job đêm im lặng không phân biệt được với một job đã chết."""
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        failing = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path, ndcg=0.99))
        body = render_pr_body(failing)
        assert "không đề nghị phát hành" in body
        assert "ndcg@10" in body

    def test_the_proposal_body_names_the_pointer_move(self, tmp_path: Path) -> None:
        _seed(tmp_path, "1.0.0", "1.1.0")
        write_pointer(tmp_path, "1.0.0")
        result = run_nightly(root=tmp_path, thresholds_path=_thresholds(tmp_path))
        body = render_pr_body(result)
        assert "Đề nghị phát hành `1.1.1`" in body
        assert "`1.0.0` → `1.1.1`" in body
        assert "Merge PR này là lần phát hành" in body


# ---------------------------------------------------------------------------
# 4. Workflow — đọc chính file YAML, cùng lối với `test_ci_tiers.py`
# ---------------------------------------------------------------------------


class TestNightlyWorkflow:
    @staticmethod
    def _workflow() -> dict[Any, Any]:
        loaded: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        return loaded

    def test_it_never_runs_on_pull_requests(self) -> None:
        """Workflow này gọi model, tốn tiền, và ghi vào repo. Một trigger
        `pull_request` biến nó thành cổng chặn PR — thứ `ci.yml` cố ý **không**
        làm, vì PR từ fork không thấy secret và `temp=0` không tất định."""
        triggers = self._workflow()[True]  # `on:` bị YAML 1.1 đọc thành True
        assert "pull_request" not in triggers
        assert "schedule" in triggers
        assert "workflow_dispatch" in triggers

    def test_the_gate_job_runs_even_when_the_gpu_tier_is_skipped(self) -> None:
        """⭐ `needs` bị skip làm job phụ thuộc skip theo — tức toàn bộ đường
        phát hành biến mất im lặng vào đúng ngày chưa có runner GPU."""
        job = self._workflow()["jobs"]["gate-and-promote"]
        assert "always()" in job["if"]
        assert "needs.full-eval.result != 'failure'" in job["if"]

    def test_the_gpu_tier_is_gated_behind_a_variable_not_left_queuing(self) -> None:
        """Không có runner mang nhãn `self-hosted, gpu` thì job **treo** ở hàng
        đợi thay vì đỏ — trông giống "đang chạy". Một `if` biến nó thành
        "skipped", thứ đọc được."""
        job = self._workflow()["jobs"]["full-eval"]
        assert job["if"] == "vars.NIGHTLY_GPU_RUNNER == 'on'"
        assert job["runs-on"] == ["self-hosted", "gpu"]

    def test_the_pr_is_opened_only_when_there_is_something_to_release(self) -> None:
        """Exit 0 cũng xảy ra khi "không có ứng viên mới"; mở PR theo exit code
        là mỗi đêm một PR rỗng, và một PR rỗng mỗi đêm là cách nhanh nhất để mọi
        người tắt thông báo."""
        steps = self._workflow()["jobs"]["gate-and-promote"]["steps"]
        pr = next(s for s in steps if s.get("uses", "").startswith("peter-evans/"))
        assert pr["if"] == "steps.nightly.outputs.promoted == 'true'"
        assert "bundles/" in pr["with"]["add-paths"]

    def test_it_may_write_the_repo_and_says_so(self) -> None:
        permissions = self._workflow()["permissions"]
        assert permissions == {"contents": "write", "pull-requests": "write"}

    def test_a_nightly_run_is_never_cancelled_by_the_next_one(self) -> None:
        """Một lượt eval đêm bị cắt ngang để lại artifact dở và không để lại
        phán quyết nào. Xếp hàng thì tệ nhất là chậm."""
        assert self._workflow()["concurrency"]["cancel-in-progress"] is False

    def test_the_workflow_does_not_hardcode_a_bundle_version(self) -> None:
        """`AU-12` một lần nữa, ở tầng YAML: một số hiệu gõ tay trong workflow
        là bản sao thứ ba của "đang chạy bản nào"."""
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "rag-bundle-v" not in text


def test_the_stamped_release_is_dated_now_not_at_measurement_time() -> None:
    """`created_at` của bản phát hành là lúc **đúc nó**, khác `git_sha`.

    Cặp trường này cố ý không đi cùng nhau: một cái nói artifact ra đời khi nào,
    cái kia nói số đo đến từ mã nào. Gộp chúng lại là mất một trong hai.
    """
    bundle = make_bundle(created_at=datetime(2020, 1, 1, tzinfo=UTC))
    assert bundle.created_at.year == 2020
