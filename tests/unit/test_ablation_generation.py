"""`W5-11` — so hai model sinh có kiểm định, và hiệu chỉnh bộ dò từ chối.

Hai module, một chủ đề: **một mức chênh không có khoảng tin cậy thì không đọc
được**, và một con số hiệu chỉnh trên một model thì không mang sang model khác
được. Cả hai bài học đều đã có giá ở dự án này (`W2-09`, `TD-77`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.eval.ablation_generation import (
    HIGHER_IS_BETTER,
    Branch,
    compare_branches,
    format_table,
    load_branch,
)
from pipeline.eval.generation_metrics import Aggregate
from pipeline.eval.refusal_calibration import calibrate


def _branch(
    name: str, scores: dict[str, dict[str, float]], judge: str = "deepseek-v4-flash"
) -> Branch:
    return Branch(name=name, models=("m",), judge_model=judge, scores=scores)


# ---------------------------------------------------------------------------
# 1. Ghép cặp — hàng rào mượn từ `compare.py`
# ---------------------------------------------------------------------------


class TestPairing:
    def test_two_judges_cannot_be_compared(self) -> None:
        """⭐⭐ `TD-66` đo được: đổi model judge làm faithfulness dịch 7,5 điểm —
        lớn hơn mọi cải thiện của cả `W2` cộng lại.

        Hiệu số giữa hai nhánh chấm bởi hai judge mang lẫn cả phần dịch ấy, và
        không có cách nào tách ra **sau khi** đã trộn. Nên đây là phép từ chối
        duy nhất trong module này ném lỗi thay vì cảnh báo.
        """
        a = _branch("a", {"faithfulness": {"q1": 1.0}}, judge="deepseek-v4-flash")
        b = _branch("b", {"faithfulness": {"q1": 0.0}}, judge="glm-5.3-flash")
        with pytest.raises(ValueError, match="hai judge khác nhau"):
            compare_branches(a, b)

    def test_only_the_shared_queries_are_compared_and_the_gap_is_named(self) -> None:
        """So 242 câu với 200 câu rồi kết luận là một dạng tự chọn mẫu. Nhưng
        lệch vài câu vẫn so được — miễn là con số ấy được in ra cạnh bảng."""
        a = _branch("a", {"m": {"q1": 1.0, "q2": 1.0, "q3": 1.0}})
        b = _branch("b", {"m": {"q1": 0.0, "q2": 0.0}})
        rows, warnings = compare_branches(a, b, iterations=200)
        assert rows[0].n == 2
        assert any("so trên 2 câu giao nhau" in w for w in warnings)

    def test_a_metric_present_on_only_one_side_is_reported_not_dropped(self) -> None:
        a = _branch("a", {"m": {"q1": 1.0}, "chỉ_a": {"q1": 1.0}})
        b = _branch("b", {"m": {"q1": 1.0}})
        _, warnings = compare_branches(a, b, iterations=200)
        assert any("chỉ_a" in w for w in warnings)

    def test_no_shared_query_is_a_warning_not_a_crash(self) -> None:
        a = _branch("a", {"m": {"q1": 1.0}})
        b = _branch("b", {"m": {"q9": 1.0}})
        rows, warnings = compare_branches(a, b, iterations=200)
        assert rows == []
        assert any("không có truy vấn chung" in w for w in warnings)


# ---------------------------------------------------------------------------
# 2. Đọc kết quả — hai chiều tốt/xấu, và "chứa 0" nghĩa là gì
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_an_identical_pair_is_not_called_a_winner(self) -> None:
        """CI chứa 0 ⇒ `—`. Không đọc thành "hai hệ thống như nhau": nó nghĩa là
        **tập này không phân biệt được** hai nhánh."""
        scores = {"faithfulness": {f"q{i}": 1.0 for i in range(30)}}
        rows, _ = compare_branches(_branch("a", scores), _branch("b", scores), iterations=500)
        assert rows[0].significant is False
        assert rows[0].better == "—"

    def test_a_clear_gap_is_called(self) -> None:
        a = _branch("a", {"faithfulness": {f"q{i}": 0.0 for i in range(40)}})
        b = _branch("b", {"faithfulness": {f"q{i}": 1.0 for i in range(40)}})
        rows, _ = compare_branches(a, b, iterations=500)
        assert rows[0].significant is True
        assert rows[0].better == "ứng viên"

    def test_a_lower_is_better_metric_is_not_read_upside_down(self) -> None:
        """⭐ `misattribution` tăng là **xấu đi**. Một bảng không phân biệt hai
        chiều sẽ in mũi tên xanh cho một lần hệ thống tệ đi."""
        assert "misattribution" not in HIGHER_IS_BETTER
        a = _branch("a", {"misattribution": {f"q{i}": 0.0 for i in range(40)}})
        b = _branch("b", {"misattribution": {f"q{i}": 1.0 for i in range(40)}})
        rows, _ = compare_branches(a, b, iterations=500)
        assert rows[0].significant is True
        assert rows[0].better == "mốc"

    def test_the_table_says_which_judge_produced_it(self) -> None:
        """Một bảng ablation không ghi judge là một bảng không tái lập được —
        và `TD-66` là lý do."""
        a = _branch("a", {"m": {"q1": 1.0}})
        b = _branch("b", {"m": {"q1": 1.0}})
        rows, warnings = compare_branches(a, b, iterations=200)
        table = format_table(a, b, rows, warnings)
        assert "deepseek-v4-flash" in table
        assert "macro" in table


def test_the_sidecar_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "x-per-query.json"
    path.write_text(
        json.dumps(
            {"run": "r", "models": ["m1"], "judge_model": "j", "faithfulness": {"q1": 1.0}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    branch = load_branch(path)
    assert branch.name == "r"
    assert branch.judge_model == "j"
    assert branch.scores == {"faithfulness": {"q1": 1.0}}


# ---------------------------------------------------------------------------
# 3. `Aggregate.paired_values` — macro chứ không micro
# ---------------------------------------------------------------------------


def test_a_query_with_many_claims_does_not_get_many_votes() -> None:
    """⭐⭐ Đơn vị lấy mẫu lại độc lập là **truy vấn**, không phải mệnh đề.

    `q1` đóng góp 3 mệnh đề (2 đúng), `q2` đóng góp 1 (sai). Micro-average là
    2/4 = 0,50; macro-average là (2/3 + 0)/2 = 0,3333. Bootstrap phải chạy trên
    macro, nếu không một truy vấn 12 mệnh đề nặng gấp 12 lần một truy vấn 1
    mệnh đề và khoảng tin cậy hẹp lại một cách giả tạo.
    """
    agg = Aggregate("faithfulness")
    for hit in (True, True, False):
        agg.add(hit, "factoid", "q1")
    agg.add(False, "factoid", "q2")

    assert agg.value == pytest.approx(0.5)  # micro
    paired = agg.paired_values()
    assert paired["q1"] == pytest.approx(2 / 3)
    assert paired["q2"] == pytest.approx(0.0)


def test_a_value_added_without_a_query_id_stays_out_of_the_paired_view() -> None:
    """Người gọi cũ không phải sửa, nhưng cũng không được lặng lẽ ghép cặp sai."""
    agg = Aggregate("m")
    agg.add(True, "factoid")
    assert agg.value == pytest.approx(1.0)
    assert agg.paired_values() == {}


# ---------------------------------------------------------------------------
# 4. `TD-77` — bộ dò từ chối
# ---------------------------------------------------------------------------


class TestRefusalCalibration:
    def test_the_arithmetic(self) -> None:
        """⚠️ Precision và recall phải **khác nhau** ở đây.

        Bản đầu dựng 1 TP / 1 FP / 1 FN, tức P = R = 0,5 = F1 — và một phép tiêm
        đổi `f1` thành `precision` sống sót qua nó. Một bài test chọn ví dụ cân
        đối là một bài test không phân biệt được ba công thức.
        """
        refusal_text = "Tôi không tìm thấy thông tin này trong tài liệu."
        answers = [
            ("q1", refusal_text),  # TP
            ("q2", refusal_text),  # TP
            ("q3", refusal_text),  # FP
            ("q4", "Ngân sách là 5 tỉ."),  # TN
            ("q5", "Ngân sách là 5 tỉ."),  # FN
            ("q6", "Ngân sách là 5 tỉ."),  # FN
            ("q7", "Ngân sách là 5 tỉ."),  # FN
        ]
        labels: dict[str, str | None] = {
            "q1": "REFUSAL",
            "q2": "REFUSAL",
            "q3": "RELEVANT",
            "q4": "RELEVANT",
            "q5": "REFUSAL",
            "q6": "REFUSAL",
            "q7": "REFUSAL",
        }
        score = calibrate(answers, labels)
        assert (score.tp, score.fp, score.fn, score.n) == (2, 1, 3, 7)
        assert score.precision == pytest.approx(2 / 3)
        assert score.recall == pytest.approx(0.4)
        # 2·(2/3)·0,4 / (2/3 + 0,4) = 0,5 — khác cả precision lẫn recall.
        assert score.f1 == pytest.approx(0.5)
        assert score.f1 != pytest.approx(score.precision)
        assert score.f1 != pytest.approx(score.recall)

    def test_an_unreadable_verdict_is_dropped_from_both_sides(self) -> None:
        """Cùng quy ước với `score_relevancy`: tính một phán quyết không đọc
        được thành "không từ chối" là ghi **lỗi của judge** thành một thuộc tính
        của bộ dò."""
        answers = [("q1", "Ngân sách là 5 tỉ."), ("q2", "Không rõ.")]
        score = calibrate(answers, {"q1": "RELEVANT", "q2": None})
        assert score.n == 1

    def test_the_sign_of_the_bias_is_the_part_that_matters(self) -> None:
        """⭐ Âm = bảng báo **thiếu** số lần từ chối, tức nó làm hệ thống trông
        tốt hơn thực tế — hướng chệch tệ nhất, cùng họ với `TD-55`."""
        hit = "Tôi không tìm thấy thông tin này."  # khớp `_REFUSAL_MARKERS`
        under = calibrate([("q1", hit), ("q2", hit)], {"q1": "REFUSAL", "q2": "REFUSAL"})
        assert under.bias == pytest.approx(0.0)

        missed = calibrate(
            [("q1", "Đáp án."), ("q2", "Đáp án.")], {"q1": "REFUSAL", "q2": "REFUSAL"}
        )
        assert missed.bias < 0
        assert missed.missed_examples  # ca bỏ sót phải in ra được, không chỉ đếm

    def test_no_refusal_at_all_does_not_divide_by_zero(self) -> None:
        score = calibrate([("q1", "Đáp án.")], {"q1": "RELEVANT"})
        assert score.f1 == 0.0
        assert score.bias == 0.0


# ---------------------------------------------------------------------------
# 5. Hai chỗ nối — danh tính bộ sinh và provider của judge
# ---------------------------------------------------------------------------


class TestGeneratorIdentity:
    def test_the_provider_is_part_of_the_identity_not_just_the_model(self) -> None:
        """Hai nhà cung cấp có thể phục vụ cùng một slug qua hai endpoint khác
        nhau, và khi đó câu trả lời vẫn là của hai hệ thống khác nhau."""
        from rag_core.settings import Settings
        from serving.api.app import primary_generator

        assert (
            primary_generator(Settings(chat_provider="deepseek", chat_model="deepseek-v4-flash"))
            == "deepseek:deepseek-v4-flash"
        )
        assert primary_generator(Settings(chat_provider="glm")) == "glm:glm-5.3-flash"

    def test_no_provider_means_no_identity_which_means_no_cache(self) -> None:
        """`""` là tín hiệu tắt cache của `ChatService`, không phải một ô dùng
        chung — xem `ChatService.generator`."""
        from rag_core.settings import Settings
        from serving.api.app import primary_generator

        assert primary_generator(Settings(chat_provider="none")) == ""


class TestJudgeProvider:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [("deepseek-v4-flash", "deepseek"), ("glm-5.3-flash", "glm")],
    )
    def test_the_family_decides_the_endpoint(self, model: str, expected: str) -> None:
        """⭐ Rẽ theo họ **suy ra từ slug**, không theo một tham số khai riêng.

        `generation_metrics` trước `W5-11` hardcode DeepSeek, nên chép đoạn rẽ
        nhánh của `calibration.py` sang là mở đường cho hai bản lệch nhau — và
        cách chúng lệch là im lặng: `reasoning_effort` gửi sang DeepSeek được
        **nhận rồi bỏ qua** (đo ở `W3-04`), phán quyết vẫn về, vẫn vào cache,
        chỉ là được sinh dưới một điều kiện khác lời khai.
        """
        from pipeline.eval.judge import DEEPSEEK_BASE_URL, JudgeConfig
        from rag_core.llm import GLM_BASE_URL

        base = GLM_BASE_URL if model.startswith("glm-") else DEEPSEEK_BASE_URL
        assert JudgeConfig(model=model, base_url=base).family == expected
