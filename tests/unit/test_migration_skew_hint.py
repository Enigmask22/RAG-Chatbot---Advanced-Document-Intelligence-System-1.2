"""`/ready` phải nói ĐÚNG CHIỀU lệch migration, vì hai chiều cần hai hành động ngược nhau.

Bản đầu của thông điệp chỉ mô tả một chiều — *"deploy đã lên trước khi migration
chạy"* — và khuyên `alembic upgrade head`. Điều kiện kích hoạt nó lại là
`found != want`, tức **cả hai** chiều. Gặp chiều kia (DB đi trước mã, vì image
cũ), lời khuyên ấy gửi người đọc đi hướng sai: migration đã chạy rồi.

⚠️ Tìm ra vì gặp thật (07/09/2026): DB ở `0005_message_user_link`, container cần
`0003_message_query_plan` sau một `docker compose up` thiếu `--build`, và
`/ready` bảo chạy migration.

Test ở đây là **unit**: `_skew_hint` chỉ đọc thư mục `alembic/`, không chạm DB.
Đặt nó ở tầng integration sẽ khiến nó chỉ chạy khi có Docker — tức không chạy ở
đúng nơi nó rẻ nhất.
"""

from __future__ import annotations

import pytest

from serving.db.engine import _skew_hint, expected_revision

HEAD = expected_revision()


def test_db_behind_code_says_run_the_migration() -> None:
    """Tổ tiên của head ⇒ DB đi sau ⇒ `alembic upgrade head` là câu trả lời đúng.

    ⚠️ Dùng `0001_initial`, không phải `0001_initial_schema`: **id revision
    không phải tên file**. File là `0001_initial_schema.py` còn id bên trong
    nó là `0001_initial`, và bản đầu của test này lấy tên file nên rơi thẳng
    vào nhánh "revision lạ" — tức nó xanh/đỏ vì một lý do khác hẳn lý do nó
    tin. Cùng cái bẫy mà `test_the_expected_revision_comes_from_the_migration_folder`
    đang đi qua nhờ may: ở `0005` thì stem trùng id.
    """
    hint = _skew_hint("0001_initial", HEAD)

    assert "đi SAU" in hint
    assert "alembic upgrade head" in hint


def test_db_ahead_of_code_does_not_tell_you_to_run_the_migration() -> None:
    """⭐⭐ Đây là ca đã gặp thật, và là ca bản cũ trả lời sai.

    Lời khuyên sai ở đây không vô hại: `alembic upgrade head` trên một DB đã đi
    trước là một lệnh **không làm gì**, nên người chạy nó kết luận phép kiểm
    hỏng chứ không kết luận image cũ.
    """
    hint = _skew_hint(HEAD, "0003_message_query_plan")

    assert "đi TRƯỚC" in hint
    assert "KHÔNG sửa được" in hint
    # Câu chốt: không được khuyên chạy migration ở cảnh này.
    assert "Chạy `alembic upgrade head`" not in hint


def test_an_unknown_revision_is_a_third_case_not_a_guess() -> None:
    """⚠️ `iterate_revisions(want, "base")` KHÔNG nổ với revision lạ — nó chỉ duyệt
    tổ tiên của `want`. Nên gộp "lạ" vào "đi trước" là đoán bừa dưới lớp áo của
    một phép kiểm, và bản nháp đầu của `_skew_hint` đã làm đúng thế.
    """
    hint = _skew_hint("mot_revision_khong_ton_tai", HEAD)

    assert "không biết revision" in hint
    assert "đi SAU" not in hint
    assert "đi TRƯỚC" not in hint


@pytest.mark.parametrize("found", ["0001_initial", HEAD, "khong_ton_tai"])
def test_every_branch_names_an_action(found: str) -> None:
    """Một thông điệp chẩn đoán không nói phải làm gì thì chỉ là một thông báo lỗi."""
    hint = _skew_hint(found, "0003_message_query_plan" if found == HEAD else HEAD)

    assert any(word in hint for word in ("Chạy", "Deploy", "Đối chiếu")), hint
