"""Bộ dò từ chối bằng từ khoá — dùng chung giữa hai plane. `W5-11`.

## Vì sao nó nằm ở `rag_core` chứ không ở `serving`

Nó ra đời ở `serving/core/metrics.py` (`W5-07`) và ở đó là đúng chỗ, cho tới khi
`W5-11` cần **hiệu chỉnh lại nó theo từng model sinh** (`TD-77`) — một việc của
Pipeline Plane. Import ngược từ `pipeline` sang `serving` làm
`test_pipeline_does_not_import_serving` đỏ, và bài test ấy đúng: pipeline phải
chạy được độc lập trên máy GPU thuê, nơi không có serving stack.

Cách sai là chép danh sách từ khoá sang chỗ thứ hai. Bảng Grafana và bảng hiệu
chỉnh khi đó đo hai bộ dò khác nhau, và con số F1 công bố thôi mô tả cái đang
chạy — đúng họ lỗi với `AU-12` (một bản sao thứ hai của một sự thật).

Nên nó chuyển xuống `rag_core`, nơi cả hai plane cùng nhìn thấy đúng một bản.
"""

from __future__ import annotations

__all__ = ["REFUSAL_MARKERS", "looks_like_refusal"]


REFUSAL_MARKERS = (
    "không tìm thấy thông tin",
    "không có thông tin",
    # ⚠️ Cần dòng riêng: `"không đủ thông tin"` **không** là chuỗi con của
    # `"không có đủ thông tin"` — chữ "có" chen vào giữa. Bốn ca bỏ sót ở nửa
    # hiệu chỉnh đều là dạng này, và nhìn hai chuỗi cạnh nhau thì rất dễ tin là
    # cái sau đã bao cái trước.
    "không có đủ thông tin",
    "không đủ thông tin",
    "không đủ dữ liệu",
    "các nguồn không",
    "tài liệu không",
    "ngữ cảnh không",
    "i could not find",
    "i don't have enough",
    "do not contain",
    "does not contain",
    "insufficient information",
    # Dạng tiếng Anh phổ biến nhất trong 242 câu trả lời thật, và bản đầu bỏ
    # sót toàn bộ: model mở câu bằng "Based on the provided context, there is
    # no information about …".
    "there is no information",
    "there is no specific",
    "no information about",
)


def looks_like_refusal(text: str) -> bool:
    """Ước lượng bằng từ khoá — **không** phải phép đo từ chối của `W5-02`.

    ## ⭐⭐ Bảng trực tuyến không đo được thứ mà eval đo, và phải nói ra

    `W5-02` đo từ chối bằng một **nhãn của judge** (`REFUSAL` trong rubric
    `judge-answer-relevancy`), và docstring của `score_refusal` nói thẳng vì
    sao không dùng từ khoá: *"một danh sách từ khoá sẽ bắt được đúng những cách
    nói tôi nghĩ ra được"*. Câu ấy vẫn đúng nguyên si ở đây.

    Nhưng gọi judge cho **mỗi** request là không làm được: nó thêm một lời gọi
    model vào đường có người đang đợi, và nó nhân đôi hoá đơn. Nên lựa chọn
    thật sự chỉ có ba: bỏ hẳn ô này khỏi bảng, gọi judge trực tuyến, hoặc dùng
    một ước lượng và **dán nhãn nó là ước lượng**.

    Lối thứ ba đúng vì công việc của một bảng khác công việc của một phép eval:

    > Một ước lượng **chệch nhưng ổn định** thì vô dụng để nói *mức*, và hoàn
    > toàn dùng được để nói *đạo hàm*.

    Không ai nên đọc `rag_refusals_suspected / rag_chat_turns` rồi bảo "hệ
    thống từ chối 12% câu hỏi" — con số ấy thuộc về `W5-02` và nó đo trên
    `golden_v1` với judge đã hiệu chỉnh (`W5-04`). Nhưng cùng tỉ lệ ấy nhảy từ
    12% lên 40% trong một giờ là một tín hiệu thật, và nó là thứ duy nhất trên
    bảng nhìn thấy được một index hỏng hay một bundle nạp nhầm.

    Tên metric mang chữ `suspected`, và `HELP` của nó nói ra điều này — vì
    người đọc bảng lúc 3 giờ sáng không đọc docstring.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
