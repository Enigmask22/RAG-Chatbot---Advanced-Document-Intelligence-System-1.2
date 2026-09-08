"""Phiên bản thư viện **đang thật sự chạy** — `TD-62`.

## Sự cố sinh ra nợ này

Container chạy `transformers 5.16.1 / sentence-transformers 6.0.1 / torch
2.14.0` trong khi `uv.lock` ghim `5.15.0 / 5.7.0 / 2.13.0`, và **cross-encoder
không phục vụ được**. Trong suốt lúc ấy `GET /admin/bundle` trả
`runtime_drift: null` — vì phép so danh tính của `W4-02` đối chiếu
`retriever_name`, tức `(model, device, dtype)`, và bộ ba ấy **mù hoàn toàn** với
phiên bản thư viện.

`uv sync --locked` + 2 test đã chặn việc tái diễn âm thầm. Cái còn lại là điều
`TD-62` ghi: **bản thân artifact vẫn khai thiếu**.

## ⚠️ Đây là **quan sát**, không phải **cưỡng chế** — và nói ra vì sao

Đưa ba gói này vào *vân tay* (từ chối phục vụ khi lệch) nghe mạnh hơn, nhưng nó
đòi hai thứ mà cái giá không đáng:

1. **Manifest phải ghi phiên bản lúc eval**, tức thêm trường vào `RagBundle` —
   và `TD-36` đã trả giá đúng một lần cho bài học *"thêm một trường có mặc định
   vào bundle schema làm vỡ chữ ký của mọi bundle cũ"*.
2. **Một lần bump patch của `transformers` sẽ từ chối phục vụ.** Hàng rào chặn
   nhiều hơn thứ nó cần chặn sẽ bị tắt, và lúc ấy nó không chặn gì cả.

Nên: khai ra, ở chỗ người vận hành nhìn, và ở log lúc khởi động. Cùng lý lẽ với
`runtime_drift` — *một cửa thoát im lặng mới là cửa thoát nguy hiểm* — chỉ khác
là ở đây không có cửa thoát nào để đóng, chỉ có một chỗ mù để soi sáng.

⚠️ Residual, ghi thẳng: điều này **không tự bắt** được sự cố trên. Nó làm người
đang gỡ lỗi thấy con số trong một lệnh `curl` thay vì phải `docker exec` vào
container. Phần tự bắt nằm ở `uv sync --locked` của lượt trước.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["MODEL_PACKAGES", "library_versions"]

#: Ba gói mà **phiên bản của chúng đổi được hành vi model**, không phải mọi gói
#: trong môi trường. `torch` (kernel, dtype), `transformers` (kiến trúc, tokenizer)
#: và `sentence-transformers` (đường nạp, pooling) — đúng bộ ba đã lệch trong sự
#: cố của `TD-62`.
#:
#: ⚠️ Cố ý **không** liệt kê cả `pip freeze`: một danh sách 200 dòng ở
#: `/admin/bundle` là một danh sách không ai đọc, và thứ cần đọc sẽ chìm trong đó.
MODEL_PACKAGES = ("torch", "transformers", "sentence-transformers")


def library_versions(packages: tuple[str, ...] = MODEL_PACKAGES) -> dict[str, str | None]:
    """Phiên bản đã cài, hoặc `None` nếu gói không có mặt.

    ⭐ `None` là một câu trả lời **có nghĩa**, không phải một lỗi cần nuốt: image
    serving chỉ dùng cross-encoder khi bundle khai reranker, nên một triển khai
    dense-only hợp lệ **sẽ** thiếu `sentence-transformers`. Ném ở đây biến một
    cấu hình đúng thành một `/admin/bundle` 500.
    """
    ra: dict[str, str | None] = {}
    for name in packages:
        try:
            ra[name] = version(name)
        except PackageNotFoundError:
            ra[name] = None
    return ra
