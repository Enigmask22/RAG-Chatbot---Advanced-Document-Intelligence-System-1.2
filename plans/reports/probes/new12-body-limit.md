# `NEW-12` — probe trần thân request

`POST` 200 MB vào một app cùng ngăn xếp với API thật (uvicorn + starlette +
pydantic), RSS lấy mẫu 5 ms/lần trên **cả cây tiến trình**.

| cấu hình | status client nhận | RSS nền | RSS đỉnh |
|---|---|---:|---:|
| `/chat` — Pydantic `max_length=8000`, không trần | 422 | 54,9 MB | **852,3 MB** |
| `/raw` — không Pydantic, chunked, không trần | 200 | 54,8 MB | 421,4 MB |
| `/raw` — không Pydantic, `content-length`, không trần | 200 | 54,5 MB | 406,3 MB |
| `/raw` + `BodyLimitMiddleware`, chunked | **413** | 54,7 MB | **55,0 MB** |
| `/raw` + `BodyLimitMiddleware`, `content-length` | **413** | 54,7 MB | **54,8 MB** |

## Hai lỗi của chính phép đo, ghi lại vì cả hai đều cho ra số "hợp lý"

1. **Chỉ lấy mẫu trước và sau** ⇒ 3,8 MB ở mọi cấu hình. CPython trả khối lớn
   về HĐH ngay khi giải phóng, nên đỉnh đã biến mất trước lần đọc thứ hai.
2. **Đo nhầm tiến trình.** `python.exe` trong venv do `uv` dựng là một
   **trampoline**: nó sinh tiến trình con rồi ngồi chờ. Sau khi sửa (1) mà chưa
   sửa (2), con số vẫn là 3,8 MB — ổn định, hợp lý về hình thức, và hoàn toàn
   vô nghĩa. Phải cộng `p.children(recursive=True)`.

⚠️ Một phép đo cho ra cùng một con số ở mọi cấu hình là **tín hiệu hỏng**, không
phải kết luận "không khác biệt".

## Thân request hợp lệ lớn nhất — đếm, không ước lượng

| | byte |
|---|---:|
| `/chat` message ASCII đầy | 8.083 |
| `/chat` message tiếng Việt đầy (3 byte/ký tự) | 24.083 |
| **`/ingest` 1000 `doc_ids` × 200 ký tự** | **204.090** |
| `/chat` + `filters` 1 KB/mục × 100 × 5 trường | 550.161 |
| `/chat` + `filters` **1 MB/mục** × 100 | **100.048.510** ⚠️ hợp lệ với mọi phép kiểm hôm nay |

Trần mặc định **1 MiB = 1.048.576 byte** ⇒ dư **5,1×** so với 204.090.
