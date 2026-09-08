# `NEW-14` — bài integration chập chờn: một ngân sách chờ **tính bằng sai đơn vị**

*2026-09-08 · `tests/integration/test_feedback.py` · 1 hàm · chi phí **$0***

> **Nợ:** *"Tầng integration đỏ **một lần** trên `b123d6d` — một commit chỉ đụng
> `plans/WORKLOG.md`. Không xác định được bài nào vì API log của Actions trả 403.
> **Không vá mù:** một bài test bị sửa theo phỏng đoán là một bài test không còn
> đo cái nó tưởng. Khi tầng integration đỏ lần nữa, đọc annotation, rồi mới
> quyết định."*

Lượt đỏ ấy đã tới, ở `27907db`.

---

## 0. ⭐ Bản vá chẩn đoán làm đúng việc nó sinh ra để làm

`W6-03` thêm một bộ phát annotation vào `ci.yml` **chính vì** lần đỏ ở `b123d6d`
chỉ để lại một dòng *"Process completed with exit code 1"*. Lần này:

```
FAILED tests/integration/test_feedback.py::TestTheEdges::test_feedback_needs_a_key_at_all
  - AssertionError: task ghi message không kết thúc
FAILED tests/integration/test_feedback.py::TestTheJoinKeyIsProven::test_a_tenant_cannot_rate_another_tenants_answer
  - AssertionError: task ghi message không kết thúc
```

Không có token, không đọc được log, mà vẫn đủ để bắt đầu. Đó là toàn bộ giá trị
của một bản vá quan sát: nó không sửa gì, nó chỉ làm lần hỏng sau **kể được tên
mình**.

## 0b. ⚠️ Và nghi phạm tôi ghi sẵn **không phải** thủ phạm

Dòng nợ ghi: *"Nghi phạm đáng soi trước: thứ tự phụ thuộc đồng hồ ở phân trang
theo con trỏ `(created_at, id)`."* Sai. Thủ phạm ở một module khác, một cơ chế
khác, không liên quan gì tới phân trang.

Nếu tôi vá theo phỏng đoán ấy thì đã **sửa đúng một chỗ không hỏng và để nguyên
chỗ hỏng** — và tệ hơn: lần đỏ sau sẽ bị đọc là *"đã sửa rồi mà vẫn đỏ"*, tức
một bằng chứng sai đẩy chẩn đoán đi xa thêm một bước. Luật *"đừng vá mù"* trong
dòng nợ không phải một câu khẩu hiệu; ở lượt này nó đã thật sự chặn một sai lầm
mà chính tôi viết ra nghi phạm cho nó.

---

## 1. Nó **chậm**, không **treo** — và đó là hai chẩn đoán khác nhau

Trước khi động vào một dòng nào, phải trả lời câu này. Vá một *treo* bằng cách
nới hạn chờ là giấu một deadlock.

Bằng chứng có sẵn trong chính lượt đỏ: **2 trong 24 bài đỏ, 22 xanh**. Cả 24 đi
qua đúng một hàm ấy. Một deadlock sẽ đỏ **24/24, mọi lượt**. Một cửa sổ đua thì
đỏ vài bài, thỉnh thoảng — đúng hình dạng quan sát được, kể cả lần ở `b123d6d`
(đỏ một lần rồi tự xanh lại ở commit ngay sau).

---

## 2. Bản cũ, và hai câu nó tự khai

```python
def _drain_saves(client: TestClient) -> None:
    """... Một request rẻ khác là cách ép vòng lặp sự kiện quay thêm vài vòng."""
    from serving.core.chat import _PENDING

    for _ in range(50):
        if not _PENDING:
            return
        client.get("/health")
    raise AssertionError("task ghi message không kết thúc")
```

Hai mệnh đề nằm ngay trong mã, và **cả hai đều sai**.

### ⭐ (a) Không cần "ép vòng lặp quay" — nó vốn đang quay

`TestClient` chạy vòng lặp sự kiện trong một **luồng portal riêng**, không trong
luồng test. Task nền tiến triển dù luồng test chỉ ngồi `sleep`. Những request
`/health` ấy chưa bao giờ làm việc mà docstring ghi là chúng làm.

Đo, bằng cách thay `client.get("/health")` bằng `time.sleep(0.001)` thuần:

| | vòng cần | thời gian chờ |
|---|---|---|
| có gọi `/health` | 7 – 9 | 9,0 – 13,4 ms |
| chỉ `sleep(1 ms)` | **5 – 8** | **6,8 – 12,6 ms** |

**24/24 vẫn xanh**, và số vòng cần còn **giảm**. Không chỉ thừa — nó còn là phần
chậm hơn của hai lựa chọn.

### ⭐⭐ (b) Ngân sách bị tính bằng **sai đơn vị**

`"50 vòng"` không phải một khoảng thời gian. Quy ra thực tế nó là

```
ngân sách = 50 × độ trễ(/health) ≈ 50 × 1,2 ms ≈ 60 ms
```

`/health` **không chạm Postgres** (`serving/api/health.py`, và nó nằm trong
`PUBLIC_PATHS`). Nên vế trái đo *độ nhanh của một endpoint trong bộ nhớ*, còn
thứ đang chờ là *một vòng đi-về Postgres*.

⚠️ **Hai đại lượng ấy không tương quan** — và đó là mấu chốt, không phải chuyện
con số nhỏ. Một lần khựng của Postgres (checkpoint, `fsync`, mở kết nối mới
trong pool sau khi pool bị rỗng) kéo dài vế phải **mà không đụng gì tới vế
trái**. Suy ra: **không hằng số nào làm nó an toàn**. Nâng 50 lên 500 chỉ làm
bài test chập chờn hiếm hơn, y hệt việc nới `sleep` trong `test_ttl_expires_entry`
(`NEW-10`, cùng ngày) sẽ chỉ thu hẹp cửa sổ đua chứ không xoá nó.

Đó là chỗ hai bài học ấy khác nhau, và cần nói rõ để không gộp làm một:

| | `test_ttl_expires_entry` | `_drain_saves` |
|---|---|---|
| đơn vị của ngân sách | **đúng** (giây) | **sai** (số vòng của một endpoint khác) |
| sai ở | giá trị quá nhỏ | đại lượng không liên quan |
| cách sửa | **lấy đồng hồ khỏi tay hệ** (tiêm `clock`) | **trả ngân sách về đơn vị thời gian** |

---

## 3. Dựng lại lượt đỏ CI, rồi mới sửa

Hai con số đầu vào đều đo được, không bịa: `/health` **1,2 ms** (= 10,9 ms ÷ 9
vòng, từ lượt đo cục bộ), ghi Postgres **200 ms** (một lần khựng khiêm tốn — nhỏ
hơn nhiều so với mức thường thấy trên runner chia sẻ).

| | kết quả | sau |
|---|---|---:|
| cũ (50 vòng `/health`) · ghi chậm 200 ms | **ĐỎ** | 76,5 ms |
| mới (hạn 10 s) · ghi chậm 200 ms | XANH | 204,1 ms |
| mới (hạn 10 s) · task **treo hẳn** | **ĐỎ**, kèm tên coroutine | 10.000,4 ms |

Hàng thứ nhất dựng lại đúng lượt đỏ. Hàng thứ ba là hàng phải có: bản vá **không
được** biến một deadlock thành một bài test xanh.

---

## 4. Bản mới

```python
DRAIN_TIMEOUT_S = 10.0   # đo cục bộ: 9,0–13,4 ms. Rộng ~750×.

deadline = time.monotonic() + DRAIN_TIMEOUT_S
while _PENDING:
    if time.monotonic() > deadline:
        raise AssertionError(
            f"task ghi message không kết thúc sau {DRAIN_TIMEOUT_S:.0f}s; "
            f"còn treo: {[t.get_coro() for t in _PENDING]}"
        )
    time.sleep(0.001)
```

Ba quyết định:

1. **Hạn tính bằng giây thật.** Nó chỉ tốn thời gian khi hệ **thật sự** hỏng:
   một `INSERT` + `COMMIT` mất hơn 10 s là một lỗi đáng đỏ, không phải một
   runner bận. Đường xanh vẫn thoát ở ~10 ms.
2. **⭐ Thông điệp in ra *cái gì* còn treo, không chỉ *rằng* có gì đó treo.** Đây
   là điểm mà lượt này học được từ chính nó: `NEW-14` tồn tại **một ngày**
   (`b123d6d` 07/09 → `27907db` 08/09) chỉ vì một dòng lỗi không nêu tên. Một
   `AssertionError` mà lượt đỏ sau phải đi điều tra lại từ đầu là cùng một lỗi
   ấy, ở quy mô nhỏ hơn.

   ⚠️ Bản nháp của chính đoạn này viết *"đúng bốn ngày"*. Không đếm, chỉ ước
   lượng — **lần thứ hai trong cùng một trang giấy**, và lần này nằm ngay cạnh
   câu rút ra bài học ấy. Ghi lại vì nó cho thấy luật ở mục 3 dưới đây không
   phải một câu đẹp: nó là thứ tôi vi phạm trong lúc đang viết ra nó.
3. **⚠️⚠️ Tham số `client` biến mất — và suýt thì không.** Bản nháp giữ nó lại
   kèm lý do nghe rất gọn: *"bỏ đi là đổi 22 chỗ gọi trong một bản vá đang
   chữa một bài test chập chờn."* Đếm ra thì có **đúng một** chỗ gọi (`_turn`,
   dòng 157). Con số 22 là số **lượt chạy** mà tôi vừa tự đo ở §2 — tôi lấy
   một con số đúng ở một ngữ cảnh và dùng nó cho một đại lượng khác.

   Đó **đúng là lỗi mà `NEW-10` §4 ghi lại cùng ngày hôm nay**: giữ một thứ
   thừa rồi bịa một lý do nghe hợp lý cho nó. Viết xong bài học ấy chưa được
   một giờ thì tái phạm — và lần này nó thậm chí còn đội lốt *"số đo"*, tức
   dạng ngụy trang tốt hơn. ⭐ Cũng là **cùng một họ với `NEW-09`**: một con số
   đúng trong điều kiện của nó, trích ra khỏi điều kiện ấy thì thành sai. Ba
   lần trong một ngày, ở ba chỗ khác nhau.

   Luật rút ra, ngắn hơn cả ba: **đếm, đừng ước lượng** — nhất là khi con số
   đang biện minh cho việc *không* làm gì.

---

## 5. Việc sinh ra từ lượt này

* ~~`NEW-14`~~ **đóng**. Tầng integration cục bộ: **313 xanh / 2 skip** (315 bài, 0 fail, 489 s).
* 💡 **Chỗ khác cùng hình dạng: không có.** `grep -rn "_PENDING" tests/` cho
  đúng một chỗ dùng. Ngân sách chờ tính bằng "số vòng của một việc khác" là một
  khuôn mẫu đáng quét, nhưng ở repo này nó chỉ có một bản.
* ⚠️ **`W6-03` giữ nguyên giá trị đã chứng minh**: bộ phát annotation là thứ
  duy nhất nối được lượt đỏ với cái tên bài, trong một repo không có token
  Actions. Đừng gỡ nó khi nào chuyện đọc log còn 403.
