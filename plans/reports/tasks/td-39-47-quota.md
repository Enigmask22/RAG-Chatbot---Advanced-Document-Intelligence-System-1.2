# `TD-39` + `TD-47` — hai bộ đếm dùng chung, và câu hỏi chặn đường là một câu hỏi **sai**

*2026-09-08 · `serving/core/quota.py` + 4 mối nối · **31 test mới** · tiêm **16/16 đỏ** · chi phí **$0***

> **Nợ `TD-39`:** *"Hạn mức nhịp đếm trong tiến trình, nên hạn mức thật là một
> hàm của số pod. 4 replica cho mỗi tenant **240** request/phút chứ không phải
> 60. […] Câu hỏi thiết kế chưa trả lời là **Redis chết thì mở cổng hay đóng
> cổng**: mở = mất hạn mức đúng lúc hệ thống đang yếu, đóng = một phụ thuộc mới
> có thể làm sập toàn bộ API. **Phải quyết trước khi viết.**"*
>
> **Nợ `TD-47`:** *"Trần chi phí ngày đếm trong tiến trình, và không phân theo
> tenant. […] Làm **cùng lượt** với `TD-39` — hai bộ đếm, một hạ tầng, và quyết
> định fail-open vs fail-closed **phải giống nhau ở cả hai**."*

---

## 0. Đo được, trên Redis thật

| mệnh đề | trước | sau |
|---|---:|---:|
| hai "replica" cùng một tenant, trần **4**/phút | **8** lượt qua | **4** |
| 10 request **đồng thời**, trần **3** | (không xác định) | **3** |
| hai replica, trần chi phí **$1**, mỗi bên tiêu $0,6 | cả hai **qua** | replica 2 **bị chặn** ở $1,20 |
| một tenant đốt hết ⇒ tenant khác | **429** | **đi bình thường** |
| Redis chết, 5 request | — | **có 200 và có 429** — API sống, hàng rào còn |

Hàng thứ hai là hàng chỉ chạy được trên Redis thật: tuần tự thì một cuộc đua
không bao giờ xảy ra.

---

## 1. ⭐⭐ Câu hỏi chặn đường của `TD-39` là một **câu hỏi sai**

Dòng nợ bảo phải quyết *"mở cổng hay đóng cổng"* trước khi viết. Quyết được —
nhưng câu trả lời là **cả hai đều sai**, vì có một lựa chọn thứ ba mà dòng nợ
không xét:

> **Redis hỏng ⇒ tụt về đúng bộ đếm trong tiến trình đang chạy hôm nay.**

Nó **không** phải "mở cổng": trần vẫn còn, chỉ là N× quá rộng — tức **đúng bằng
bảo đảm hiện tại, không bao giờ tệ hơn**. Và nó không biến sự cố Redis thành sự
cố API.

⚠️ Điều đó cũng sửa lại lý lẽ của `TD-47` (*"quyết định phải giống nhau ở cả
hai"*). Hai câu trả lời **giống nhau**, nhưng **không phải vì chúng buộc phải
giống** — mà vì cả hai đều đã có sẵn một hàng rào cục bộ **có biên**. Nếu một
trong hai không có, câu trả lời của nó đã khác. Lý do đúng quan trọng hơn kết
luận đúng: người sau thêm bộ đếm thứ ba sẽ áp nhầm luật.

---

## 2. ⭐⭐ Và quyết định ấy **chưa đủ** — "mở cổng" phải nhanh

Redis **từ chối kết nối** thì đường lui rẻ. Redis **treo** — phân vùng mạng,
node đóng băng, đĩa đầy — thì mỗi request trả giá bằng cả socket timeout
**trước khi** được tụt về. Với timeout mặc định của `redis-py` đó là hàng giây
cho **mọi** request: API không trả 5xx nhưng chậm tới mức không dùng được.

⭐ Đó **đúng là** ca *"một phụ thuộc mới có thể làm sập toàn bộ API"* mà dòng nợ
cảnh báo — nó chỉ không sập theo cách người ta hình dung. **Một bản vá dừng ở §1
sẽ mang đúng tên "fail-open" mà vẫn hỏng y như fail-closed.**

Chữa bằng `CircuitBreaker` của `W4-08` — đã có, đã có test, đúng hình dạng này.
Không viết bộ ngắt mạch thứ hai. Hai bộ đếm dùng **hai** mạch riêng: `EVAL` và
`INCRBYFLOAT` hỏng độc lập, và một mạch dùng chung sẽ tắt cả hai vì lỗi của một.

### ⚠️ Và bộ ngắt mạch ấy suýt tự khoá vĩnh viễn

Bản đầu gọi `allow()` **bên trong** `try`, ném khi mạch mở. Nhánh `except` chạy
`record("failure")` — mà `record("failure")` **đặt lại `_opened_at`**. Hệ quả:
trong lúc request vẫn đến đều, đồng hồ nguội **không bao giờ chạy hết**, và
Redis **không bao giờ được thử lại**.

Một cơ chế thêm vào để một sự cố tạm thời đừng lan ra, tự biến nó thành **vĩnh
viễn**. Bắt được bằng cách đọc `CircuitBreaker.record` chứ không bằng test —
đọc mã của thứ mình đang dùng lại, không chỉ chữ ký của nó.

---

## 3. ⭐⭐ `TD-47` có nửa thứ hai mà Redis **không** tự giải quyết

*"Một tenant đốt hết ngân sách thì mọi tenant còn lại nhận `429`"* là chuyện của
**khoá**, không của hạ tầng. Đưa `DailyBudget` lên Redis y nguyên thì bốn replica
sẽ cùng chia sẻ đúng một bộ đếm toàn cục — và vẫn chặn nhầm.

Cách hiển nhiên — cho `DailyBudget` biết `tenant_id` — **phá ranh giới hai
plane**: nó sống trong `rag_core/llm/router.py`, nơi phục vụ **cả đường eval**,
và ở đường ấy không có tenant nào. `NEW-01` có test AST chặn đúng chiều này.

Nên trần theo tenant áp ở **serving** (`ChatService._prepare`), ngay cạnh trần
toàn cục; trần toàn cục **ở lại** làm hàng rào thô thứ hai — nó là thứ duy nhất
còn chặn khi Redis hỏng và cả hai đường lui cùng bị nới ra. Cùng lý lẽ với
`tenant_filter()`: tenancy là khái niệm của biên HTTP, không của lõi truy hồi.

⚠️ Và nó **hỏi trước, ghi sau**: `peek` ở `_prepare` (trước khi tốn một lượt truy
hồi), `charge` với **chi phí thật** ở cuối stream, chạy nền. Ghi nhận một ước
lượng rồi không sửa lại là cách để sổ chi tiêu trôi khỏi hoá đơn.

---

## 4. ⭐⭐ Tôi làm tầng unit phụ thuộc vào việc **máy dev có Redis hay không**

`quota_shared` mặc định `True`, nên mọi `create_app` trong test đi qua Redis nếu
có. Ba bài của `test_auth_ratelimit` **xanh khi Docker tắt và đỏ khi Docker bật**
— và tôi chỉ thấy vì tình cờ bật Docker lại để chạy bài Redis thật.

Đúng họ với `NEW-14` sáng nay: **một bài test mà kết quả do một thứ ngoài nó
quyết định**. Lần này thứ ấy còn không nằm trong repo.

Chữa: fixture khai `quota_shared=False` **tường minh** (nó đang kiểm bộ đếm cục
bộ), và thêm một bài ở tầng app ghim đúng tính chất production — `quota_shared`
bật + Redis không tồn tại ⇒ **vừa có 200 vừa có 429**. Bỏ một trong hai vế là
hỏng: chỉ 200 nghĩa là mất hàng rào, chỉ 429 nghĩa là API sập theo Redis.

---

## 5. Tiêm lỗi: **16/16 đỏ**, sau hai lượt

Lượt một **12/16**. Bốn phép sống, và **một trong bốn là phép tiêm tồi của
tôi**:

| phép sống | hoá ra |
|---|---|
| `M13` bỏ chặn theo tenant ở `_prepare` | ⭐⭐ **Chỗ nối của `TD-47` không có bài test nào.** Tôi viết 18 bài cho `quota.py` và **không bài nào** đi qua `ChatService`. Cùng lỗi lượt hai của `NEW-10`: mệnh đề trung tâm chưa được kiểm vì mọi bài đều dựng thẳng đối tượng thay vì đi đường thật. |
| `M3` ghi failure khi mạch đã mở | ⭐⭐ Bài của tôi dùng `cooldown_s=0.0` và **không thể đỏ**: với cooldown 0 thì mạch vào half-open ngay cả khi đồng hồ vừa bị đặt lại. Viết lại để đo thứ **quan sát được trực tiếp** (`record` có được gọi không) thay vì một hệ quả gián tiếp qua đồng hồ. |
| `M16` `quota_shared` mặc định `False` | Không bài nào ghim mặc định — mà một hạng mục về hạn mức phân tán mặc định tắt là một hạng mục **không chạy ở production**. |
| `M4` Lua "lấy đồng hồ tiến trình" | ⚠️ **Phép tiêm sai, không phải lỗ test.** Nó đọc `ARGV[3]` vốn không tồn tại, nên Lua lỗi ⇒ đường lui nuốt ⇒ test vẫn xanh. Sửa phép tiêm thành `local now = 0`: đỏ ngay. **Một mutant sống vì nó không diễn đạt được điều mình định nói cũng là một kết quả sai** — nó báo một lỗ test không có thật. |

---

## 6. Việc sinh ra từ lượt này

* ~~`TD-39`~~ **đóng**: hạn mức nhịp dùng chung, có đường lui, có ngắt mạch, có
  bộ đếm `degraded`. ⚠️ Nửa thứ hai của dòng nợ — *"request chưa xác thực hoàn
  toàn không bị chặn"* — **vẫn mở, và đúng chỗ**: `AuthMiddleware` từ chối bằng
  header sau 0 byte (`NEW-12` §3), nên thứ bị tiêu là **kết nối**, và chặn theo
  IP thuộc về reverse proxy.
* ~~`TD-47`~~ **đóng**: theo tenant, trên Redis, ghi chi phí thật.
* ⚠️ `TD-63` **hẹp lại một nửa**: `NEW-10` ghi *"đi đường (b) thì single-flight,
  `CostBudget` và hạn mức nhịp **cùng lúc** mất hiệu lực theo tỉ lệ số replica —
  ba nợ một chỗ trả"*. Hai trong ba đã trả. Còn **single-flight**, và nó là cái
  khó nhất: gộp request đang bay cần một khoá phân tán, không phải một bộ đếm.
* 💡 **`degraded` chưa lên bảng.** Nó là con số duy nhất phân biệt *"hạn mức
  đang đúng"* với *"hạn mức đang là N× và không ai biết"*, và hôm nay nó chỉ
  sống trong log. Một dòng ở `/metrics` là việc nhỏ; chưa mở nợ vì `W5-07` đã có
  chỗ đúng để đặt và lượt này không đụng tới `MetricsSink`.
