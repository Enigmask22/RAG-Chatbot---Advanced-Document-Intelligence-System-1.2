# `W6-02` — Demo công khai: một mối nối, và bốn thứ tưởng là quyết định hoá ra là phép đo

> Ngày 07/09/2026 · chi phí **$0,0034** (2 lượt sinh thật lúc chạy thử) ·
> `plans/reports/probes/w602-cpu-feasibility.json` ·
> `plans/reports/probes/w602-local-mode.json` ·
> `plans/reports/probes/w602-local-parity.json` ·
> `plans/reports/probes/w602-local-latency.json`

DoD: link public mở ra hỏi được ngay < 30 s, có rate limit chống lạm dụng.
Ràng buộc mang theo: `TD-57` (image CUDA 7,35 GB vô dụng ở đây), `TD-72` (làm
nóng), §8 của `security-final.md` (danh sách phơi sáng), và quyết định của
người dùng: sinh bằng **DeepSeek + hạn mức cứng**.

---

## 0. Ba câu trả lời

**Space có phải hệ thống thật không?** Có, và kiểm được: `requirements.txt` cài
`rag-platform` từ `git+…@<sha>`. Không một dòng `rag_core` hay `serving` nào
được chép sang. Toàn bộ điều phối là `ChatService` — cùng lớp `POST /chat` gọi.

**Truy hồi có còn là hệ thống đã đo không?** Có, và đó là một *phép đo* chứ
không phải một lời hứa: 30 truy vấn golden, **overlap@20 = 1,0**, top-1 giống
**30/30** giữa kho local mode và Qdrant server.

**Cái gì giữ hoá đơn?** `daily_total = 500` câu/ngày ⇒ trần **~$1/ngày**. Trần
theo khách cũng có, nhưng nó **giả được bằng một header** và điều đó được viết
ra ở ba chỗ: docstring, README của Space, và giao diện.

---

## 1. ⭐⭐ Bốn thứ tưởng là quyết định thiết kế, hoá ra là phép đo

Hạng mục này bắt đầu bằng bốn câu hỏi mà tôi đã có sẵn câu trả lời "hợp lý", và
cả bốn đều sai. Mỗi cái tốn một script và $0.

| Câu hỏi | Câu trả lời "hợp lý" | Phép đo nói gì |
|---|---|---|
| Chạy CPU được không? | "rerank chậm nhưng chịu được" | **88,1 s** ở `c=50` trên 2 vCPU — gấp **3×** *toàn bộ* ngân sách 30 s |
| Tier miễn phí nào? | "CPU basic, API-only" | Tier CPU miễn phí **không tồn tại** cho Gradio/Docker Space; đường duy nhất là ZeroGPU ⇒ **bắt buộc** GPU |
| Truy hồi trên Space? | "viết lại bằng numpy" | `qdrant-client` local mode chạy **9/9** lời gọi của đường thật ⇒ không viết lại gì |
| Kết quả có lệch không? | "xấp xỉ, chấp nhận được" | **Trùng khớp hoàn toàn**, 30/30 |

Cột giữa là bốn quyết định tôi suýt viết thành mã. Cột phải là bốn script.

---

## 2. ⭐⭐ Mối nối duy nhất, và vì sao chỉ được có một

Space cần **một** thứ khác hệ thống thật: việc chạm GPU phải nằm trong thân một
hàm `@spaces.GPU`. Mọi thứ khác giữ nguyên.

Nên có đúng một lớp bọc, đặt ở đúng ranh giới ấy — `Retriever.retrieve`, trong
đó có embed truy vấn (GPU), tìm hybrid (CPU), cross-encoder (GPU):

```python
class ZeroGpuRetriever(Retriever):
    def retrieve(self, query, top_k=10, *, filters=None, precomputed=None):
        return _retrieve_on_gpu(query, top_k, filters, precomputed)
```

Lý lẽ chống lại việc bọc hẹp hơn là một phép tính, không phải khẩu vị. Bọc
riêng embedder và riêng reranker tiết kiệm ~730 ms thời gian GPU-đang-gắn mỗi
câu (phần quét sparse thuần Python của local mode), nhưng đổi lấy **hai** lần
vào hàng đợi cấp node — và lần thứ hai **có thể không lấy được suất GPU sau khi
lần thứ nhất đã lấy được**, tức một câu hỏi chết giữa chừng, một chế độ hỏng
mới. Hạn mức khách chưa đăng nhập là 2 phút/ngày; ở ~1,5 s mỗi câu thì bọc rộng
vẫn cho ~80 câu. Đổi 730 ms lấy việc không có câu nào chết giữa chừng.

### ⭐⭐ `TD-72` bị nền tảng đảo ngược

`TD-72` vá lượt lạnh bằng **một lượt truy hồi thật** ngay sau khi kích hoạt
bundle (13.386 → 4.427 ms). Trên ZeroGPU đó chính là thứ **cấm**: compute CUDA
ngoài `@spaces.GPU` không có GPU thật đằng sau.

Nhưng nền tảng lại **đòi** nửa còn lại của cùng ý tưởng. Tài liệu HF: *"models
must be placed on `cuda` at the root module level… Lazy-loading or moving models
to CUDA inside `@spaces.GPU` is discouraged."*

Nên hai nửa tách ra: `materialise_weights()` nạp trọng số lên `cuda` ở module
scope (hợp lệ — ZeroGPU giả lập CUDA ngoài hàm rồi gói trọng số ra đĩa), còn
lượt làm nóng có compute thì `warmup=False`. **Cùng một mục tiêu; nền tảng đảo
ngược nửa nào được phép.**

---

## 3. ⭐⭐ Lớp bọc của tôi làm chết một thứ đào xuyên qua nó — và báo cáo thành công

Bản đầu của `ZeroGpuRetriever` đặt tên thuộc tính là `target` và **từ chối**
`__getattr__`, với một lý lẽ nghe rất hợp lý mà tôi tự viết vào docstring: *một
lớp bọc bắt-tất-cả trông như có mọi method của mọi retriever.*

Lượt chạy thử đầu tiên in ra:

```
'trong_so_da_nap': {'reranker': True}
```

Không có `embedder`. `embedder_of()` đào chuỗi `retriever.base.store.embeddings`
bằng duck-typing và **dừng lại ở lớp bọc**. Hệ quả trên ZeroGPU: 2,2 GB trọng số
BGE-M3 nạp bên trong lời gọi `@spaces.GPU` **đầu tiên**, tính vào hạn mức của
người dùng đầu tiên, đúng thứ tài liệu vừa dặn tránh.

Hai điều đáng ghi hơn cả cái bug:

**Thứ nhất — quyết định ấy đã có sẵn trong repo, và tôi đảo nó mà không đọc.**
`serving/core/instrument.py` viết từ `W5-06`:

> `__getattr__` uỷ quyền là bắt buộc chứ không phải tiện tay: `embedder_of()`
> của `W4-10` đào `retriever.base.store.embeddings` bằng duck-typing, và một lớp
> bọc không uỷ quyền sẽ làm semantic cache **tắt lặng lẽ**.

Cùng hàm, cùng cơ chế, cùng kết luận — đã viết ra một lần, ở đúng file mà lớp
bọc mới lẽ ra phải bắt chước.

**Thứ hai — hàm của tôi báo cáo thành công.** `materialise_weights` có dòng
`if holder is None: continue`, gộp "bundle không khai reranker" (hợp lệ) với
"không đào ra được embedder" (lắp sai) vào một nhánh. Nên lần lắp sai đầu tiên
trả về một dict *trông ổn* thay vì một tiếng nổ. Bản sau phân biệt hai ca:
embedder vắng mặt ⇒ `RuntimeError` kèm câu chẩn đoán; reranker vắng mặt ⇒ hợp
lệ; nạp **hỏng** ⇒ vẫn log-và-đi-tiếp (cùng lý lẽ `BundleRegistry._warm`).

⚠️ Cái giá của việc theo quy ước: tên `_inner` khiến `_unwrap_traced` bóc được
lớp này ⇒ `wants_precomputed` trả True ⇒ `ChatService` sẽ embed câu hỏi ở tiến
trình chính, ngoài `@spaces.GPU`. Hôm nay không xảy ra vì cả nhánh ấy nằm sau
`cache is not None` và Space chạy `cache=None`. Một cái bẫy **được ghi ra** tốt
hơn một cái bẫy né bằng một tên khác thường mà lặng lẽ hỏng chỗ khác.

---

## 4. ⭐⭐ Một kiểu dữ liệu hứa một chế độ mà mã từ chối phục vụ

`ChatService.sessions` khai `async_sessionmaker[AsyncSession] | None` từ
`W4-06`. Dòng thứ hai của `_prepare`:

```python
if self.sessions is None:
    raise GenerationUnavailable("chưa cấu hình Postgres cho serving")
```

HF Space không có Postgres. Tôi lắp `sessions=None` vì chữ ký cho phép, và biết
mình sai ở *runtime*, sau khi 2,7 GB trọng số đã nạp xong.

Chế độ ấy có thật và có người dùng: một endpoint RAG không lưu vết. Nên bản vá
là **làm cho kiểu nói đúng**, không phải thu hẹp kiểu:

* `_history` — có `conversation_id` mà không có kho ⇒ `ConversationNotFound`,
  **không** trả `[]`. Trả `[]` để client tin nó có mạch hội thoại trong khi mỗi
  lượt là độc lập là đúng kiểu hỏng câm dự án này từ chối ở mọi tầng.
* `_open_turn` — vẫn phát id. Khung `meta` là hợp đồng với client, không phải
  hệ quả của việc có Postgres; chúng chỉ không trỏ tới hàng nào.
* `_save` — trả về sớm.

⭐ **Và gỡ dòng chặn ấy đi thì 2.453 bài test vẫn xanh.** Không bài nào canh nó,
theo cả hai chiều. Đó là lý do `tests/unit/test_chat_stateless.py` tồn tại.

---

## 5. Kho index: 239 MB, và một phép đo về danh tính chứ không về kích thước

`scripts/export_local_index.py` cuốn 15.814 point từ Qdrant server sang một kho
local mode (170 s, một file sqlite 239 MB).

Câu hỏi đáng hỏi không phải "có đủ 15.814 point không" — một kho đếm đủ mà xếp
hạng khác là một hệ thống **khác**, và không gì đỏ. Nên phép kiểm là xếp hạng:

| | |
|---|---|
| overlap@20 (30 truy vấn golden) | **1,0000** (min 1,0, 30/30 hoàn hảo) |
| top-1 giống hệt | **30/30** |
| trung vị độ trễ, server | 719,6 ms |
| trung vị độ trễ, local | 1.546,9 ms |

Điều này **không** hiển nhiên: server tìm xấp xỉ (HNSW), local mode nhân ma
trận toàn bộ. Ở 15.814 point thì HNSW không bỏ sót gì, nên hai bên trùng nhau —
nhưng đó là một sự thật đo được về *quy mô này*, không phải một tính chất.

Phân rã 830 ms chênh lệch (`w602-local-latency.json`):

| chặng | local mode |
|---|---|
| `embed_query_hybrid` | 11,5 ms |
| dense `c=50` | **77,0 ms** |
| sparse `c=50` | **764,4 ms** |
| hybrid `c=50` | 726,3 ms |

⭐ Toàn bộ chênh lệch là **nhánh sparse**: local mode quét tuyến tính bằng
Python thuần trên 15.814 vector thưa. Đây là cái giá của "không viết lại truy
hồi", và nó rẻ hơn nhiều so với giá của một bản cài thứ hai.

⚠️ Kho local ăn **908 MB RAM** khi mở — gấp 3,8× kích thước trên đĩa, vì nó nạp
toàn bộ vào bộ nhớ. Cộng embedder + reranker ⇒ ~4,2 GB.

---

## 6. Trần chi tiêu: cái nào thật sự giữ hoá đơn

Quyết định của người dùng: DeepSeek thật + hạn mức cứng. `space/guard.py`:

| tham số | mặc định | ý nghĩa |
|---|---|---|
| `DEMO_DAILY_TOTAL` | 500 | trần **tổng** ⇒ ~$1/ngày ở $0,002/câu |
| `DEMO_PER_IP_DAILY` | 20 | trần theo khách |
| `DEMO_PER_IP_BURST` | 3 / 60 s | chống bấm liên tục |
| `DEMO_GENERATION=off` | — | cầu dao thủ công, một chiều |

Bốn quyết định, và cái thứ hai là cái phải đọc:

1. **Đếm TRƯỚC khi gọi model.** Đếm-khi-thành-công nghe công bằng hơn và sai
   theo hướng đắt tiền: một lượt sinh chết giữa chừng vẫn đã tiêu token, nên
   một vòng lặp thử-lại sẽ **miễn phí vô hạn**. `commit()` gộp xin phép và ghi
   nhận thành một thao tác nguyên tử — `check()` rồi `commit()` ở hai lời gọi
   là đúng cửa sổ đua mà `W6-05` đo được 8 request lọt qua ở tầng cache
   (`AU-11`).
2. **⭐⭐ Trần theo IP là khuyến cáo, trần tổng mới là hàng rào.** Space đứng
   sau proxy HF nên `request.client.host` là địa chỉ proxy — dùng nó thì mọi
   khách gộp thành một người. Địa chỉ thật ở `x-forwarded-for`, mà header ấy
   **người gọi ghi được**. Vượt trần theo IP chỉ cần đổi một header. Điều đó
   không làm nó vô nghĩa (nó chặn lạm dụng vô tình, vốn là đa số), nhưng nó có
   nghĩa là con số suy ra chi phí tối đa phải là `daily_total`.
3. **Chỉ giữ hash có muối của IP.** IP là dữ liệu cá nhân; bộ đếm chỉ cần biết
   "có cùng một người không". Muối ngẫu nhiên mỗi tiến trình.
4. **Bộ đếm trong RAM ⇒ khởi động lại là xoá.** Space miễn phí không có ổ bền.
   Trần thật ra là "mỗi vòng đời tiến trình". Không ai ngoài chủ Space kích được
   việc khởi động lại, nên chấp nhận được — nhưng nó vẫn là một cách con số 500
   nói dối, nên nó được nói ra.

⭐ Từ chối **sinh** không phải từ chối **truy hồi**: truy hồi không tốn tiền, và
người chạm trần vẫn xem được hệ thống tìm ra gì. Với một demo RAG thì đó là
phần đáng xem nhất.

Ngoài ra `DailyBudget` của `W4-08` (`chat_daily_budget_usd = 1.0`) vẫn chạy —
một trần thứ hai, tính bằng đô la thật thay vì bằng số câu, ở một tầng khác.

---

## 7. ⚠️ Space **không** chạy đúng bộ phụ thuộc đã eval

| | repo | ZeroGPU |
|---|---|---|
| Python | 3.13.11 | chỉ **3.12.12** hoặc **3.10.13** |
| torch | 2.13.0+cu126 | tài liệu liệt kê tới 2.11, nói "to latest" |

`retriever_name` mà `TD-38` đối chiếu **không** mã hoá phiên bản torch hay
Python, nên `_check_identity` xanh trong khi môi trường vẫn lệch. Điều đó có
thể dịch vài chữ số cuối của điểm cross-encoder fp16.

Không có phép kiểm nào bắt được, và tôi **không** dựng một phép kiểm giả để làm
như có. Ghi thành `TD-87`.

---

## 8. Đo được

| | |
|---|---|
| Khởi động (laptop, trọng số đã cache) | **28,1 s** — mở index 3,4 s · kích hoạt 17,5 s · nạp trọng số 7,2 s |
| Một lượt đầy đủ (laptop, GPU thật) | 442 khung SSE · **6,6 s** · prepare 2.043 ms · TTFB 920 ms |
| Chi phí một câu | **$0,001691** |
| Trích dẫn xác minh | 3/4 |
| Kho index | 15.814 point · 239 MB đĩa · **908 MB RAM** |
| Gói gửi lên Space | 11 file · 239,0 MB |

---

## 9. Chấm bằng tiêm lỗi

**25/25 đỏ** sau hai lượt. Lượt một: 22/25 — một phép không tiêm được (lỗi
chuỗi tìm kiếm của chính bộ tiêm) và **hai lỗ thật trong test của tôi**:

* `M15` (*luôn truyền `precomputed`*) **sống sót** vì nhánh nền giả của tôi có
  sẵn `precomputed=None` trong chữ ký — nên `assert calls[-1]["precomputed"] is
  None` đúng dù lớp bọc có truyền tường minh hay không. Luật thật là *nhánh nền
  có thể **không có** kwarg ấy*. Bài mới dùng một nhánh nền không nhận, và đỏ
  bằng `TypeError`.
* `M19` (*`_save` chạy cả khi không có sessions*) **sống sót** vì `_save` nuốt
  mọi exception vào `logger.exception` — đúng như nó nên làm, vì task nền không
  có ai bắt. Hệ quả: một `_save` quên kiểm sẽ nổ mà **không đổi một khung SSE
  nào**. Chỗ duy nhất nhìn thấy được là log, nên bài mới đọc log: *không trạng
  thái* phải nghĩa là **không thử ghi**, không phải *thử rồi thất bại êm*.

Và một bài test tự nó đỏ ngay lần chạy đầu vì lý do đáng ghi: bản đầu của
`test_doc_con_tro_bang_read_pointer` hỏi `'"CURRENT"' not in source` — và đỏ vì
đúng **dòng chú thích giải thích tại sao không được làm thế** có chứa chuỗi ấy.
Cùng họ với ba lỗ mà lượt tiêm của `W6-03` tìm ra: *"chuỗi này có xuất hiện ở
đâu không"* chưa bao giờ là câu hỏi đang cần hỏi. Bản sau soi **AST**, nơi chú
thích không tồn tại.

---

## 10. ⭐⭐ Ba lượt CI, và cái thứ ba là lỗi đáng ghi nhất của cả hạng mục

### `7d14449` — 3/4 job đỏ, và **exit code 2** chứ không phải một bài đỏ

`pytest` **import mọi module test khi collect**, kể cả module mà `-m` loại ra
ngay sau đó. Test của hạng mục này nạp `guard`/`zerogpu`, và `zerogpu`
`import spaces`. Tầng unit và integration không cài `--extra space` ⇒ chết ở
bước collect.

⭐⭐ Và **bản vá chẩn đoán của `W6-03` không đọc được đúng ca này.** Nó lấy mục
`short test summary info` — mục chỉ tồn tại khi đã chạy được bài nào. Exit code
2 là chết *trước* đó, nên bước phát annotation ra rỗng và lượt đỏ lại về đúng
chữ `"exit code 2"`, y như trước khi có bản vá. Sửa: không có mục summary thì
phát 25 dòng cuối — *vắng mặt mục ấy tự nó là một chẩn đoán*, không phải một
chỗ trống. Thêm cùng cơ chế cho `mypy`.

### `c96e293` — 3/4 xanh, `lint` vẫn đỏ, và giờ đọc được

Annotation trả về đúng một dòng:

```
space/app.py:356: error: "Textbox" has no attribute "submit"
```

Cùng mã ấy `make lint` **xanh trên máy**. Câu hỏi thật không phải "sửa dòng
nào" mà **"vì sao hai bên khác nhau"**.

Giả thuyết đầu của tôi: `mypy_path = "packages:space"` tách bằng `os.pathsep`,
nên là *một* đường dẫn vô nghĩa trên Windows và *hai* trên Linux. **Tự bác bỏ
bằng `MYPYPATH=space uv run mypy` — vẫn xanh.** (Vẫn bỏ dòng ấy đi: nó sai theo
một cách khác đáng bỏ, chỉ là không phải nguyên nhân.)

### ⭐⭐ Nguyên nhân thật: mypy xanh trên máy tôi **vì tôi đã chạy app**

Gradio **tự sinh 62 file `.pyi` vào site-packages** lúc class component được
tạo (`component_meta.create_or_modify_pyi`). Dấu thời gian của
`gradio/components/textbox.pyi` trên máy này là **14:34** — đúng lúc tôi chạy
lượt thử đầu-cuối. Nên bề mặt kiểu của gradio chỉ *tồn tại* sau khi ai đó đã
chạy chương trình:

| | `.pyi` có? | `mypy` |
|---|---|---|
| máy dev, đã `make space-run` | có (62 file) | **xanh** |
| runner CI, chưa bao giờ chạy app | không | **đỏ** |

Đo chứ không suy: chuyển tạm 62 file ấy ra ngoài rồi `mypy --no-incremental`
cho **đúng một dòng** lỗi của CI. Sau bản vá thì xanh ở **cả hai** trạng thái.

Chú thích ở `ci.yml` đã ghi từ `W5-09`: *"phán quyết của `make lint` không phải
một tính chất của mã, nó là tính chất của mã **cộng một môi trường**"*. Đây là
yếu tố **thứ ba**: mã + môi trường + **lịch sử thao tác trong môi trường đó**.
Và nó hỏng theo chiều nguy hiểm nhất — xanh cho người vừa chạy app, đỏ cho mọi
người khác, tức người gây ra nó là người duy nhất không nhìn thấy nó.

Bản vá: `gradio.*` khai `follow_imports = "skip"`. Một bề mặt kiểu **sinh lúc
chạy** không được làm đầu vào của một phép kiểm **tĩnh**. Giá phải trả viết
thẳng trong `pyproject.toml`: gõ sai tên tham số Gradio không bị mypy bắt —
nhưng nó chưa từng bị bắt một cách *đáng tin*, vì "có bị bắt hay không" phụ
thuộc vào việc đã chạy app hay chưa.

---

## 11. Việc sinh ra từ lượt này

* `TD-87` — Space chạy Python/torch khác môi trường eval, và `retriever_name`
  không mã hoá điều đó.
