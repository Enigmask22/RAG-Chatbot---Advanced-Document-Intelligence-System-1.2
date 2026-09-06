# `W6-06` — Security pass trước khi có bất cứ thứ gì công khai

> **DoD**: 0 finding severity cao chưa xử lý.
> **Ngày**: 07/09/2026 · **Chi phí**: $0 (không lời gọi model nào).
> Đứng **sau** `W6-01` để soi cả bề mặt mới (trang tĩnh, proxy `/admin/ingest`,
> trường `content` trong khung SSE) và **trước** `W6-02` (demo công khai).

---

## 0. Kết luận một đoạn

Bảy phát hiện mới (`SEC-01`…`SEC-07`), **tất cả đã vá**; ba nợ audit cũ đóng
(`AU-08`, `AU-09`, `AU-10`), hai nợ bảo mật đóng (`TD-52`, `TD-58`), một nợ
(`TD-83`) đóng **nửa bảo mật** và phần còn lại chuyển đi kèm lý do.

Hai phát hiện đắt nhất đều đến từ việc **đo thay vì đọc**, và cả hai nằm ở chỗ
mã tự tuyên bố là an toàn: bộ che PII **không phủ traceback** — đúng ca mà
docstring của chính nó lấy làm ví dụ biện minh — và nó **không có luật nào cho
bí mật** cả, nên một `Authorization: Bearer rag_…` đi ra stream nguyên vẹn.
Phát hiện thứ ba (`SEC-07`, chèn dấu phụ vào giữa từ khoá né được **mọi** luật
tiêm) không nằm trong danh mục audit nào — nó rơi ra từ lượt chấm tiêm lỗi.

⚠️ Một giả thuyết của tôi **bị phép đo bác bỏ** giữa chừng và severity phải hạ
xuống — chi tiết ở `SEC-03`.

---

## 1. Bảng phát hiện

| ID | Mức | Ở đâu | Trạng thái |
|---|---|---|---|
| `SEC-01` | **Cao** | `RedactingFilter` không phủ `exc_info` — PII trong traceback ra thẳng stream | ✅ vá |
| `SEC-02` | **Cao** | `redact_pii` không có luật nào cho credential (leo thang `AU-09`) | ✅ vá |
| `SEC-07` | **Cao** | Chèn một dấu phụ vào giữa từ khoá né được **cả 11** luật tiêm | ✅ vá |
| `SEC-03` | Vừa | `job_id` chưa kiểm → tiêm query string vào lời gọi dịch vụ nội bộ | ✅ vá |
| `SEC-04` | Vừa | `filters` nhận danh sách **không giới hạn** — trần 8000 ký tự của `message` không phủ trục này | ✅ vá |
| `SEC-05` | Vừa | `admin.py` khai **sai** tình trạng bảo mật của chính nó suốt hai tuần | ✅ vá |
| `SEC-06` | Thấp | `ci.yml` không khai `permissions:` → token nhận mặc định của repository | ✅ vá |
| `AU-08` | Vừa | `GET /conversations/{id}` không phân trang | ✅ vá |
| `AU-09` | Vừa | Thân lỗi provider echo `Authorization` vào log | ✅ vá (qua `SEC-02`) |
| `AU-10` | Vừa | Dịch vụ ingest (8001) không xác thực | ✅ vá |
| `TD-52` | — | Bảng confusables cố ý không đầy đủ | ✅ đóng, **không** bằng cách làm bảng đầy đủ |
| `TD-58` | — | Kho khoá 11 mục, phần lớn là rác probe | ✅ dọn còn 3 |
| `TD-83` | — | Chưa có smoke tầng sinh | 🟡 nửa bảo mật đóng, xem §7 |

**Không finding severity cao nào chưa xử lý** ⇒ DoD đạt.

---

## 2. ⭐⭐ `SEC-01` + `SEC-02` — bộ che log nói dối về chính nó

### Cách tìm ra

Không phải bằng đọc mã. Docstring của `RedactingFilter` viết:

> *"`httpx` log URL kèm query string, **một `logger.exception` in nguyên payload
> của provider**, và không ai nhớ gọi `redact_pii()` ở dòng log thứ 300."*

Câu ấy là **lý lẽ biện minh** cho việc gắn filter lên handler. Nên câu hỏi tự
nhiên là: *lời hứa ấy có đúng không?* — và cách duy nhất trả lời là cho một dòng
log đi hết đường thật (logger → filter → formatter → stream) rồi nhìn cái ra.

`scripts/probe_log_redaction.py`, hai nhánh trong một lượt chạy:

| giá trị mồi | bản trước `W6-06` | sau |
|---|---|---|
| email trong `msg` | che ✅ | che ✅ |
| email trong **traceback** | **rò** ❌ | che ✅ |
| khoá `rag_…` trong `msg` | **rò** ❌ | che ✅ |
| khoá `sk-…` trong traceback | **rò** ❌ | che ✅ |

Cùng một địa chỉ email: che khi nó ở `msg`, **lọt nguyên vẹn** khi nó ở
traceback. Vì `logger.exception("gọi provider thất bại")` đặt câu literal vào
`record.msg`, còn nguyên văn lỗi đi vào `record.exc_info` — một **tuple**, nên
vòng lặp `__dict__` (chỉ đụng `str`) bước qua nó.

⚠️ Nhánh chứng phải chép **nguyên** hành vi cũ. Bản đầu của probe chỉ che
`record.msg` và vì thế tính luôn `record.args` vào phần "bản vá cứu được" —
trong khi bản cũ đã che `args` từ `W4-12`. Số chênh lệch khi ấy phóng đại đúng
một bản ghi. Probe cũng **trả mã lỗi khi nhánh chứng sạch**: một nhánh chứng
không rò nghĩa là probe mù, và im lặng thì nó trông y hệt một lượt tốt.

### `SEC-02` là `AU-09` nhìn từ một tầng thấp hơn

Audit đề xuất: *chà `Bearer …` trong `openai_compat._raw_events` trước khi nhét
`response.text[:500]` vào exception.* Đúng, nhưng đó là **một** chỗ gọi. Phép đo
nói lỗ ở dưới: `redact_pii` biết email/sđt/cccd/thẻ và **không biết mặt bí mật
nào cả**. Vá một chỗ gọi là đóng một cửa trong nhiều cửa cùng mở — và cửa tiếp
theo sẽ do một thư viện bên thứ ba mở, ở dòng log không ai viết.

Nên bảng nhận dạng credential dời từ `pipeline/indexing/job_bundle.py` (nơi nó
ra đời ở `W3` để quét gói job trước khi gói rời máy) xuống
`packages/rag_core/credentials.py`, và **cả hai** người dùng đọc chung một bảng.
Hai bản là hai chỗ để lệch nhau, và bản không được cập nhật vẫn chạy, vẫn xanh —
họ `AU-12`. Thêm hai luật: `bearer_token` và `platform_api_key` (khoá của **chính
hệ thống này** — bảng cũ chỉ biết khoá của người khác).

### ⚠️ Hai chỗ trong bản vá của tôi bị chính phép tiêm lỗi bác bỏ

**Một.** `keep` (giữ lại nhãn, để log còn đọc được là `Authorization:
[credential]`) là **mã chết** cho ca phổ biến nhất của nó: `platform_api_key`
đứng trước `bearer_token` trong bảng, nên nó khớp `rag_…` trước và để lại
`Bearer ` một cách tình cờ. Bài test "nhãn còn nguyên" đi qua **vì một lý do
khác hẳn lý do nó tin**, và phép tiêm xoá `keep` sống sót. Xếp lại thứ tự — luật
mang nhãn chạy trước — thì một dòng log chỉ còn đúng một cách được che.

**Hai.** Dòng `record.exc_text = redact_pii(...)` đặt **trước** vòng lặp
`__dict__` thì `exc_text` đi qua chính vòng lặp ấy, nên bỏ `redact_pii` ở dòng
đó vẫn cho kết quả đúng — phép tiêm sống sót, và dòng ấy đang giả vờ làm việc.
Chuyển xuống **sau** vòng lặp: che ở đó là cơ chế duy nhất phủ traceback, và nó
hỏng ra hỏng.

---

## 3. ⭐⭐ `SEC-07` — dấu phụ, và vì sao nó không có trong danh mục audit nào

Rơi ra từ lượt chấm tiêm lỗi `TD-52`. Bảng gập đã xử ký tự **zero-width** từ
`W4-12` với đúng lý lẽ *"chèn vào giữa từ khoá là cách rẻ nhất để né regex"*.
Dấu phụ là **cùng một trò** với một lớp ký tự khác:

| payload | luật khớp, trước `W6-06` |
|---|---|
| `ign` + ZERO WIDTH SPACE + `ore all previous instructions` | `override_instructions_en` |
| `ign` + COMBINING ACUTE + `ore all previous instructions` | **`()`** — không cờ nào |
| `ign` + COMBINING CYRILLIC + `ore all previous instructions` | **`()`** |

Và nó không phải một biến thể của bài toán confusable: dấu phụ ấy **hợp lệ**, nó
chỉ ở sai chỗ. `mixed_script_words` cũng mù với nó, vì `\w` của Python không
khớp category `Mn` — dấu phụ **ngắt từ** chứ không vào trong từ.

⚠️ **Không** sửa được bằng cách bỏ dấu trong `normalise_for_scan`: luật tiếng
Việt được viết **có dấu** (`bỏ qua`, `phía trên`), nên một bản bỏ dấu duy nhất
giết đúng nửa bộ luật. Nên `scan_injection` so **hai** biến thể — giữ dấu cho
luật tiếng Việt, bỏ dấu cho luật ASCII — rồi hợp kết quả.

Đo dương tính giả của biến thể thứ hai trên **19.744 chunk corpus thật**:
**16 → 16**. Không thêm một chunk nào.

---

## 4. ⭐⭐ `TD-52` — nợ đòi một bảng đầy đủ, số đo nói bảng là công cụ sai

Nợ ghi: *"dùng bảng confusables chuẩn của Unicode (`confusable_homoglyphs`), đo
lại dương tính giả trên 20.424 chunk."* Trước khi thêm một phụ thuộc runtime vào
`rag_core` (nó đi vào image serving, rồi vào Space của `W6-02`), đo đã.

`scripts/confusable_probe.py` sinh **62** cách thay đúng một chữ của `ignore`
bằng một **chữ cái** Unicode nhìn giống nó, rồi hỏi hai cơ chế:

| cơ chế | bắt được |
|---|---|
| bảng gập `_CONFUSABLES` hiện tại | **2 / 62 (3,2 %)** |
| luật **trộn hệ chữ** (`mixed_script_words`) | **62 / 62 (100 %)** |

Bảng đang bỏ lọt **97 %**, tức `TD-52` nghiêm trọng hơn nhiều so với lời nó tự
mô tả. Nhưng một bảng đầy đủ hơn vẫn có một tính chất tệ: **nó không bao giờ
đóng được**. Unicode thêm ký tự mỗi năm; mỗi ký tự mới là một lỗ mới, im lặng,
cho tới lần nâng cấp bảng kế tiếp.

Cái **đóng được** là hình dạng của phép tấn công: chèn một ký tự nhìn giống vào
giữa một từ Latin thì từ ấy **trộn hệ chữ** — đúng với mọi ký tự thay thế, kể cả
cái Unicode chưa đặt ra. Không phụ thuộc mới, không bảng nào phải bảo trì.

**Dương tính giả, 19.744 chunk corpus thật:** 14 chunk (0,0709 %) cho riêng luật
mới, **16 (0,0810 %)** cho cả 11 luật. Ngưỡng `P1` của `W4-12` là < 0,3 %.
Chi phí quét 169 µs/chunk.

⚠️ Không giấu: **cả 14 chunk đều là biến công thức** — `ΔTC`, `εij`, `μg`,
`βPTAij`, `λij`. Corpus là báo cáo kinh tế lượng, và một biến Hy Lạp cạnh chữ
Latin *đúng là* trộn hệ chữ. Chấp nhận vì cờ chỉ **gắn nhãn** chứ không bỏ chunk
(`W4-12`), và 0,081 % vẫn dưới ngưỡng.

⚠️ **Phần dư có thật, và nó là chỗ một bảng vẫn đúng việc:** ký tự nhìn giống
Latin **và bản thân là chữ Latin** (`ɡ`, `ı`, `ɔ`, `ᴏ`) — một từ toàn Latin thì
không trộn hệ chữ nào cả. Khác tập xuyên-hệ-chữ (vô hạn), tập này **đếm được**:
nhóm ngữ âm ở Latin Extended-B/IPA. Thêm 27 mục vào bảng gập; giờ chúng khớp
đúng luật cụ thể (`override_instructions_en`) chứ không chỉ `mixed_script`.

⚠️ **Mẫu số của phép đo cũng là một giả thuyết.** Lượt đầu cho 85 % vì bộ sinh
mẫu thử lọc theo **tên** Unicode, nên nó kéo vào `COMBINING LATIN SMALL LETTER E`
(vẽ một chữ e tí xíu *phía trên* chữ bên cạnh) và `PARENTHESIZED LATIN SMALL
LETTER E` (vẽ `(e)`). Không cái nào lừa được mắt người, tức không cái nào là một
phép tấn công — mà cả hai vẫn nằm trong mẫu số. Lọc theo category (`Lu/Ll/Lo/Lt`)
xong mới ra 62/62. Tôi suýt ghi con số 85 % vào báo cáo.

⚠️ Con số chunk ở đây là **19.744**, không phải 20.424 của `W4-12`: hai lượt
không dùng cùng tham số chunking. Ghi ra thay vì lặng lẽ mượn con số cũ.

---

## 5. `SEC-03` — và một giả thuyết của tôi bị phép đo bác bỏ

`GET /admin/ingest/{job_id}` nối `job_id` thẳng vào URL của dịch vụ nội bộ.

Tôi viết ra giả thuyết: *`%2e%2e%2f%2e%2e%2f…` cho phép gọi tới đường dẫn tuỳ ý
trên dịch vụ ingest, gồm cả đường xếp job với `recreate=true`.* Đo trên router
thật thì **sai**: mọi thứ mang `%2f` bị chặn ở tầng định tuyến và handler không
bao giờ thấy.

Cái **thật sự** tới được handler:

| `job_id` | path đi ra | query đi ra |
|---|---|---|
| `%2e%2e` → `..` | `/` | — |
| `abc%3Fx%3D1` → `abc?x=1` | `/ingest/abc` | **`x=1`** |
| `x%00y` | *ném `httpx.InvalidURL`* | — |

Nên severity là **vừa, không cao**: lùi được **một** đoạn đường dẫn, và tiêm
được query string tuỳ ý vào lời gọi tới một dịch vụ nội bộ. Hôm nay
`GET /ingest/{job_id}` không đọc query nào nên tác hại gần 0 — nhưng đó là tính
chất của *dịch vụ kia*, không phải một hàng rào ở đây.

⭐ Ca `\x00` là lỗi thứ hai, khác loại: `httpx.InvalidURL` **không** kế thừa
`httpx.HTTPError`, nên nó xuyên qua `except` của `_call` và thành 500. Một phép
kiểm ở đầu route đóng cả hai bằng một dòng.

⚠️ Tập ký tự hợp lệ là **URL-safe**, không phải hex, dù id thật là `uuid4().hex`.
Hex chặt hơn mà **không an toàn hơn** — cả hai loại sạch `/ . ? # %` — đổi lại nó
ghim proxy vào định dạng id của một dịch vụ khác.

⚠️ Bài test chứng minh bằng **không có lời gọi mạng nào xảy ra**, chứ không bằng
mã trạng thái: một 422 đúng lý do và một 422 vì proxy đang tắt trông giống hệt
nhau từ ngoài. Và danh sách ca thử chỉ chứa những giá trị **thật sự tới được
handler** — đưa traversal nhiều đoạn vào sẽ là một test xanh nhờ router chứ
không nhờ bản vá.

---

## 6. `AU-10` — biến một quy ước triển khai thành một bất biến của mã

`AU-10` ghi: *"Giảm nhẹ hiện tại: bind `127.0.0.1`, Docker không expose."* Cả
hai vế đúng, và cả hai nằm **ngoài mã**: một cờ dòng lệnh và một dòng YAML. Một
`--host 0.0.0.0` gõ vội trong lúc gỡ lỗi xoá sạch chúng, không để lại gì trong
diff và không có gì đỏ.

`pipeline.ingest.app.guard`:

1. `INGEST_API_TOKEN` có đặt ⇒ mọi route (trừ `/healthz`) đòi đúng token ấy.
   So bằng `hmac.compare_digest` — khác `ApiKeyStore`, ở đó digest là *khoá tra
   dict* nên không có phép so nào chạy trên bí mật; ở đây có đúng một token và
   thời gian so **là** một kênh phụ.
2. Không đặt ⇒ **chỉ loopback gọi được**. Máy dev không phải cấu hình gì thêm,
   mà một lần bind ra ngoài cũng không mở được cửa.

⭐ Có token thì loopback **không** còn là đường tắt — ngược lại thì một tiến
trình bất kỳ trên cùng máy, kể cả một trang web mở trong trình duyệt của người
vận hành, vẫn gọi được.

⚠️ **14 bài integration đỏ cùng lúc** khi bản vá vào, và đó là bằng chứng nó thật
sự chặn: `TestClient` mặc định host `"testclient"`, **không** phải loopback. Các
test giờ phải khai `client=("127.0.0.1", …)` — tức phải nói ra mình gọi từ đâu.

---

## 7. Nợ đã đóng, và một nợ đóng nửa

**`TD-58` — kho khoá.** Nợ ghi "5 khoá, phần lớn là rác probe". Thực tế **11** —
nó lớn lên và không ai thấy, vì chưa có đường liệt kê. Đáng chú ý: hai khoá
`rate_limit_per_minute = 100000` do `W6-05` cấp cho load test, nằm trong kho
production. Thêm `list` + `revoke` (thu hồi theo `key_id`, không theo digest —
digest là thứ **không ai cầm**), dọn còn **3**: chat · admin · metrics.

⚠️ Nợ viết "mint lại đúng hai khoá"; đúng là **ba** — `W5-07` thêm Prometheus
scrape sau khi nợ được ghi.

⚠️⚠️ **Thu hồi chưa có hiệu lực cho tới khi server khởi động lại** (kho chỉ nạp
lúc khởi động, `W4-04`). CLI **in ra** điều đó, không chỉ ghi trong docstring:
một người vận hành vừa xoá một khoá bị lộ và tin rằng mình đã xong là tình huống
tệ hơn cả việc không có lệnh thu hồi.

**`TD-83` — smoke tầng sinh.** Nợ chuyển sang `W6-06` để đi cùng "phần quản lý
secret". Phần ấy **đã làm và đã đóng**: `ci.yml` (chạy trên `pull_request`) không
đọc `secrets.*` nào và giờ khai `permissions: contents: read`; `nightly.yml` giữ
secret nhưng chỉ trigger `schedule` + `workflow_dispatch` — không cái nào fork
điều khiển được; không workflow nào dùng `pull_request_target`. Có test đọc
**file workflow thật** ghim cả ba tính chất.

⚠️ Phần "thêm một tầng smoke gọi model" thì **không** đóng được ở đây, và lý do
mới: job `full-eval` của `nightly.yml` đang bị gác sau
`vars.NIGHTLY_GPU_RUNNER == 'on'` — chưa bật. Nối smoke sinh vào đó là nối vào
một job không chạy, tức một cổng trông như có mà không gác gì. Ba lý do gốc
(`TD-41` không tất định · fork không thấy secret · tốn tiền mỗi commit) vẫn
nguyên. Nợ ở lại `[ ]`, chỗ trả kế tiếp là khi có runner.

---

## 8. ⭐⭐ Danh sách phơi sáng cho `W6-02` — cái gì đang được che bằng loopback

Đây là phần dùng được ngay của lượt rà này. Gần như mọi hàng rào hạ tầng hôm nay
là **`127.0.0.1`**, và `W6-02` là lúc giả định ấy chết.

| thứ | hàng rào hôm nay | phải làm gì trước khi công khai |
|---|---|---|
| Qdrant `:6333` | bind loopback, `QDRANT_API_KEY` **rỗng** | không expose; nếu phải, đặt API key |
| Postgres `:5432` | bind loopback, mật khẩu `rag_local_dev_only` | không expose; đổi mật khẩu; role `rag_app` (không superuser) đã đúng |
| Redis `:6379` | bind loopback, **không mật khẩu** | không expose; `requirepass` nếu rời máy |
| Ingest API `:8001` | ~~chỉ bind~~ → **`guard`** (`AU-10`) | đặt `INGEST_API_TOKEN`, hoặc để `INGEST_API_URL` trống (mặc định) |
| Grafana `:3001` | `admin` / `admin_local_dev_only` | không deploy cùng; hoặc đổi mật khẩu |
| Langfuse `:3000` | khoá `sk-lf-rag-platform-local` trong YAML | dùng project riêng, khoá riêng |
| `/ready` | công khai — lộ **tên phiên bản bundle** | chấp nhận được sau LB; xem lại nếu ra Internet |
| `/docs`, `/openapi.json` | **cần khoá** | giữ nguyên |
| Hạn mức nhịp | chỉ tính **sau** xác thực (`TD-39`) | request không khoá đi qua tự do ⇒ cần chặn theo IP ở reverse proxy |
| Kích thước thân request | không trần (ngoài trần từng trường) | đặt `client_max_body_size` ở proxy |

⚠️ Dòng cuối là lỗ duy nhất `W6-06` **không** đóng được trong mã: Pydantic chỉ
thấy thân request **sau** khi nó đã được đọc trọn vào bộ nhớ. `SEC-04` đóng trục
`filters`, nhưng trần thật phải ở tầng trước ứng dụng.

---

## 9. Đã kiểm, ổn

Ép tenant ba tầng (token → `tenant_filter()` → RLS Postgres, và `set_config`
tham số hoá dù `tenant_id` đến từ token) · so khoá API bằng tra dict trên digest,
không có phép so nào chạy trên bí mật · SSE không tiêm được tên event, `data`
luôn qua `json.dumps` · không `eval`/`exec`/`pickle`/`yaml.load` không an toàn ·
không `verify=False` · `secrets` chứ không `random` cho nonce và khoá · không
file nhạy cảm nào bị git theo dõi, và lịch sử git sạch (chuỗi duy nhất trông như
khoá là một fixture test) · mọi cổng compose bind `127.0.0.1` · UI không bao giờ
`innerHTML`, CSP `default-src 'none'`, và API dùng header `Authorization` chứ
không cookie ⇒ CSRF không áp dụng · không CORS middleware nào được cài.

---

## 10. Tiêm lỗi: 32/32 đỏ — sau ba lượt

| lượt | đỏ | sống | cái mà lượt ấy dạy |
|---|---|---|---|
| 1 | 20/30 | 10 | ba tầng test bị bỏ sót, và **hai lỗi thật trong bản vá của tôi** |
| 2 | 27/30 | 3 | hai bộ test lấy danh sách kiểm từ chính thứ nó kiểm |
| 3 | **32/32** | 0 | — |

Bốn lỗ đáng ghi:

* **Bộ chấm không chạy tầng integration** ⇒ ba phép tiêm phân trang (`M16`–`M18`)
  sống sót. Một lượt tiêm bỏ sót cả một tầng test trông y hệt một lượt tiêm
  không tìm ra gì — họ "đỏ giả" của `NEW-08`, ngược chiều.
* **Test tham số hoá theo `SECRET_PATTERNS`** ⇒ xoá một luật khỏi bảng thì ca thử
  của nó **cũng biến mất khỏi tham số hoá**, không có gì đỏ. Một bộ test lấy
  danh sách kiểm từ chính thứ nó đang kiểm thì không kiểm được việc *xoá*. Đổi
  sang tham số hoá theo danh sách của **test**, cộng một phép kiểm hai chiều.
* **Mẫu thử chạm nhiều luật cùng lúc** ⇒ hai luật che cho nhau. `Authorization:
  Bearer rag_…` chạm cả `bearer_token` lẫn `platform_api_key`, nên xoá luật nào
  cũng vẫn "đã che". Giờ mỗi luật có một mẫu **chỉ nó** bắt được, và test chứng
  minh điều đó bằng cách **bỏ chính luật ấy ra** rồi đòi mẫu thử không còn được
  che.
* **Nới `PUBLIC_PATHS` không bị bắt** ⇒ mọi phép kiểm khác đọc `PUBLIC_PATHS` làm
  chuẩn, nên nới nó ra là nới luôn cả cái thước. Thêm một bản sao **có chủ đích**
  của allow-list trong test: lý lẽ ngược với `AU-12` vì giá trị của một
  allow-list nằm ở chỗ nới nó phải là **hai chữ ký ở hai file**.

Và hai lỗi thật trong bản vá (`keep` là mã chết vì thứ tự luật; `exc_text` được
che nhờ vòng lặp `__dict__` chứ không nhờ dòng của chính nó) — cả hai đã sửa,
chi tiết ở §2. Cộng một dòng phòng-thân `normalize("NFC", …)` mà phép tiêm chứng
minh là **không thể** khác kết quả, nên nó bị xoá: giữ một dòng không chứng minh
được là để lại cho người sau một câu hỏi không có đáp án.

---

## 11. Đo cuối

* **72 test mới** — 61 (`tests/security/test_w606_hardening.py`) + 8
  (`test_route_authz.py`) + 2 ca luật mới trong `test_no_secret_in_job_bundle.py`
  + 1 integration phân trang
* **2 501 xanh** bộ mặc định (3 skip) · **313 xanh** integration
* ruff · mypy · `mypy --platform linux` sạch
* **32/32** phép tiêm đỏ
* Chi phí **$0**

**Evidence**: `probes/w606-log-redaction.json` ·
`probes/w606-td52-confusables.json` · `scripts/probe_log_redaction.py` ·
`scripts/confusable_probe.py`
