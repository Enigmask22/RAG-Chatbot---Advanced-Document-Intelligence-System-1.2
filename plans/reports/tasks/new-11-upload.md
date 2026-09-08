# `NEW-11` — upload được CHO PHÉP, bằng cách bắt nó đi qua cửa đã có

*2026-09-08 · `pipeline/ingest/upload.py` (mới) + `app.py` + `serving/api/ingest.py` + UI · 22 test unit + 4 proxy + 3 integration · tiêm **11/11 đỏ** · chi phí **$0***

> **Nợ `NEW-11`:** *"Chốt chuyện tải tài liệu lên: cho phép, hay không bao giờ.
> Đây là một quyết định về license trước, endpoint sau: ai chịu trách nhiệm cho
> tài liệu người dùng đẩy vào, nó có được trộn vào cùng collection với corpus
> công khai không, và câu trả lời trích dẫn nó thì trích dẫn cái gì."*
> Người dùng chốt 08/09/2026: **cho phép**.

## 1. Câu trả lời không phải "thêm một ngoại lệ" — là "không có ngoại lệ"

Nỗi sợ của dòng nợ là nút tải-lên mở đúng con đường mà quy tắc cứng *"corpus
phải công khai, license cho phép redistribute"* sinh ra để đóng. Thiết kế được
chọn đóng nỗi sợ ấy bằng cấu trúc: upload **không phải** một đường tắt vào
prompt — nó là *cửa nhận tài liệu của Pipeline Plane*, và tài liệu tải lên đi
qua **đúng cái cửa** mà 60 tài liệu World Bank đã đi:

* `license` phải nằm trong `LICENSE_ALLOWLIST` (ND bị từ chối với đúng lý do
  đã viết ở `W0`: chunking + sinh context là tác phẩm phái sinh);
* `source_url` **công khai là trường bắt buộc** — nghĩa của upload ở đây là
  *"thêm một tài liệu công khai vào sổ đăng ký"*, không phải "đẩy file riêng";
* entry đi vào **cùng manifest**, qua `CorpusEntry` + `validate_manifest`
  (trùng nội dung = 409, không phải một bản sao lặng lẽ).

**Ai chịu trách nhiệm** thì manifest phải nói được: proxy `/admin/ingest/upload`
đóng dấu `uploaded_by = tenant:key` từ principal **đã xác thực**; client tự
khai trường ấy bị `extra="forbid"` từ chối — một danh tính do kẻ gửi chọn không
được phép thành sử liệu.

## 2. ⭐⭐ Vì sao tài liệu tải lên KHÔNG được ghi thẳng vào collection đang phục vụ

`TD-38` kiểm `n_chunks` của collection so với bundle manifest. Ghi thêm điểm
vào `rag_bgem3_ctx` làm con số ấy lệch ⇒ lần `/admin/bundle/reload` kế tiếp
**từ chối** — và đó là hành vi đúng, không phải trở ngại: mọi con số eval trong
manifest nói về một corpus không còn tồn tại. Nên đường sản xuất là:

```
upload (đăng ký) → job ingest (doc_ids) → build bundle mới → gate → promote
```

Tài liệu tới người dùng **qua một bundle đã đo** — đúng luận đề trung tâm của
cả kiến trúc. Điều này được viết vào docstring, CHECKLIST và chú thích trên
form UI, vì nó là chỗ người vận hành sau này dễ "sửa cho tiện" nhất.

## 3. Một danh sách license, ba người dùng, và phép ghim bằng quan hệ

Serving không được import pipeline (`test_architecture_boundaries`), nên
`LICENSE_ALLOWLIST` dời về `rag_core.schemas` — nó là **từ vựng chung hai
plane** — và `pipeline.corpus.manifest` re-export giữ nguyên mọi câu import cũ.
Ba chỗ dùng (validator service, validator proxy, mảng `LICENSES` của form) ghim
với nhau bằng **test quan hệ** (họ `NEW-13`): bài test parse mảng JS từ
`index.html` và so **bằng tập hợp** với allowlist; trần 512 KiB tồn tại hai lần
(hai plane) và một bài test khẳng định chúng bằng nhau — mutant "proxy nới trần
lên 1 MiB" chết vì đúng bài này.

Vì sao proxy kiểm license *lần nữa*: `_call` che thân lỗi của dịch vụ trong
(`AU-03`), nên nếu chỉ dịch vụ trong kiểm thì người dùng nhận một 4xx câm.
Kiểm sớm ở proxy là cách duy nhất câu "giấy phép nào được nhận" tới được form.

## 4. Trần byte, không phải trần ký tự

`BodyLimitMiddleware` (`NEW-12`) đếm **byte**; tiếng Việt có dấu là 2–3
byte/ký tự. Một trần đếm ký tự để lọt request 600 KB rồi để nó chết ở
middleware với thông điệp về *thân request* thay vì về *tài liệu*. Validator đo
`len(content.encode("utf-8"))`; mutant `M2` (đếm ký tự) chết vì bài 200.000 chữ
"ạ" = 600.000 byte.

## 5. Cơ học ghi: file trước, manifest sau, manifest nguyên tử

Hai lần ghi không có transaction chung. File ghi trước — hỏng ở bước hai để lại
một file mồ côi vô hại (mọi đường đọc đi qua manifest); chiều ngược lại để lại
một entry trỏ vào hư không và `iter_documents` nổ cho **mọi** lượt ingest sau.
Manifest ghi qua `.tmp` + `os.replace` vì worker có thể đang đọc nó ở tiến
trình khác. Cả ba tính chất có mutant riêng (M5 dọn-mồ-côi, M6 nguyên tử, M4
manifest-rỗng-thì-từ-chối — một config gõ nhầm không được sinh manifest song
song).

## 6. Tiêm lỗi: 11/11 đỏ

Phủ cả bốn tầng: service (M1–M7), proxy (M8 quên đóng dấu danh tính, M9 trần
trôi), app (M10: 409 thành 400 — chỉ integration thấy), UI (M11: mảng license
thiếu một mục — test quan hệ bắt).

⚠️ **Giới hạn nói ra**: (a) chỉ nhận `.txt`/`.md` UTF-8 — PDF chờ Docling
(`W3-01`); (b) file tải lên chưa nằm trong DVC — bước `dvc add` thuộc quy trình
promote bundle, ghi ở đây thay vì tự động hoá vì promote là đường có người
duyệt; (c) với config contextual, chunk mới chưa có ngữ cảnh dán — dưới trần
`min_coverage 0.99` khi tài liệu nhỏ, và build **tự dừng** khi vượt, đúng thiết
kế của `W3-04`.
