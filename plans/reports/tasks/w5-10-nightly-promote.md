# `W5-10` — eval đêm + đường phát hành tự động, và lần chạy thật đầu tiên của nó là một lần **từ chối**

> Gộp ba món đã hẹn trả cùng chỗ: `TD-71` (phán quyết gate không nằm trong
> bundle), `AU-12` (smoke hardcode số hiệu bundle), `TD-82` (vân tay index phụ
> thuộc hệ điều hành). Cả ba đụng cùng một câu hỏi — *"bundle nào đang phục vụ,
> và ai nói thế"* — nên tách ra làm ba lần là ba lần đọc lại cùng một chỗ.

---

## 0. Kết quả một dòng

Có một job đêm chạy được: nó chọn ứng viên, gọi gate, và **chỉ khi PASS** mới
đúc một bundle phát hành đã ký, dời con trỏ `CURRENT`, rồi gói cả hai vào một PR
cho người ký cuối. Lần chạy thật đầu tiên trên bundle thật cho **`INCOMPARABLE`,
exit 2, không đề nghị gì** — và đó là kết quả đúng, không phải một lần chạy hỏng.
Nhánh PASS được diễn tập riêng trên bản sao đĩa thật để nó không phải là một
nhánh chưa ai đi qua.

| | |
|---|---|
| Lần chạy thật | `make nightly` → `INCOMPARABLE`, exit 2, 12 PASS / 5 FAIL / 3 SKIP |
| Diễn tập nhánh PASS | ứng viên 0.3.0 → đúc **0.3.1**, con trỏ `0.2.1 → 0.3.1`, `gate.status=PASS` nằm **trong** manifest |
| Test mới | 39 (`test_nightly.py` 37 + `test_index_config.py` 2) + 3 bài viết lại ở `test_build_bundle.py` |
| Tiêm lỗi | **23/23 đỏ** — lượt một 21/23, hai phép sống sót đều là **lỗ trong test** |
| Chi phí | **$0** (không gọi model nào) |

---

## 1. ⭐⭐ `AU-12` không phải một hằng số gõ sai — nó là **bản sao thứ hai của một sự thật**

`pipeline/eval/smoke.py` viết `DEFAULT_BUNDLE = Path("bundles/rag-bundle-v0.2.1/manifest.json")`,
còn CI lẫn Makefile không truyền `--bundle`. Đọc riêng thì nó vô hại. Đặt cạnh
`serving/api/app.py`, nơi bundle đang phục vụ được **suy ra** bằng "bản semver
cao nhất", thì hai bên là hai câu trả lời độc lập cho cùng một câu hỏi.

Bump bundle lên `0.3.0` và quên sửa hằng số ⇒ cổng PR vẫn xanh, và thứ nó gác là
`components.retrieval.options` của **bản cũ** — đúng loại thay đổi mà
`retrieval_options` (`W5-09`) sinh ra để bắt. Bug ở tầng meta: nó không làm hỏng
một truy vấn nào, nó làm hỏng **niềm tin vào cổng**. Và nó không tự lộ ra được,
vì triệu chứng của nó là màu xanh.

### Con trỏ, chứ không phải một hằng số thứ ba

Sửa bằng cách xoá bản sao: `bundles/CURRENT` chứa đúng một dòng version, và cả
`_startup_version` lẫn `default_bundle()` đọc nó.

Việc dựng con trỏ phơi ra thêm **hai** lỗi mà `AU-12` không nêu, cùng một gốc —
suy luận `latest_bundle`:

1. **`save_bundle` là một lần deploy.** Đúc một release candidate để chạy gate —
   thứ job đêm này làm mỗi đêm — đủ để đổi cái serving nạp ở lần restart kế
   tiếp, kể cả khi bundle ấy vừa **trượt** gate.
2. **Rollback không sống qua restart.** `POST /admin/bundle` đổi bundle trong bộ
   nhớ; lần khởi động sau, `latest_bundle` lại chọn bản cao nhất và lặng lẽ huỷ
   kết quả của lần rollback.

Nhánh suy luận vẫn còn — test và môi trường dev chưa có con trỏ — nhưng nó
**khai ra là mình đang đoán** bằng một `logger.warning` có test ghim. Còn cổng PR
thì **không** có nhánh dự phòng: nó tồn tại để gác đúng cấu hình đang phục vụ,
nên đoán là hỏng đúng cái việc của nó.

> 💡 Một phép kiểm chống được đúng lớp lỗi này:
> `test_serving_and_the_pr_gate_read_the_same_pointer` dựng cảnh có một bundle
> **mới hơn** bản đang trỏ tới. Bên nào quay về suy luận theo semver thì hai giá
> trị rời nhau ngay.

---

## 2. ⭐⭐ `TD-71` — phán quyết gate phải nằm **trong** bundle, và cách sai để làm điều đó

`RagBundle.gate` có từ `W4-01`, và tới hết `W5-09` luôn là `NOT_RUN`: phán quyết
sống trong `gate-*.json` cạnh bundle. Hệ quả đúng như `TD-71` ghi — không có gì
để một đường promote tự động **kiểm** trước khi đổi con trỏ.

Cách hiển nhiên: mở manifest, điền `gate`, ký lại. Nó sai, và lý do nằm ngay
trong docstring của `store.py`: luật số 1 là *không ghi đè một version đã tồn
tại*, vì câu "số đo này thuộc về `v0.2.1`" chỉ có nghĩa khi `v0.2.1` không đổi
được. Một manifest sửa được sau khi ký thì chữ ký chỉ còn là trang trí.

Nên `promote()` **đúc một bản patch mới**: cùng components, cùng eval, khác đúng
ba thứ — số hiệu, phán quyết, và một dòng `notes`.

### ⭐⭐ `git_sha` là trường dễ ghi đè nhất, và ghi đè nó là mất đường truy nguyên

Phản xạ là ghi commit của lần promote. Nhưng schema nói rõ trường ấy là *"đường
duy nhất đi ngược từ một artifact đang chạy về mã đã tạo ra nó"* — và mã tạo ra
**những con số** trong bundle này là commit của ứng viên, không phải commit của
cái đêm mà một job CI dời con trỏ. Ghi đè nó là đổi một trường **truy nguyên**
lấy một trường **nhật ký**, im lặng. Commit lúc promote vẫn được ghi — ở `notes`,
nơi nó đúng là nhật ký. Mutation `M16` ghim chuyện này.

### ⭐ Ba cách một bản phát hành bị dán dấu PASS nhầm

`promote()` từ chối cả ba, và cả ba đều trông bình thường trong log:

| | tình huống | vì sao nguy hiểm |
|---|---|---|
| 1 | phán quyết không PASS | hiển nhiên — nhưng phép kiểm phải nằm ở **hàm ghi đĩa**, không chỉ ở nhánh `if` của người gọi |
| 2 | `evaluate_gate(a, …)` rồi `promote(b, verdict)` | dán một dấu PASS lên artifact **chưa ai chấm** |
| 3 | PASS mà `champion is None` | `--no-champion` bỏ hẳn nhóm luật hồi quy: chưa ai hỏi "có tụt so với bản đang chạy không" |

### ⭐ `model_validate`, không phải `model_copy`

`model_copy` bỏ qua toàn bộ validator — cái bẫy mà `tests/unit/test_bundle.py`
đã ghi lại một lần cho fixture. Ở đây hậu quả nặng hơn: đường này **ghi ra đĩa và
ký**, nên một bundle mà schema sẽ từ chối vẫn ra được một manifest có chữ ký hợp
lệ. Chỗ hai đường tách nhau **quan sát được**: payload là JSON, `created_at` là
chuỗi ISO — `model_validate` ép về `datetime`, `model_copy` nhét thẳng chuỗi vào
một trường khai kiểu `datetime` rồi trả về một object nửa đúng kiểu, vẫn ghi
được, vẫn đọc lại được. Không phép kiểm nào ở phía đĩa bắt được, nên bài test
phải kiểm **object trả về** (`M18`).

---

## 3. ⭐⭐ `TD-82` — sửa một hàm băm là sửa **mọi artifact đã ghi bằng nó**

`IndexConfig.fingerprint` băm cả `contexts_path`, và `Path` serialise ra
`data\contexts\…` trên Windows / `data/contexts/…` trên POSIX. Cùng một config,
hai vân tay — trên đúng cái trường tồn tại để chứng minh *"index này được build
bằng config này"*. Vô hình suốt `W1`…`W5-08` vì mọi lần build đều trên một máy;
lượt CI đầu trên Linux (`W5-09`) làm nó lộ ra.

Bản vá là một `field_serializer` trả `as_posix()`. Phần khó không nằm ở đó.

### Ba lựa chọn, và vì sao chọn cái thứ ba

| | cách | hệ quả |
|---|---|---|
| 1 | sửa công thức, **đúc lại ba manifest** (dự tính gốc của `TD-82`) | sửa một trường bên trong artifact **đã ký** để nó khớp mã hôm nay — biến chữ ký thành trang trí, phá đúng luật số 1 của `store.py` |
| 2 | sửa công thức, để nguyên artifact cũ | mọi vân tay đã ghi thành "không khớp": lần `make index` kế tiếp bị chặn, `build_bundle` từ chối đóng gói số đo hợp lệ. Bản vá một lỗi **im lặng** tự tạo ra một lỗi **ồn ào** |
| 3 | **sửa công thức + biết đọc công thức cũ** | artifact cũ vẫn nhận diện được, vẫn dùng được, và được nâng cấp tại chỗ ở lần ghi kế tiếp |

`fingerprint_status(recorded) → "current" | "legacy" | "mismatch"`. Cố ý **không**
trả `True`/`False`: "chấp nhận nhưng nói ra" chỉ đúng cho đường **đọc lại**
artifact cũ, không đúng cho đường **ghi** artifact mới, nên người gọi phải tự
quyết cảnh báo hay dừng.

`legacy_windows_fingerprint` dùng `PureWindowsPath`, thứ viết dấu `\` trên **cả**
Windows lẫn Linux. Nhờ vậy ba manifest đã commit được nhận là `"legacy"` trên
**mọi** hệ điều hành — và bài
`test_the_sample_bundle_matches_the_real_index_config`, trước đây bị
`skipif(os.name != "nt")`, giờ chạy ở cả hai nơi. Nửa số máy chạy CI vừa lấy lại
được một phép kiểm.

Chấp nhận thôi chưa đủ: `_reconcile_state` **nâng** state lên vân tay mới, nếu
không thì dòng cảnh báo ấy vĩnh viễn và giá trị cũ không bao giờ biến mất
(`M5`).

### Bài test thay cho bài ghim nợ

Bài cũ (`test_the_index_fingerprint_is_still_path_separator_dependent`) đọc
`inspect.getsource(fingerprint)` và đỏ vào đúng ngày ai đó sửa — đúng thiết kế,
nhưng sau đó nó không gác được gì, và nhất là nó **mù với trường path tiếp
theo**. Bài mới quét chính payload đi vào hàm băm và từ chối mọi giá trị chứa
`\`. Thêm một `Path` vào `fingerprint` mà quên chuẩn hoá ⇒ đỏ **trước khi**
artifact nào kịp mang giá trị lệch nền tảng.

> ⚠️ Còn nợ: ba manifest đã ký vẫn mang giá trị `legacy`. Chúng chỉ lên
> `current` khi có một lần build index thật đúc ra bundle mới. `TD-82` giữ mở
> với phạm vi thu hẹp lại đúng câu đó.

---

## 4. Vòng đêm: ba câu hỏi tách rời

1. **Có ứng viên mới không?** — bundle semver cao hơn bản `CURRENT` trỏ tới.
2. **Ứng viên có qua gate không?** — gọi `pipeline.eval.gate`, không sao chép
   lại một luật nào. `nightly.py` **không biết** ngưỡng là bao nhiêu, và đó là
   điều kiện để nó không trở thành bản sao thứ hai của gate.
3. **Nếu qua thì đề nghị gì?** — bundle đã ký + con trỏ đã dời, gói trong một PR.

### ⭐⭐ Champion mặc định là bản **đang phục vụ**, không phải bản kề dưới

`gate._pick_champion` chọn bản semver cao nhất thấp hơn ứng viên. Đúng cho một
lần chấm thủ công. Sai cho một quyết định phát hành: câu hỏi ở đây là *"có tụt so
với thứ người dùng đang nhận không"*, và **sau một lần rollback thì thứ người
dùng đang nhận không phải bản kề dưới**. Cảnh cụ thể: đã rollback về `1.0.0`
trong khi `1.1.0` vẫn nằm trên đĩa, rồi `1.2.0` xuất hiện — mặc định của gate sẽ
so với `1.1.0`, bản không ai đang dùng. Hai hàm, hai câu hỏi, nên nightly truyền
champion tường minh (`M20`).

### ⭐ PASS mà vẫn không phát hành

Ứng viên **chính là** bản đang chạy ⇒ dừng. Không có phép kiểm này thì mỗi đêm
sinh một patch mới dán dấu PASS lên đúng hệ thống đêm trước: số hiệu tăng, thông
tin không (`M21`).

### ⭐ Bốn mã thoát, không phải ba

`0` PASS · `1` FAIL · `2` INCOMPARABLE · **`3` promote bị từ chối**. Gộp `3` vào
`1` là nói sai việc phải làm: FAIL nghĩa là *hệ thống* chưa đủ tốt, `3` nghĩa là
hệ thống đủ tốt mà **đường phát hành** hỏng. Hai người khác nhau phải thức dậy.

### ⭐ "Không đề nghị gì" cũng sinh artifact

Một job đêm im lặng khi không có việc là một job không phân biệt được với một job
đã chết. Mọi lượt chạy ghi `nightly-*.json` + một thân PR đọc được, kể cả lượt
kết luận "không có ứng viên".

---

## 5. Workflow: cái chạy được hôm nay và cái chưa

`nightly.yml` tách khỏi `ci.yml` vì nó vi phạm cả bốn tính chất của cổng chặn PR:
nó gọi model, cần GPU, cần secret, và ghi vào repo.

**Tầng 1 `full-eval`** cần nhãn runner `self-hosted, gpu` — chưa có máy nào mang
nhãn ấy. Để nguyên thì job **treo ở hàng đợi** thay vì đỏ, thứ tệ hơn cả đỏ vì nó
trông giống "đang chạy". Nên nó bị chặn sau `if: vars.NIGHTLY_GPU_RUNNER == 'on'`
và lịch sử Actions ghi **"skipped"**.

**Tầng 2 `gate-and-promote`** chạy trên `ubuntu-latest`, không GPU, không secret
— nó đọc số đo đã nằm trong bundle. Chạy được ngay hôm nay. Chủ ý: **đường phát
hành phải chạy được trước khi có máy đo**, nếu không thì lần đầu nó chạy sẽ là
lần đầu nó được thử.

Ba chi tiết có test ghim, vì cả ba là những cách đường phát hành biến mất im
lặng:

* `if: always() && needs.full-eval.result != 'failure'` — một `needs` bị **skip**
  làm job phụ thuộc skip theo, tức toàn bộ đường phát hành biến mất vào đúng ngày
  chưa có runner GPU. Vẫn chặn nếu tầng 1 chạy và **đỏ**.
* Bước mở PR gác bằng `git diff -- bundles/`, **không** bằng exit code: exit 0
  cũng xảy ra khi "không có ứng viên mới", và một PR rỗng mỗi đêm là cách nhanh
  nhất để mọi người tắt thông báo.
* `cancel-in-progress: false` — một lượt eval đêm bị cắt ngang để lại artifact dở
  và không để lại phán quyết nào.

### Vì sao PR chứ không push thẳng

Gate là một phán quyết dựa trên số đo. Nó **không** biết corpus có vừa đổi
không, có sự cố nào đang mở không, có thay đổi vận hành nào phải đi cùng bundle
không. PR để nguyên phán quyết ấy làm bằng chứng và để chữ ký cuối cho người:
chi phí một cú click, đổi lại một đường phát hành tin được lúc 3 giờ sáng.

---

## 6. Lần chạy thật — và vì sao `INCOMPARABLE` là kết quả đúng

```
$ make nightly
INFO ứng viên = 0.2.1 (đang phục vụ: 0.2.1)
INFO không đề nghị phát hành: gate INCOMPARABLE — không đề nghị phát hành
exit 2
```

Năm luật trượt (`plans/reports/runs/nightly-20260906.json`):

| nhóm | luật | chi tiết |
|---|---|---|
| comparability | generator là bí danh | `deepseek-chat@2026-09` — `TD-70`, chưa trả |
| comparability | `evaluated_with_generator` | ứng viên `deepseek-v4-flash` ≠ champion `deepseek-chat@2026-09` |
| comparability | `judge_identity` | champion `0.2.0` không mang judge |
| absolute | `citation_accuracy` | `0.8308 < 0.85` — số **trong manifest**, chưa phải `0,8662` của `NEW-08` |
| absolute | `p95_end_to_end_ms` | `4706,5 > 3500` |

Hành động đầu tiên của một đường phát hành tự động là một lần **từ chối**, và nó
từ chối vì đúng lý do: hai lần đo chưa đặt cạnh nhau được thì mọi so sánh phía
sau đều rỗng. Hai ô `absolute` trượt cũng đã có chủ: `citation_accuracy` chỉ lên
`0,8662` khi có một bundle đúc từ lần chấm lại của `NEW-08`; `p95` là đòn bẩy
tầng sinh, thuộc `W5-11`.

## 7. Diễn tập nhánh PASS — vì một nhánh chỉ từng thấy ở trạng thái từ chối là một nhánh chưa ai biết có chạy không

Đây là mặt kia của bài học `G5` ở `W5-09` ("một cổng chỉ từng được nhìn ở trạng
thái xanh"). Bản sao `bundles/` trong thư mục tạm, ứng viên `0.3.0` mang
`p95=3100` **bịa có nhãn**, chạy đúng CLI thật —
`plans/reports/probes/w5-10-promote-rehearsal.json`:

```
ứng viên 0.3.0 → đúc 0.3.1 · con trỏ 0.2.1 → 0.3.1 · exit 0
gate trong manifest: PASS vs champion 0.2.1
git_sha 87f912b8ae — GIỮ NGUYÊN của ứng viên
checksum: sha256:08c1a6ab537cc536e…
notes: "…Ghi chú của ứng viên: DIỄN TẬP — số đo tầng sinh ở đây là BỊA…"
```

Dòng cuối là thứ đáng giá nhất trong lượt diễn tập: `notes` của ứng viên được
mang theo vào bản phát hành, nên cảnh báo "đây là số bịa" **đi cùng artifact**
thay vì ở lại trong script sinh ra nó.

---

## 8. Test và tiêm lỗi

39 bài mới + 3 bài viết lại. Tiêm **23 phép, 23 đỏ** — nhưng lượt một là
**21/23**, và hai phép sống sót đều là **lỗ trong test**, không phải trong mã:

* **`M9`** `current_bundle` fallback về `latest_bundle` khi thiếu con trỏ. Bài
  test chạy trên thư mục **rỗng**, nên `None` là câu trả lời của cả đường đúng
  lẫn đường sai. Thêm bundle vào thư mục thì hai đường tách nhau.
* **`M22`** hạ `EXIT_PROMOTION_REFUSED` từ `3` xuống `1`. Bài test viết
  `assert result.exit_code == EXIT_PROMOTION_REFUSED` — **đọc chính hằng số bị
  đổi**, nên nó xanh với mọi giá trị. Sửa thành số viết thẳng, cộng một câu về
  **quan hệ**: mã ấy không được trùng mã nào của gate.

> 💡 Cùng một họ lỗi ở hai hình dạng: một phép kiểm chỉ đo được thứ gì đó khi
> đường đúng và đường sai cho **hai** kết quả khác nhau. Thư mục rỗng và hằng số
> tự tham chiếu đều xoá mất sự khác nhau ấy.

Sau khi bịt: tiêm lại `M9`/`M22` → **cả hai đỏ**.

## 9. Đo cuối

| | |
|---|---|
| Bộ mặc định (tầng CI) | **2 232 xanh**, 2 skip |
| `ruff check` · `ruff format` · `mypy` · `mypy --platform linux` | sạch |
| Tiêm lỗi | 23/23 đỏ |
| Chi phí | **$0** |

## 10. Nợ

**Trả xong**: `TD-71` (phán quyết nằm trong bundle, có promote đúc nó ra) ·
`AU-12` (một nguồn sự thật cho bundle đang phục vụ).

**Thu hẹp, còn mở**: `TD-82` — công thức đã sửa, đường đọc lại đã có, còn lại
đúng một việc: ba manifest đã ký lên `current` ở lần build index thật kế tiếp.

**Chưa trả, có chỗ trả**: `TD-83` (smoke tầng sinh) → `W5-11`, vì một cổng tầng
sinh phải dùng **kiểm định** chứ không ngưỡng tuyệt đối (`TD-41`), và máy kiểm
định cần **hai** nhánh model để so — thứ `W5-11` mới có. Ghi thẳng vào
`nightly.yml` chỗ nó sẽ nằm. · `TD-81` (smoke không phủ rerank) — nightly là lối
(b) đã nêu, mở được khi có runner GPU. · `TD-70` (bí danh generator) là lý do
`comparability` trượt hôm nay.

**Mới**: `TD-85` — `nightly.yml` chưa từng chạy trên GitHub. Không có runner GPU
cho tầng 1 và không có credential để `workflow_dispatch` tầng 2 từ máy này; YAML
được gác bằng 7 bài test đọc chính file ấy, nhưng đọc file không phải chạy nó —
cùng loại giới hạn đã ghi ở `G5` của `W5-09`.
