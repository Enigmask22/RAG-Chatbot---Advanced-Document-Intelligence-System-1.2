# `TD-56` — sửa câu chữ cho khớp số đo, không sửa số đo cho vừa câu

*2026-09-08 · chỉ chạm `plans/CHECKLIST.md` · 0 dòng mã · chi phí **$0** · người dùng duyệt câu chữ mới*

> **Nợ `TD-56`:** *"Câu 1 của `G4` viết 'clone sạch ≤ 5 phút' nhưng số đo nói
> ~40 giây (cache ấm) / ~10+ phút (sạch thật)."*

## 1. Vì sao đây là một quyết định của người dùng, không phải một edit

Đổi lời một gate là **dời cột gôn** — cùng loại hành vi mà dòng end-to-end
3.500 ms bị giữ ở ❌ để tránh. Nó chỉ hợp lệ khi: (a) số đo có trước và không
bị đụng tới, (b) câu mới **hẹp hơn** thực tế đo được chứ không rộng hơn, và
(c) người đặt gate duyệt. Cả ba thoả: số của `W4-13` giữ nguyên (~40 giây ấm /
~10 phút sạch), câu mới "≤ 5 phút **khi cache đã ấm**" là mệnh đề mà số đo
*thừa sức* đỡ (dư 7,5×), và người dùng chốt 08/09/2026.

## 2. Cái sai của câu gốc nằm ở đâu

"Từ clone sạch ≤ 5 phút" viết ở `W1`, trước khi biết đồng hồ của một clone
sạch bị chi phối bởi hai thứ **không nén được**: build image 7,35 GB (~10
phút, phần lớn là wheel torch CUDA 2,5 GB) và tải 4,4 GB trọng số model — cả
hai là chi phí *một lần cho mỗi máy*, không phải chi phí của mỗi lần dựng.
Một gate trộn chi phí một-lần vào chi phí lặp lại thì hoặc không bao giờ ✅
(giữ 🟨 vĩnh viễn cho một hệ đã chạy tốt), hoặc bị "làm tròn lên". Câu mới
tách hai thứ đó ra và **ghi rõ chi phí lần đầu ngay trong gate** — người sau
đọc gate vẫn biết sự thật đầy đủ.

## 3. Hệ quả trên sổ

* Tiêu chí 1 của `G4`: `[~]` → `[x]`, kèm khối sử liệu giữ nguyên văn câu gốc
  và lý do nó bị bác — tick không xoá lịch sử.
* `G4` **✅ 3/3** · bảng tổng quan `W4 → G4 ✅` · tally gate **4/6**.
* README không đổi: hai bản chưa từng khai mệnh đề thời gian này (đã kiểm
  bằng grep cả "5 phút/5 min/clone sạch/clean clone").

⚠️ **Giới hạn nói ra**: "≤ 5 phút khi cache ấm" đo trên đúng một máy (máy dev,
image + `HF_HOME` sẵn). Một máy CI/máy mới toanh là một clone sạch — gate này
cố ý *không* hứa gì cho nó ngoài dòng ghi chú chi phí một-lần.
