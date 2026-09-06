# Eval đêm — không đề nghị phát hành

**Lý do**: gate INCOMPARABLE — không đề nghị phát hành

## Gate: **INCOMPARABLE** (exit 2)

Ứng viên `0.2.1` · champion `0.2.0` · 12 PASS / 5 FAIL / 3 SKIP
Ngưỡng: `configs/eval/gate.yaml`

### Luật trượt

- ❌ **generator không phải bí danh** (comparability) — 'deepseek-chat@2026-09' là bí danh: 'deepseek-chat' hiện được phục vụ bởi 'deepseek-v4-flash' (đo ở `W5-03`) và sẽ đổi khi nhà cung cấp ra bản mới. Hai bundle cùng ghi chuỗi này có thể đã đo bằng hai model khác nhau.
- ❌ **evaluated_with_generator** (comparability) — ứng viên 'deepseek-v4-flash' ≠ champion 'deepseek-chat@2026-09'
- ❌ **judge_identity** (comparability) — ứng viên 'deepseek-v4-flash|judge-answer-relevancy@v1,judge-faithfulness@v2|reasoning=false' ≠ champion '<không có judge>'
- ❌ **citation_accuracy** (absolute) — 0.8308 < 0.85
- ❌ **p95_end_to_end_ms** (absolute) — 4706.5000 > 3500.0

> ⚠️ `INCOMPARABLE` bảo sửa **phép đo**, không phải sửa hệ thống. Hai lần đo chưa đặt cạnh nhau được thì mọi so sánh phía sau đều rỗng.

<details><summary>Toàn bộ luật</summary>

| | nhóm | luật | chi tiết |
|---|---|---|---|
| ❌ | comparability | `generator không phải bí danh` | 'deepseek-chat@2026-09' là bí danh: 'deepseek-chat' hiện được phục vụ bởi 'deepseek-v4-flash' (đo ở `W5-03`) và sẽ đổi khi nhà cung cấp ra bản mới. Hai bundle cùng ghi chuỗi này có thể đã đo bằng hai model khác nhau. |
| ✅ | comparability | `golden_set` | 'golden_v1' |
| ❌ | comparability | `evaluated_with_generator` | ứng viên 'deepseek-v4-flash' ≠ champion 'deepseek-chat@2026-09' |
| ❌ | comparability | `judge_identity` | ứng viên 'deepseek-v4-flash\|judge-answer-relevancy@v1,judge-faithfulness@v2\|reasoning=false' ≠ champion '<không có judge>' |
| ✅ | validity | `chưa chấm được · answer_relevancy` | 0.0% phán quyết không đọc được (trần 5%) |
| ✅ | validity | `chưa chấm được · faithfulness` | 0.0% phán quyết không đọc được (trần 5%) |
| ✅ | validity | `chưa chấm được · misattribution` | 0.0% phán quyết không đọc được (trần 5%) |
| ✅ | validity | `chưa chấm được · uncited_grounding` | 0.0% phán quyết không đọc được (trần 5%) |
| ✅ | validity | `judge đã hiệu chỉnh` | κ vs người = 0.737 (tối thiểu 0.6) |
| ✅ | absolute | `ndcg@10` | 0.7079 ≥ 0.6 |
| ✅ | absolute | `recall@5` | 0.7847 ≥ 0.7 |
| ✅ | absolute | `faithfulness` | 0.9877 ≥ 0.92 |
| ❌ | absolute | `citation_accuracy` | 0.8308 < 0.85 |
| ✅ | absolute | `refusal_accuracy` | 0.9091 ≥ 0.85 |
| ❌ | absolute | `p95_end_to_end_ms` | 4706.5000 > 3500.0 |
| ⏭ | regression | `citation_accuracy` | champion không mang metric này |
| ⏭ | regression | `faithfulness` | champion không mang metric này |
| ✅ | regression | `ndcg@10` | 0.7079 vs champion 0.7079 (+0.0000, cho phép tụt 0.01) |
| ✅ | regression | `recall@5` | 0.7847 vs champion 0.7847 (+0.0000, cho phép tụt 0.01) |
| ⏭ | regression | `refusal_accuracy` | champion không mang metric này |

</details>
