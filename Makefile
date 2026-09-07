.DEFAULT_GOAL := help
SHELL := /bin/bash

# `uv run` tự đồng bộ môi trường nên không cần activate venv thủ công.
PY := uv run

.PHONY: help
help:  ## Liệt kê các target
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- môi trường
.PHONY: install
install:  ## Cài toàn bộ dependency (kể cả extras nặng)
	uv sync --all-extras

.PHONY: install-dev
install-dev:  ## Cài dependency tối thiểu để lint + unit test
	uv sync --extra dev

# ---------------------------------------------------------------- chất lượng
.PHONY: lint
lint:  ## ruff check + ruff format --check + mypy
	$(PY) ruff check .
	$(PY) ruff format --check .
	$(PY) mypy

.PHONY: fmt
fmt:  ## Tự sửa format + import order
	$(PY) ruff check --fix .
	$(PY) ruff format .

.PHONY: test
test:  ## Unit test (không cần Docker)
	$(PY) pytest -m "not integration and not gpu and not e2e"

.PHONY: test-integration
test-integration:  ## Integration test (cần `make up` trước)
	$(PY) pytest -m integration

.PHONY: test-gpu
test-gpu:  ## Test cần GPU + trọng số model thật (BGE-M3 ~2,2GB)
	$(PY) pytest -m gpu

.PHONY: test-all
test-all:  ## Toàn bộ test trừ e2e (e2e cần image: `make up-api && make smoke`)
	$(PY) pytest -m "not e2e"

.PHONY: cov
cov:  ## Unit test + báo cáo coverage của rag_core
	$(PY) pytest -m "not integration and not gpu and not e2e" \
		--cov --cov-report=term-missing --cov-report=html:.coverage_html

# ---------------------------------------------------------------- hạ tầng
COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env
# `W5-06`. Project riêng, KHÔNG `--env-file .env`: stack quan sát tự khai
# đủ biến của nó, và một `.env` thiếu một dòng không được phép làm nó lệch
# cấu hình so với lần dựng trước.
LANGFUSE := docker compose -f infra/docker-compose.langfuse.yml
METRICS := docker compose -f infra/docker-compose.metrics.yml

# Clone sạch chưa có `.env`; compose sẽ chết vì `--env-file`. Tự tạo từ mẫu để
# `make up` chạy được ngay, secret vẫn phải điền tay sau.
.env:
	cp .env.example .env
	@echo ">> Đã tạo .env từ .env.example. Điền API key trước khi chạy pipeline."

.PHONY: up
up: .env  ## Bật Qdrant + Postgres + Redis, đợi tới khi healthy
	$(COMPOSE) up -d --wait

.PHONY: up-api
up-api: up  ## `W4-13`: bật thêm API (build image nếu cần), đợi /ready xanh
	$(COMPOSE) --profile api up -d --build --wait

.PHONY: smoke
smoke:  ## `W4-13`: e2e qua compose thật — cần `make up-api` trước
	$(PY) pytest tests/e2e -m e2e

.PHONY: docker-usage
docker-usage:  ## Docker đang chiếm bao nhiêu, và bao nhiêu là rác
	@docker system df
	@echo
	@echo "⭐ Volume: kích thước BIỂU KIẾN (docker system df đếm cái này) vs BLOCK THẬT."
	@echo "   Qdrant cấp phát trước từng trang 32 MB dạng sparse ⇒ hai số lệch ~10×."
	@docker run --rm -v rag-platform_qdrant_data:/v alpine:3.20 sh -c \
	  'echo "   qdrant_data: $$(du -smb /v | cut -f1) MB biểu kiến / $$(du -sm /v | cut -f1) MB thật"' || true
	@echo
	@echo "Đã dùng THẬT bên trong đĩa ảo (so con số này với kích thước vhdx):"
	@wsl -d docker-desktop -e sh -c \
	  'nsenter -t 1 -m -- df -h /mnt/docker-desktop-disk 2>/dev/null | tail -1' || true
	@echo
	@echo "Kích thước vhdx trên ổ D:"
	@powershell.exe -NoProfile -Command \
	  "'{0:N2} GiB' -f ((Get-Item 'D:\wsl\DockerDesktopWSL\disk\docker_data.vhdx').Length/1GB)" || true

.PHONY: docker-clean
docker-clean:  ## Dọn build cache + image mồ côi. KHÔNG đụng volume (index Qdrant nằm đó)
# ⭐⭐ Build cache là thứ phình to nhất và im lặng nhất trong dự án này.
#
# `serving/Dockerfile` đúc một image **7 GB** (torch CUDA — `TD-57`), và mỗi lần
# `make up-api` build lại là thêm một lớp cache. Đo ngày 07/09/2026: **18 lần
# build → 21,5 GB cache**, nhiều hơn cả tổng số image. Không có gì báo, không có
# gì hiện ra ở `docker images`.
#
# ⚠️⚠️ CỐ Ý KHÔNG có `docker volume prune` và KHÔNG có `system prune -a`:
#   * `rag-platform_qdrant_data` là **index thật** — 7 collection còn sống,
#     20.424 chunk BGE-M3. Dựng lại nó tốn hàng giờ GPU.
#     ⭐ Nó chỉ chiếm **1,2 GB** block thật, KHÔNG phải 13 GB như `docker system
#     df` báo — xem `docker-usage` bên trên. Giá trị của nó là giờ GPU, không
#     phải dung lượng; đừng để con số phồng làm mờ lý do thật.
#   * `rag-platform_postgres_data` giữ lịch sử hội thoại + feedback của `W5-08`.
#   * `rag-serving:local` (7 GB) mất đi là 6 phút build lại cho mỗi lần `make smoke`.
#
# ⚠️ Và một cái bẫy: `docker container prune` làm mọi volume thành "mồ côi" dưới
# mắt `docker system df`, tức một lệnh `volume prune` gõ ngay sau đó sẽ xoá sạch
# index. Nên target này KHÔNG xoá container đã dừng.
	docker builder prune -af
	docker image prune -f
	@echo
	@docker system df
	@echo
	@echo "⚠️  Chỗ vừa giải phóng nằm BÊN TRONG đĩa ảo — ổ D chưa nhận lại được."
	@echo "    vhdx của WSL chỉ phình ra, không tự co (chế độ sparse bị Microsoft"
	@echo "    tắt vì rủi ro hỏng dữ liệu). Muốn trả về ổ D thì xem DOCKER-DISK.md"

.PHONY: up-langfuse
up-langfuse:  ## `W5-06`: bật Langfuse tự dựng (web+worker+clickhouse+minio+redis+pg)
	$(LANGFUSE) up -d --wait

.PHONY: down-langfuse
down-langfuse:  ## Tắt Langfuse (giữ nguyên volume dữ liệu)
	$(LANGFUSE) down

.PHONY: logs-langfuse
logs-langfuse:  ## Xem log Langfuse
	$(LANGFUSE) logs -f --tail=100

.PHONY: metrics-token
metrics-token:  ## `W5-07`: cấp khoá cho Prometheus scrape /metrics (chạy MỘT lần)
	$(PY) python -m serving.core.auth mint --tenant metrics --rpm 600 --token-file secrets/metrics-token

.PHONY: up-metrics
up-metrics:  ## `W5-07`: bật Prometheus + Grafana (bảng "RAG Health" ở :3001)
	@mkdir -p secrets
	$(METRICS) up -d --wait

.PHONY: down-metrics
down-metrics:  ## Tắt Prometheus + Grafana (giữ nguyên volume dữ liệu)
	$(METRICS) down

.PHONY: feedback-export
feedback-export:  ## `W5-08`: xuất câu 👎 thành file ứng viên golden set (TENANT=public)
	$(PY) python -m serving.core.feedback export --tenant $(or $(TENANT),public) --out $(or $(OUT),data/golden/candidates.jsonl)

.PHONY: down
down:  ## Tắt hạ tầng (giữ nguyên volume dữ liệu)
	$(COMPOSE) --profile api down

.PHONY: down-clean
down-clean:  ## Tắt hạ tầng và XÓA volume dữ liệu
	$(COMPOSE) --profile api down -v

.PHONY: ps
ps:  ## Trạng thái các service
	$(COMPOSE) --profile api ps

.PHONY: logs
logs:  ## Xem log hạ tầng
	$(COMPOSE) logs -f --tail=100

# ---------------------------------------------------------------- dữ liệu (DVC)
# Remote khai ở `.dvc/config.local` (không commit). Chưa cấu hình thì các target
# này sẽ báo lỗi của DVC — xem `plans/reports/tasks/w1-09-dvc.md`.
.PHONY: data-pull
data-pull:  ## Lấy corpus từ DVC remote về `data/corpus/`
	$(PY) dvc pull

.PHONY: data-push
data-push:  ## Đẩy corpus hiện tại lên DVC remote
	$(PY) dvc push

.PHONY: data-status
data-status:  ## So working tree với cache DVC và với remote
	$(PY) dvc status
	$(PY) dvc status -c

.PHONY: data-verify
data-verify:  ## Đối chiếu corpus DVC theo dõi với manifest (bắt lệch giữa 2 cơ chế)
	$(PY) python -m pipeline.corpus.dvc_state

.PHONY: data-track
data-track:  ## Ghi lại hash sau khi corpus đổi, rồi kiểm chéo với manifest
	$(PY) dvc add data/corpus
	$(PY) python -m pipeline.corpus.dvc_state

# ---------------------------------------------------------------- pipeline
BUNDLE ?= baseline
BASE ?= baseline
CAND ?= chunk550
INDEX_CONFIG = configs/indexing/$(BUNDLE).yaml
# Nhánh truy hồi (W2-03) — thuộc lần **đo**, không thuộc index, nên không nằm
# trong file config. `RUN` tách khỏi `BUNDLE` để hai nhánh của cùng một index ghi
# ra hai bộ báo cáo thay vì ghi đè lên nhau.
MODE ?= dense
RUN ?= $(BUNDLE)
# Tham số RRF (W2-04) — rỗng thì dùng mặc định (k=60, candidate_k=50). Ví dụ:
#   make eval-retrieval BUNDLE=bgem3 MODE=hybrid RUN=bgem3-rrf-c100 RRF_ARGS="--candidate-k 100"
RRF_ARGS ?=
# Tham số rerank (W2-05) — cùng chỗ với RRF_ARGS vì nhánh `reranked` nhận cả hai:
# nó bọc một nhánh nền, nên tham số của nền vẫn có nghĩa. Ví dụ:
#   make eval-rerank BUNDLE=bgem3 RUN=bgem3-rr-c100 RERANK_ARGS="--rerank-candidates 100"
RERANK_ARGS ?=

.PHONY: corpus
corpus:  ## Tải corpus theo config (idempotent)
	$(PY) python scripts/fetch_corpus.py --config configs/corpus/worldbank_vietnam.yaml

.PHONY: index
index:  ## Build index Qdrant (BUNDLE=baseline). Cần `make up` trước
	$(PY) python -m pipeline.indexing.build_index --config $(INDEX_CONFIG) \
		--report plans/reports/probes/index-$(BUNDLE).json

.PHONY: index-dry
index-dry:  ## Chunk thử vài tài liệu và in thống kê, không chạm Qdrant
	$(PY) python -m pipeline.indexing.build_index --config $(INDEX_CONFIG) --dry-run

.PHONY: gate
gate:  ## `W5-05`: gate phát hành cho một bundle (BUNDLE=0.2.1). Exit 1=FAIL, 2=INCOMPARABLE
	$(PY) python -m pipeline.eval.gate --bundle $(BUNDLE) \
		--html plans/reports/runs/gate-$(BUNDLE).html \
		--json plans/reports/runs/gate-$(BUNDLE).json

.PHONY: nightly
nightly:  ## `W5-10`: gate ứng viên → đề nghị phát hành. Exit 1=FAIL, 2=INCOMPARABLE, 3=promote hỏng
	$(PY) python -m pipeline.eval.nightly --out-dir plans/reports/runs $(NIGHTLY_ARGS)

.PHONY: nightly-dry
nightly-dry:  ## Như `nightly` nhưng KHÔNG đúc bundle và KHÔNG dời con trỏ
	$(PY) python -m pipeline.eval.nightly --out-dir plans/reports/runs --no-promote

BUNDLE_ROLLBACK ?=
.PHONY: bundle-rollback
bundle-rollback:  ## `W5-10`: dời con trỏ CURRENT về một bản cũ (BUNDLE_ROLLBACK=0.2.0)
	$(PY) python -c "from pathlib import Path; from pipeline.bundle.promote import rollback; 		print(rollback(Path('bundles'), '$(BUNDLE_ROLLBACK)'))"

.PHONY: smoke-eval
smoke-eval:  ## `W5-09`: smoke eval truy hồi trên index đóng băng (không model, $$0)
	$(PY) python -m pipeline.eval.smoke

.PHONY: smoke-fixture
smoke-fixture:  ## `W5-09`: dựng lại fixture đóng băng — CẦN index thật + GPU
	$(PY) python -m pipeline.eval.smoke_fixture

.PHONY: eval-compare
eval-compare:  ## So hai lần chạy eval có kiểm định (BASE=baseline CAND=chunk550)
	$(PY) python -m pipeline.eval.compare $(BASE) $(CAND) \
		--out plans/reports/compare/cmp-$(BASE)-vs-$(CAND).md

# Chiều chia nhóm (W2-08-prep). `BY` quét mọi nhóm và TỰ hiệu chỉnh đa so sánh;
# `CAT`/`LANG` so một nhóm đã nêu trước và KHÔNG hiệu chỉnh. Hai loại suy luận
# khác nhau, nên hai target khác nhau — xem docstring `compare_by_group`.
BY ?= category
CAT ?=
# ⚠️ `QLANG`, KHÔNG phải `LANG`: `LANG` là biến môi trường chuẩn (`en_US.UTF-8`),
# make thừa kế nó, nên `LANG ?=` không bao giờ có tác dụng và `--lang` luôn nhận
# một chuỗi locale. Target này hỏng từ `W2-08-prep` tới khi `W2-09` gọi nó thật.
QLANG ?=

.PHONY: eval-compare-by
eval-compare-by:  ## So theo TỪNG nhóm, có hiệu chỉnh Bonferroni (BY=category|lang)
	$(PY) python -m pipeline.eval.compare $(BASE) $(CAND) --by $(BY) \
		--out plans/reports/compare/by$(BY)-$(BASE)-vs-$(CAND).md

# Bảng ablation N ô (W2-08). `RANK` phải nêu tường minh: thứ hạng đổi theo metric,
# và một bảng không nói nó xếp theo cái gì là một bảng mời đọc sai người thắng.
EXP_PREFIX ?= e1-
RANK ?= ndcg@10
RANK_SLUG ?= ndcg
ABL_BASE ?= e1-baseline-dense

.PHONY: ablation
ablation:  ## Bảng ablation 14 ô + tập tương đương, có p/CI từng dòng (W2-08)
	$(PY) python -m pipeline.eval.ablation --prefix $(EXP_PREFIX) \
		--baseline $(ABL_BASE) --rank-by $(RANK) \
		--out plans/reports/compare/ablation-exp-001-$(RANK_SLUG).md

# Nhóm nào cải thiện nhiều nhất (W2-09). Khác `eval-compare-by`: ở đó mỗi nhóm so
# với CHÍNH NÓ ở lần chạy kia (Δ ≠ 0?); ở đây các nhóm so VỚI NHAU (Δ_A > Δ_B?).
CONTRAST_BASE ?= e1-baseline-dense
CONTRAST_CAND ?= e1-rr-bgem3-reranked-onhybrid-rc100

.PHONY: contrast
contrast:  ## Xếp hạng nhóm theo mức cải thiện + tập tương đương (W2-09)
	$(PY) python -m pipeline.eval.contrast \
		$(CONTRAST_BASE) $(CONTRAST_CAND) --by $(BY) \
		--out plans/reports/compare/contrast-$(BY)-exp-001.md

.PHONY: eval-compare-subset
eval-compare-subset:  ## So trên MỘT nhóm đã nêu trước, không hiệu chỉnh (CAT=… LANG=…)
	$(PY) python -m pipeline.eval.compare $(BASE) $(CAND) \
		$(if $(CAT),--category $(CAT)) $(if $(QLANG),--lang $(QLANG))

.PHONY: backfill-payload
backfill-payload:  ## Vá payload phẳng + payload index của collection ĐÃ build (W2-06)
	$(PY) python -m pipeline.indexing.backfill_payload --config $(INDEX_CONFIG) \
		--report plans/reports/probes/w2-06-backfill-$(BUNDLE).json

.PHONY: backfill-payload-dry
backfill-payload-dry:  ## Chỉ báo payload index còn thiếu, không ghi gì
	$(PY) python -m pipeline.indexing.backfill_payload --config $(INDEX_CONFIG) --dry-run

.PHONY: truncation
truncation:  ## Đo phần text bị model embedding cắt (TD-11). Không cần Qdrant
	$(PY) python -m pipeline.indexing.truncation_report --config $(INDEX_CONFIG) \
		--report plans/reports/probes/truncation-$(BUNDLE).json

.PHONY: goldenset-dry
goldenset-dry:  ## Xem phân bố lô chunk + prompt mẫu, KHÔNG gọi API
	$(PY) python -m pipeline.goldenset.generate --index-config $(INDEX_CONFIG) --dry-run

.PHONY: goldenset-draft
goldenset-draft:  ## Sinh nháp golden set bằng DeepSeek (TỐN TIỀN API)
	$(PY) python -m pipeline.goldenset.generate --index-config $(INDEX_CONFIG)

.PHONY: goldenset-anchor
goldenset-anchor:  ## Neo nhãn nháp vào văn bản gốc (span thay chunk_id — TD-12)
	$(PY) python -m pipeline.goldenset.anchor --index-config $(INDEX_CONFIG) \
		--report plans/reports/goldenset/goldenset-anchor.json

.PHONY: goldenset-triage
goldenset-triage:  ## Chạy retriever thật lên tập nháp → hàng đợi review (W1-11)
	$(PY) python -m pipeline.goldenset.triage --index-config $(INDEX_CONFIG)

.PHONY: goldenset-freeze
goldenset-freeze:  ## Đóng băng golden_v1 từ decisions_v1.csv (cần review xong)
	$(PY) python -m pipeline.goldenset.freeze \
		--report plans/reports/goldenset/goldenset-v1.json

.PHONY: goldenset-verify
goldenset-verify:  ## Đối chiếu golden_v1.jsonl với checksum đi kèm
	$(PY) python -m pipeline.goldenset.freeze --verify

.PHONY: eval-retrieval
eval-retrieval:  ## Eval retrieval trên index đã build (BUNDLE=baseline MODE=dense RUN=tên)
	$(PY) python -m pipeline.eval.retrieval_eval \
		--index-config $(INDEX_CONFIG) --run-name $(RUN) --retrieval-mode $(MODE) $(RRF_ARGS) $(RERANK_ARGS)

.PHONY: known-item
known-item:  ## Known-item search: tra mã tài liệu, đo dense vs sparse (W2-03)
	$(PY) python scripts/known_item_probe.py --config $(INDEX_CONFIG) \
		--report plans/reports/probes/w2-03-known-item.json

.PHONY: eval-rerank
eval-rerank:  ## Eval reranked trên cấu hình THẮNG của W2-04 (BUNDLE=bgem3 RUN=tên)
	$(PY) python -m pipeline.eval.retrieval_eval \
		--index-config $(INDEX_CONFIG) --run-name $(RUN) --retrieval-mode reranked \
		--rerank-base hybrid --rrf-k 1 --candidate-k 20 $(RERANK_ARGS)

.PHONY: loader-fixtures
loader-fixtures:  ## Sinh lại 7 fixture cho loader (W3-01, W3-02) — idempotent
	$(PY) python scripts/make_loader_fixtures.py

SCAN ?= tests/fixtures/loaders/scanned-page.pdf
.PHONY: scan-probe
scan-probe:  ## In mật độ text layer từng trang của một PDF (W3-02)
	$(PY) python -m rag_core.loaders.scan $(SCAN) --per-page

.PHONY: ingest-api
ingest-api:  ## Chạy API điều khiển ingestion (W3-08) ở cổng 8001
	$(PY) uvicorn pipeline.ingest.app:app --host 127.0.0.1 --port 8001 --reload

.PHONY: ingest-worker
ingest-worker:  ## Chạy worker arq xử lý job ingest (W3-08)
	$(PY) arq pipeline.ingest.worker.WorkerSettings

.PHONY: incr-probe
incr-probe:  ## Sửa một dòng trong corpus thật rồi đo phải embed lại bao nhiêu (W3-07)
	$(PY) python scripts/incremental_probe.py --report plans/reports/probes/w3-07-incremental.json

.PHONY: corpus-pin
corpus-pin:  ## Ghim text_sha256 + parse_fingerprint vào manifest (TD-22). Không --write = chỉ báo cáo
	$(PY) python scripts/pin_manifest.py $(PIN_ARGS)

.PHONY: parse-pin-probe
parse-pin-probe:  ## Dựng lại chế độ hỏng TD-22 trên tài liệu thật rồi cho xem nó bị chặn
	$(PY) python scripts/parse_pin_probe.py --report plans/reports/probes/td-22-parse-pin.json

PCK ?= 10
PCCFG ?= pc256
.PHONY: pc-probe
pc-probe:  ## Đo gộp trùng parent + độ nở ngữ cảnh trên index small-to-big (W3-05)
	$(PY) python scripts/parent_child_probe.py --config configs/indexing/$(PCCFG).yaml --top-k $(PCK) --report plans/reports/probes/w3-05-parent-child-$(PCCFG)-k$(PCK).json

TOKENS ?= 256
.PHONY: token-probe
token-probe:  ## So chunk_size theo ký tự vs theo token trên corpus thật (W3-06)
	$(PY) python scripts/token_sizing_probe.py --tokens $(TOKENS)

STRUCT ?= tests/fixtures/loaders/chuong-i.docx
.PHONY: structure-probe
structure-probe:  ## Đo StructureChunker trên một tài liệu thật (W3-03)
	$(PY) python scripts/structure_probe.py $(STRUCT)

.PHONY: structure-corpus
structure-corpus:  ## Đếm xem corpus hiện tại còn bao nhiêu cấu trúc để dùng (W3-03)
	$(PY) python scripts/structure_probe.py --corpus

.PHONY: loader-probe
loader-probe:  ## Cho corpus qua docling và đếm xem span golden còn sống bao nhiêu (W3-01)
	$(PY) python scripts/loader_probe.py

.PHONY: filter-probe
filter-probe:  ## Đo giá của metadata filter theo độ chọn lọc (W2-06)
	$(PY) python scripts/filter_probe.py --config $(INDEX_CONFIG) --mode $(MODE) \
		--report plans/reports/probes/w2-06-filter-probe.json

# ---------------------------------------------------------------- thí nghiệm
EXP ?= exp-001-retrieval
EXP_CONFIG = configs/eval/$(EXP).yaml

.PHONY: exp-dry
exp-dry:  ## In bảng grid + preflight, KHÔNG nạp model và không chạy ô nào (W2-07)
	$(PY) python -m pipeline.experiments.runner --config $(EXP_CONFIG) --dry-run

.PHONY: exp
exp:  ## Chạy hết grid thí nghiệm, resume được (EXP=exp-001-retrieval)
	$(PY) python -m pipeline.experiments.runner --config $(EXP_CONFIG) --keep-going

.PHONY: exp-rerun
exp-rerun:  ## Chạy lại CẢ những ô state ghi là đã xong
	$(PY) python -m pipeline.experiments.runner --config $(EXP_CONFIG) --no-resume --keep-going

.PHONY: exp-backfill
exp-backfill:  ## Dựng lại view MLflow từ báo cáo đã có, KHÔNG chạy lại eval (W2-07)
	$(PY) python -m pipeline.experiments.backfill --config $(EXP_CONFIG)

.PHONY: mlflow-ui
mlflow-ui:  ## Mở MLflow UI trên store cục bộ (http://127.0.0.1:5000)
	$(PY) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

.PHONY: rerank-probe
rerank-probe:  ## Trần vùng phủ + truncation + bão hoà sigmoid + độ trễ rerank (W2-05)
	$(PY) python scripts/rerank_probe.py --config $(INDEX_CONFIG) \
		--report plans/reports/probes/w2-05-rerank-probe.json

# W3-04 — sinh ngữ cảnh. B = số chunk mỗi lời gọi (TD-32: gộp 8 rẻ hơn 4,6×).
CTX_B ?= 8
CTX_REQUESTS = data/contexts/requests-b$(CTX_B).jsonl.gz
CTX_OUT ?= data/contexts/contexts.jsonl

.PHONY: ctx-dry
ctx-dry:  ## W3-04: in thống kê prefill + prompt mẫu, KHÔNG ghi gì, KHÔNG gọi LLM
	$(PY) python -m pipeline.indexing.contextualize prepare \
		--batch-size $(CTX_B) --dry-run

.PHONY: ctx-prepare
ctx-prepare:  ## W3-04: corpus → gói request. CTX_B= (chunk mỗi lời gọi, mặc định 8)
	$(PY) python -m pipeline.indexing.contextualize prepare \
		--batch-size $(CTX_B) --out $(CTX_REQUESTS)

.PHONY: ctx-run-glm
ctx-run-glm:  ## W3-04: sinh ngữ cảnh bằng GLM. CTX_B= LIMIT= CAP= CONC= FALLBACK=1
	$(PY) python -m pipeline.indexing.contextualize run \
		--backend glm --requests $(CTX_REQUESTS) \
		$(if $(LIMIT),--limit $(LIMIT),) --cost-cap $(or $(CAP),3.0) \
		--concurrency $(or $(CONC),16) \
		$(if $(FALLBACK),--skip-done-chunks,) \
		--out $(CTX_OUT) \
		--report plans/reports/probes/w3-04-contexts-b$(CTX_B).json

.PHONY: ctx-run-deepseek
ctx-run-deepseek:  ## W3-04: sinh ngữ cảnh bằng DeepSeek — có prefix cache thật, nhưng đắt hơn GLM
	$(PY) python -m pipeline.indexing.contextualize run \
		--backend deepseek --requests $(CTX_REQUESTS) \
		$(if $(LIMIT),--limit $(LIMIT),) --cost-cap $(or $(CAP),3.0) \
		--concurrency $(or $(CONC),16) \
		$(if $(FALLBACK),--skip-done-chunks,) \
		--out $(CTX_OUT) \
		--report plans/reports/probes/w3-04-contexts-deepseek-b$(CTX_B).json

.PHONY: ctx-fallback
ctx-fallback:  ## W3-04: lượt lùi cho chunk mà lô gộp bị chốt chặn từ chối (TD-32)
	$(MAKE) ctx-run-glm CTX_B=4 FALLBACK=1 CAP=$(or $(CAP),1.0)
	$(MAKE) ctx-run-glm CTX_B=1 FALLBACK=1 CAP=$(or $(CAP),1.0)

.PHONY: ctx-coverage
ctx-coverage:  ## W3-04: bao nhiêu chunk đã có ngữ cảnh, và thiếu ở đâu
	$(PY) python -m pipeline.indexing.contextualize coverage \
		--requests data/contexts/requests-b1.jsonl.gz --out $(CTX_OUT)


# Sao lưu artifact ngữ cảnh vào git. Nó tốn ~$5,90 tiền API thật để sinh, và
# `data/contexts/*` nằm trong .gitignore vì luật ấy viết cho gói request 285 MB
# sinh lại miễn phí — hai loại có kinh tế ngược nhau, cùng một thư mục.
# Nén xuống 1,8 MB nên vào git thẳng được, không cần DVC remote.
.PHONY: ctx-backup
ctx-backup:  ## W3-04: nén + băm artifact ngữ cảnh để commit (chạy lại sau khi sinh lại ngữ cảnh)
	$(PY) python -m pipeline.indexing.backup_contexts

.PHONY: ctx-verify
ctx-verify:  ## W3-04: bản trong git có khớp bản trên đĩa không
	$(PY) python -m pipeline.indexing.backup_contexts --check

.PHONY: db-up
db-up:  ## W4-05: alembic upgrade head (chạy bằng role owner)
	$(PY) alembic upgrade head
	$(PY) alembic current

.PHONY: db-down
db-down:  ## W4-05: alembic downgrade base. ⚠️ XOA het du lieu
	$(PY) alembic downgrade base

.PHONY: db-rev
db-rev:  ## W4-05: sinh migration moi. M="mo ta"
	$(PY) alembic revision --autogenerate -m "$(M)"

.PHONY: api-key
api-key:  ## W4-04: cấp API key mới. TENANT=acme [SCOPE=admin] [RPM=60]. In ra MỘT lần
	$(PY) python -m serving.core.auth mint --tenant $(TENANT) $(if $(SCOPE),--scope $(SCOPE),) $(if $(RPM),--rpm $(RPM),)

.PHONY: api-keys
api-keys:  ## W6-06/TD-58: liệt kê khoá đang có (key_id, tenant, rpm, scope). KHÔNG in key thô
	$(PY) python -m serving.core.auth list

.PHONY: api-key-revoke
api-key-revoke:  ## W6-06/TD-58: thu hồi một khoá. KEY_ID=public-884ac8. Cần khởi động lại API
	$(PY) python -m serving.core.auth revoke --key-id $(KEY_ID)

.PHONY: serve
serve:  ## W4-03: chạy API serving (đọc BUNDLE_ROOT / BUNDLE_VERSION từ .env)
# `python -m serving`, KHÔNG `uvicorn` trực tiếp (W4-06): trên Windows uvicorn
# dựng `ProactorEventLoop`, thứ mà driver async của psycopg không chạy được —
# nên `POST /chat` trả 503 ở mọi request. Xem docstring `serving/__main__.py`.
	$(PY) python -m serving --host 127.0.0.1 --port 8000

.PHONY: serve-check
serve-check:  ## W4-03: hỏi /health + /ready của một tiến trình đang chạy
	curl -fsS http://127.0.0.1:8000/health && echo
	curl -sS -o /dev/null -w 'ready: HTTP %{http_code}' http://127.0.0.1:8000/ready; echo
	curl -sS http://127.0.0.1:8000/admin/bundle && echo

# ------------------------------------------------------------------ W6-05
# Load test. Ba lệnh, chạy ở ba cửa sổ terminal:
#   1) make loadtest-stub          (provider giả — không tốn tiền)
#   2) make serve-stub             (server thật trỏ vào stub)
#   3) make loadtest-sweep         (quét concurrency, ghi báo cáo)
LT_LEVELS ?= 1,2,4,8,16,32
LT_DURATION ?= 60
LT_PROFILE ?= deepseek
LT_RUN ?= w605-sweep

.PHONY: loadtest-stub
loadtest-stub:  ## `W6-05`: provider OpenAI-compat giả ở :8199 (PROFILE=fast|deepseek|slow)
	$(PY) python -m loadtest.stub_llm --port 8199 --profile $(LT_PROFILE)

.PHONY: serve-stub
serve-stub:  ## `W6-05`: server thật nhưng tầng sinh trỏ vào stub — KHÔNG gọi API trả tiền
	DEEPSEEK_BASE_URL=http://127.0.0.1:8199 DEEPSEEK_API_KEY=stub-not-a-real-key \
		$(PY) python -m serving --host 127.0.0.1 --port 8000

.PHONY: loadtest-sweep
loadtest-sweep:  ## `W6-05`: quét concurrency (LT_LEVELS=… LT_DURATION=… LT_RUN=…)
	$(PY) python -m loadtest.sweep --levels $(LT_LEVELS) --duration $(LT_DURATION) \
		--label $(LT_RUN) --metrics-token "$(LT_METRICS_TOKEN)" \
		--out plans/reports/runs/$(LT_RUN).json

.PHONY: loadtest-dup
loadtest-dup:  ## `W6-05`: N request TRÙNG nhau đồng thời → đếm lời gọi provider (AU-11)
	LOADTEST_DUPLICATE=1 $(PY) python -m loadtest.sweep --levels $(LT_LEVELS) \
		--duration $(LT_DURATION) --label $(LT_RUN)-dup \
		--metrics-token "$(LT_METRICS_TOKEN)" \
		--out plans/reports/runs/$(LT_RUN)-dup.json

.PHONY: job-bundle
job-bundle:  ## W0-08: dựng gói job cho RunPod (git archive + gói request)
	bash scripts/runpod_job.sh bundle

.PHONY: job-verify
job-verify:  ## W0-08: quét bí mật trong gói job. CHẠY TRƯỚC KHI ĐẨY LÊN POD
	bash scripts/runpod_job.sh verify
