# Serving Plane — `W4-13`.
#
# ## ⭐⭐ Vì sao image này mang torch CUDA thay vì torch CPU
#
# Phản xạ đúng cho một image serving là cắt mọi thứ không cần: torch CPU-only
# nhỏ hơn torch CUDA khoảng **4 GB**, và một image 2 GB thì dễ sống hơn một
# image 6 GB ở mọi khâu.
#
# Ở dự án này nó **sai**, vì một lý do không nằm trong Dockerfile: bundle
# `v0.2.0` khai `retriever_name = …@cuda:L512:float16:n50`, và phép kiểm danh
# tính của `TD-38`/`W4-02` so tên runtime dựng ra với tên đã eval. Một container
# CPU dựng ra `…@cpu:…:float32:…`, không khớp, nên nó **từ chối nạp bundle** —
# `/ready` 503, `/chat` 503. Muốn nó chạy thì phải bật
# `BUNDLE_ALLOW_RUNTIME_DRIFT`, tức tuyên bố "tôi biết mình đang phục vụ một hệ
# thống khác hệ thống đã đo". Với một image demo mà cả `G4` lẫn CV trỏ vào, đó
# là cái giá sai để trả cho 4 GB.
#
# Điều kiện để lựa chọn này đứng vững đã được **đo, không đoán**:
# `docker run --gpus all nvidia/cuda:12.6.0-base nvidia-smi` thấy RTX 4060 từ
# trong container. Máy đích không có GPU thì đường lùi là hai bước tường minh:
# bỏ khối `deploy.resources` của service `api` trong `docker-compose.yml`, và
# đặt `BUNDLE_ALLOW_RUNTIME_DRIFT=1` — lúc đó `GET /admin/bundle` khai
# `runtime_drift`, nên việc đang phục vụ một hệ thống khác là thứ nhìn thấy
# được chứ không phải một cửa thoát im lặng.
#
# ## Trọng số model KHÔNG nướng vào image
#
# BGE-M3 (2,2 GB) + bge-reranker-v2-m3 (2,2 GB) = 4,4 GB. Nướng vào thì image
# gần 11 GB và mọi lần sửa một dòng Python phải đẩy lại toàn bộ. Chúng đi qua
# volume `HF_HOME` (xem compose): image mỏng, cache dùng chung với đường chạy
# trên host, và `docker build` không phụ thuộc mạng của Hugging Face.
#
# ⚠️ Hệ quả phải nói to: lần chạy **đầu** trên một máy sạch tải 4,4 GB. Câu
# "≤ 5 phút" của `G4` chỉ đúng khi cache đã ấm.

# ---------------------------------------------------------------- builder ---
FROM python:3.12-slim AS builder

# ⭐⭐ `uv sync --locked`, KHÔNG phải `uv pip install -r pyproject.toml`.
#
# Bản `W4-13` viết `-r pyproject.toml` và kèm một chú thích khẳng định image
# "không thể lệch phiên bản so với `uv.lock`". Chú thích ấy **sai**, và `W5-01`
# trả tiền để biết: `-r pyproject.toml` **giải lại** phụ thuộc từ đầu mỗi lần
# build, còn `uv.lock` được COPY vào chỉ để làm khoá cache lớp. Bốn lần rebuild
# trong vòng vài giờ đủ để image trôi khỏi môi trường đã đo:
#
#   lock / môi trường dev : transformers 5.15.0 · sentence-transformers 5.7.0 · torch 2.13.0
#   image sau khi trôi    : transformers 5.16.1 · sentence-transformers 6.0.1  · torch 2.14.0
#
# Hậu quả không phải lý thuyết: cross-encoder nạp ra **dtype trộn** (embedding
# fp16, `key.weight` fp32) và **mọi** request có truy hồi trả 503 —
# `mat1 and mat2 must have the same dtype, but got Half and Float`.
#
# ⚠️ Và `GET /admin/bundle` vẫn báo `runtime_drift: null` suốt lúc đó. Phép kiểm
# danh tính `TD-38` soi tên model / thiết bị / dtype / max_length — **không** soi
# phiên bản thư viện. Nên artifact vẫn khai "đúng hệ thống đã đo" trong khi hệ
# thống đã khác. → `TD-62`.
#
# `--locked` mạnh hơn `--frozen`: nó **đỏ** khi `uv.lock` không còn khớp
# `pyproject.toml`, tức sửa dependency mà quên lock lại thì build hỏng thay vì
# lặng lẽ resolve ra một thứ khác.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Lớp phụ thuộc tách khỏi lớp mã nguồn: sửa một dòng Python không được kéo theo
# việc tải lại 2,5 GB wheel torch.
COPY pyproject.toml uv.lock README.md ./
COPY packages/rag_core/__init__.py packages/rag_core/__init__.py

# `--no-install-project`: gói ứng dụng không cài vào venv — mã nguồn tới qua
# `COPY` + `PYTHONPATH` ở tầng runtime. Nguồn index `pytorch-cu126` đã khai
# trong `[[tool.uv.index]]`, nên ở đây không cần `--extra-index-url`; và không
# cần nghĩa là không lệch được với môi trường dev.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project \
        --extra embeddings --extra qdrant --extra serving --extra llm

# ---------------------------------------------------------------- runtime ---
FROM python:3.12-slim AS runtime

# `libgomp1` là phụ thuộc runtime của torch mà image slim không có; thiếu nó thì
# lỗi hiện ra lúc `import torch`, tức sau khi build đã xong và trông như thành
# công. `curl` cho healthcheck của compose.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Chạy **không phải root**. Không phải nghi thức: container này mount cache HF
# từ host, và một tiến trình root ghi vào đó để lại file mà người dùng trên host
# không xoá được bằng quyền của mình.
RUN useradd --create-home --uid 10001 rag

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # `packages/` không được cài dạng editable trong image; thêm vào path thay
    # vì `pip install -e` để lớp mã nguồn vẫn chỉ là một `COPY`.
    PYTHONPATH=/app/packages \
    HF_HOME=/models \
    # Tắt telemetry của HF: một container serving không có lý do gì gọi ra
    # ngoài ngoài chính hai model nó cần.
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY packages/ /app/packages/
COPY serving/ /app/serving/
COPY alembic.ini /app/alembic.ini
# ⚠️ `alembic/` phải đi cùng `alembic.ini`, và thiếu nó KHÔNG làm image hỏng
# lúc build: `/health` xanh, bundle nạp xong, Qdrant đếm được — chỉ `/ready`
# trả 503 với "Path doesn't exist: /app/alembic". Đúng kiểu hỏng mà phép
# kiểm "DB sẵn sàng nghĩa là migration đã chạy" của `W4-03` sinh ra để bắt,
# và lần chạy container đầu tiên bắt được nó thật.
COPY alembic/ /app/alembic/
COPY bundles/ /app/bundles/

# `/models` thuộc về `rag` để tiến trình ghi được cache khi volume trống.
RUN mkdir -p /models && chown -R rag:rag /models /app
USER rag

EXPOSE 8000

# ⚠️ Một worker. `--workers N` nhân số bản sao của BGE-M3 + reranker trong VRAM,
# và 8 GB của RTX 4060 chỉ chứa được một bộ. Mở rộng theo chiều ngang ở đây là
# thêm **container**, không phải thêm worker — và nó cần một GPU nữa.
#
# ⭐ `WEB_CONCURRENCY` thay cho `--workers 1` (`TD-75`, `W6-05`): uvicorn đọc
# biến này khi không có cờ, và `/metrics` đọc **cùng biến ấy** để phát
# `rag_scrape_workers`. Với hai chỗ khai riêng, ai đó đổi cờ mà quên biến sẽ
# làm bản phơi bày khai một giả định không còn đúng — mà cả điểm của gauge đó
# là để nó nói thật về giả định của mình.
ENV WEB_CONCURRENCY=1
CMD ["uvicorn", "serving.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
