"""`W3-08` — API điều khiển ingestion + worker arq, chạy thật trên Redis + Qdrant.

DoD ba vế, mỗi vế một nhóm:

* `POST /ingest` trả `job_id` **< 200 ms** → `TestKhongLamGiNangOEndpoint`
* `GET /ingest/{id}` có tiến độ → `TestTienDo`
* retry khi worker chết → `TestRetry`

## Vế thứ ba được kiểm tới đâu, và tới đâu thì không

Cơ chế phục hồi của arq: job **ở lại hàng đợi** trong lúc chạy, worker giữ một
khoá `in-progress:{job_id}` có hạn `job_timeout + 10s`, và `job_try` được `INCR`
ở **mỗi** lần nhặt job (`arq/worker.py:450-465` và `:482`). Worker chết ⇒ khoá
hết hạn ⇒ worker khác nhặt lại với `job_try = 2`.

Ở đây kiểm **nửa của mình**: `job_try > 1` phải đi vào `JobStatus.attempt`, phải
được log, và job chạy lại phải cho kết quả đúng. Nửa của arq (khoá hết hạn sau
khi tiến trình bị giết) **không** được kiểm bằng cách giết tiến trình thật — nó
được đọc trong mã nguồn arq và ghi lại ở trên. Nói rõ ranh giới đó thay vì để
người đọc tưởng cả chuỗi đã được chạy thử → xem `TD-29`.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("arq", reason="cần extra `serving`: uv sync --extra serving")
pytest.importorskip("fastapi", reason="cần extra `serving`: uv sync --extra serving")

import yaml
from arq import Worker, create_pool
from arq.connections import RedisSettings
from fastapi.testclient import TestClient

from pipeline.corpus.manifest import CorpusEntry, write_manifest
from pipeline.ingest.app import create_app
from pipeline.ingest.schemas import IngestRequest, JobState, resolve_config
from pipeline.ingest.store import JobStore
from pipeline.ingest.tasks import MAX_TRIES, ingest_job, is_transient
from rag_core.schemas import DocType, Language

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

_LINE = "Ngân sách nhà nước năm 2024 tăng chi đầu tư công cho hạ tầng giao thông. "


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, Path]]:
    """Corpus 3 tài liệu + một config index tên ngẫu nhiên, trong thư mục tạm.

    `INGEST_CONFIG_DIR` trỏ vào đây thay vì nới lỏng hàng rào của `resolve_config`:
    hàng rào ấy là biện pháp bảo mật, và một test mà phải tắt nó đi để chạy được
    là một test đang kiểm sai thứ.
    """
    from rag_core.settings import get_settings

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entries = []
    for index in range(3):
        doc_id = f"d-{index}"
        payload = f"{doc_id}. {_LINE * 12}".encode()
        (corpus / f"{doc_id}.txt").write_bytes(payload)
        entries.append(
            CorpusEntry(
                doc_id=doc_id,
                relative_path=f"{doc_id}.txt",
                source_url=f"https://example.org/{doc_id}",
                license="CC BY 4.0",
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
                source="test",
                lang=Language.VI,
                doc_type=DocType.DEV_REPORT,
            )
        )
    write_manifest(tmp_path / "manifest.csv", entries)

    name = f"ing-{uuid.uuid4().hex[:8]}"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "tenant_id": "test",
                "collection": name.replace("-", "_"),
                "manifest_path": str(tmp_path / "manifest.csv"),
                "corpus_dir": str(corpus),
                "embedding_model": "hashing:64",
                "use_cache": False,
                "state_dir": str(tmp_path / "state"),
                "chunking": {
                    "strategy": "fixed",
                    "chunk_size": 200,
                    "chunk_overlap": 0,
                    "min_chunk_size": 50,
                    "max_chunk_size": 400,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("INGEST_CONFIG_DIR", str(config_dir))
    # Hàng đợi riêng cho MỖI test. Dùng chung thì worker của test này nhặt job
    # của test kia, job hỏng vì không thấy config, và cả hai test đỏ vì một lý do
    # không liên quan đến thứ chúng đang kiểm. Đã xảy ra thật ở lượt chạy đầu.
    monkeypatch.setenv("INGEST_QUEUE", f"arq:test:{name}")
    get_settings.cache_clear()
    try:
        yield name, tmp_path
    finally:
        get_settings.cache_clear()
        from rag_core.embedding import HashingEmbeddingProvider
        from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

        collection = name.replace("-", "_")
        client = QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=64), collection=collection, url=QDRANT_URL
        ).client
        if client.collection_exists(collection):
            client.delete_collection(collection)


@pytest.fixture
def client(workspace: tuple[str, Path]) -> Iterator[TestClient]:
    """Phụ thuộc `workspace` **có chủ ý**, dù không dùng giá trị của nó.

    `create_app()` đọc `INGEST_QUEUE`/`INGEST_CONFIG_DIR` ngay lúc vào lifespan.
    Không khai phụ thuộc thì pytest dựng `client` trước `workspace` (theo thứ tự
    tham số), app nối vào hàng đợi **mặc định**, và mọi job nằm im ở `queued` —
    một kiểu hỏng không hề giống nguyên nhân của nó.

    ⭐⭐ `W6-06` / `AU-10`: `client=` phải khai tường minh. Mặc định của
    `TestClient` là host `"testclient"` — **không** phải loopback — nên từ khi
    `guard` tồn tại, mọi test ở đây trả 403 cho tới khi nó nói ra mình gọi từ
    đâu. Đó là hành vi đúng, và việc 14 bài đỏ cùng lúc là bằng chứng rằng bản
    vá thật sự chặn chứ không chỉ trông giống chặn.
    """
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as running:
        yield running


async def _drain(
    queue: str,
    *,
    max_tries: int = MAX_TRIES,
    expect_failure: bool = False,
    qdrant_url: str = QDRANT_URL,
) -> int:
    """Chạy một worker cho tới khi hàng đợi rỗng, rồi tắt. Trả số job đã xử lý.

    `expect_failure` vì `run_check` **ném** `FailedJobs` khi có job hỏng — hợp lý
    cho một health check, nhưng ở đây job hỏng chính là thứ đang kiểm.
    """
    from arq.worker import FailedJobs

    pool = await create_pool(RedisSettings.from_dsn(REDIS_URL), default_queue_name=queue)
    worker = Worker(
        functions=[ingest_job],
        redis_pool=pool,
        queue_name=queue,
        max_tries=max_tries,
        job_timeout=120,
        burst=True,
        poll_delay=0.05,
        ctx={"qdrant_url": qdrant_url, "qdrant_api_key": None},
    )
    try:
        return await worker.run_check(max_burst_jobs=10)
    except FailedJobs:
        if not expect_failure:
            raise
        return 0
    finally:
        await worker.close()


class TestKhongLamGiNangOEndpoint:
    def test_tra_job_id_duoi_200ms(self, client: TestClient, workspace: tuple[str, Path]) -> None:
        name, _ = workspace
        started = time.perf_counter()
        response = client.post("/ingest", json={"config": name})
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert response.status_code == 202
        body = response.json()
        assert len(body["job_id"]) == 32
        assert body["state"] == "queued"
        assert elapsed_ms < 200, f"{elapsed_ms:.0f} ms — endpoint đang làm việc nặng"

    def test_khong_nap_torch_khi_import_app(self) -> None:
        """API chỉ đẩy job; kéo torch vào là mất vài giây khởi động và ~2 GB RAM.

        Kiểm bằng tiến trình con vì test suite này đã nạp torch từ lâu.
        """
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent("""
            import sys
            import pipeline.ingest.app  # noqa: F401
            heavy = [m for m in ("torch", "sentence_transformers", "docling") if m in sys.modules]
            print(",".join(heavy))
        """)
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "", f"import kéo theo: {out.stdout.strip()}"


class TestHangRaoConfig:
    @pytest.mark.parametrize(
        "bad",
        ["../../.env", "..\\..\\secrets", "/etc/passwd", "bgem3/../../x", "Bgem3"],
    )
    def test_tu_choi_ten_khong_phai_ten(self, client: TestClient, bad: str) -> None:
        assert client.post("/ingest", json={"config": bad}).status_code == 422

    def test_tu_choi_config_khong_ton_tai(
        self, client: TestClient, workspace: tuple[str, Path]
    ) -> None:
        response = client.post("/ingest", json={"config": "khong-co-that"})
        assert response.status_code == 400
        assert "khong-co-that" in response.json()["detail"]

    def test_resolve_config_chan_thoat_thu_muc_ke_ca_khi_qua_mat_validator(
        self, tmp_path: Path
    ) -> None:
        """Hàng rào thứ hai kiểm **kết quả**, nên nó không phụ thuộc danh sách ký tự xấu."""
        (tmp_path / "inside").mkdir()
        with pytest.raises(ValueError, match="trỏ ra ngoài"):
            resolve_config("../outside", root=tmp_path / "inside")

    def test_job_khong_ton_tai_tra_404(self, client: TestClient) -> None:
        assert client.get(f"/ingest/{uuid.uuid4().hex}").status_code == 404


class TestTienDo:
    async def test_build_index_bao_tien_do_tung_tai_lieu(self, workspace: tuple[str, Path]) -> None:
        """Nguồn của thanh tiến độ, kiểm tách khỏi Redis để không phụ thuộc thời điểm."""
        from pipeline.indexing.build_index import build_index
        from pipeline.indexing.config import load_index_config

        name, _ = workspace
        config = load_index_config(resolve_config(name))
        seen: list[tuple[int, int]] = []
        build_index(config, qdrant_url=QDRANT_URL, on_progress=lambda d, t: seen.append((d, t)))

        assert seen[0] == (0, 3), "phải báo tổng số TRƯỚC khi làm gì"
        assert seen[-1] == (3, 3)
        assert [d for d, _ in seen] == [0, 1, 2, 3], "tiến độ phải đơn điệu tăng"

    async def test_vong_doi_day_du_qua_api(
        self, client: TestClient, workspace: tuple[str, Path]
    ) -> None:
        name, _ = workspace
        job_id = client.post("/ingest", json={"config": name}).json()["job_id"]

        pending = client.get(f"/ingest/{job_id}").json()
        assert pending["state"] == "queued"
        assert pending["progress_ratio"] == 0.0

        assert await _drain(f"arq:test:{name}") == 1

        done = client.get(f"/ingest/{job_id}").json()
        assert done["state"] == "done", done.get("error")
        assert done["attempt"] == 1
        assert done["documents_done"] == done["documents_total"] == 3
        assert done["progress_ratio"] == 1.0
        assert done["chunks_embedded"] > 0
        assert done["finished_at"]

    async def test_chi_index_lai_mot_tai_lieu_khong_go_hai_cai_kia(
        self, client: TestClient, workspace: tuple[str, Path]
    ) -> None:
        """`doc_ids` là **phạm vi lượt chạy**, không phải bộ lọc corpus.

        Nếu nó đi vào bộ lọc thì `build_index` coi hai tài liệu kia là "đã rời
        manifest" và **xoá** chúng khỏi collection — một `POST` vô hại làm mất
        2/3 index.
        """
        name, _ = workspace
        queue = f"arq:test:{name}"
        first = client.post("/ingest", json={"config": name}).json()["job_id"]
        await _drain(queue)
        total = client.get(f"/ingest/{first}").json()["chunks_embedded"]
        assert total > 0

        second = client.post("/ingest", json={"config": name, "doc_ids": ["d-1"]}).json()["job_id"]
        await _drain(queue)
        scoped = client.get(f"/ingest/{second}").json()
        assert scoped["state"] == "done", scoped.get("error")
        assert scoped["documents_total"] == 1

        from rag_core.embedding import HashingEmbeddingProvider
        from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

        store = QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=64),
            collection=name.replace("-", "_"),
            url=QDRANT_URL,
        )
        assert {c.doc_id for c in store.fetch_doc_chunks(["d-0", "d-1", "d-2"])} == {
            "d-0",
            "d-1",
            "d-2",
        }, "index lại một tài liệu đã xoá mất hai tài liệu kia"


class TestRetry:
    async def test_loi_tat_dinh_KHONG_duoc_thu_lai(
        self, client: TestClient, workspace: tuple[str, Path]
    ) -> None:
        """Xoá manifest ⇒ hỏng hẳn ở lần đầu, không thử lại. Và đó là hành vi đúng.

        Tôi viết test này để chứng minh `max_tries` hoạt động, rồi nó đỏ và buộc
        đọc `arq/worker.py:613-633`: arq chỉ thử lại `Retry`/`RetryJob`/
        `CancelledError`. Thử lại một job thiếu manifest chỉ cho ba lần hỏng y hệt
        nhau — nên test đổi chiều thành ghim đúng hành vi ấy.
        """
        name, root = workspace
        job_id = client.post("/ingest", json={"config": name}).json()["job_id"]
        (root / "manifest.csv").unlink()

        await _drain(f"arq:test:{name}", expect_failure=True)

        failed = client.get(f"/ingest/{job_id}").json()
        assert failed["state"] == "failed"
        assert failed["attempt"] == 1, "lỗi tất định không được thử lại"
        assert "CorpusIntegrityError" in failed["error"]

    async def test_loi_ha_tang_thi_CO_thu_lai(
        self, client: TestClient, workspace: tuple[str, Path]
    ) -> None:
        """Qdrant không kết nối được ⇒ `tasks.is_transient` ⇒ `Retry` ⇒ thử lại thật."""
        name, _ = workspace
        queue = f"arq:test:{name}"
        job_id = client.post("/ingest", json={"config": name}).json()["job_id"]

        # Cổng đóng: lỗi kết nối thật, không phải exception dựng sẵn.
        await _drain(queue, qdrant_url="http://127.0.0.1:59999", expect_failure=True)

        failed = client.get(f"/ingest/{job_id}").json()
        assert failed["state"] == "failed"
        assert failed["attempt"] == MAX_TRIES, (
            f"lỗi hạ tầng phải được thử đủ {MAX_TRIES} lần, mới {failed['attempt']}"
        )

    def test_phan_loai_loi(self) -> None:
        assert is_transient(ConnectionError("qdrant chết"))
        assert is_transient(RuntimeError("bọc ngoài")) is False
        wrapped = RuntimeError("bọc ngoài")
        wrapped.__cause__ = TimeoutError("hết giờ")
        assert is_transient(wrapped), "phải nhìn xuyên qua chuỗi __cause__"
        assert not is_transient(ValueError("config sai"))

    async def test_attempt_lon_hon_1_di_vao_trang_thai(
        self, client: TestClient, workspace: tuple[str, Path]
    ) -> None:
        """Nửa của **mình** trong kịch bản "worker chết": `ctx["job_try"]` phải hiện ra.

        Dựng lại bằng cách đặt trước khoá đếm lần thử mà arq `INCR` ở mỗi lần nhặt
        job (`arq/worker.py:482`) — đúng trạng thái Redis còn lại sau khi một
        worker chết giữa chừng. Nửa của arq (khoá `in-progress` hết hạn) không
        được kiểm ở đây; xem docstring đầu file và `TD-29`.
        """
        name, _ = workspace
        job_id = client.post("/ingest", json={"config": name}).json()["job_id"]

        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        try:
            await pool.setex(f"arq:retry:{job_id}", 3600, "1")
        finally:
            await pool.aclose()

        await _drain(f"arq:test:{name}")
        done = client.get(f"/ingest/{job_id}").json()
        assert done["state"] == "done", done.get("error")
        assert done["attempt"] == 2, "lần nhặt sau một worker chết phải là lần thứ 2"


class TestTrangThaiSongONgoaiTienTrinh:
    async def test_api_khoi_dong_lai_van_doc_duoc_job(
        self, client: TestClient, workspace: tuple[str, Path]
    ) -> None:
        """Trạng thái ở Redis chứ không trong RAM — một dict sẽ qua test này nếu
        test chỉ dùng một client, nên phải dựng hẳn một app thứ hai."""
        name, _ = workspace
        job_id = client.post("/ingest", json={"config": name}).json()["job_id"]
        # `client=` như fixture — xem lý do ở đó (`AU-10`).
        with TestClient(create_app(), client=("127.0.0.1", 50001)) as reborn:
            assert reborn.get(f"/ingest/{job_id}").json()["job_id"] == job_id


class TestStoreKhongNuotDuLieu:
    async def test_patch_vao_job_khong_ton_tai_tra_none(self) -> None:
        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        try:
            store = JobStore(pool)
            assert await store.patch(uuid.uuid4().hex, state=JobState.DONE) is None
        finally:
            await pool.aclose()

    def test_request_tu_choi_field_la(self) -> None:
        with pytest.raises(ValueError):
            IngestRequest.model_validate({"config": "x", "khong-co-field-nay": 1})
