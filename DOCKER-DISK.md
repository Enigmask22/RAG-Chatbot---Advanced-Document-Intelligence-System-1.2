# Docker chiếm hết ổ D — vì sao, và cách lấy lại

> Ghi 07/09/2026 sau một lần `docker_data.vhdx` phình lên **61,6 GB** trong khi
> dữ liệu thật bên trong chỉ **14,5 GB**.

## Chẩn đoán một phút

```bash
make docker-usage
```

Ba con số cần so với nhau:

| số | ý nghĩa |
|---|---|
| `docker system df` → tổng | Docker *nghĩ* nó đang dùng bao nhiêu |
| `df` bên trong đĩa ảo | dung lượng **thật** đang chiếm |
| kích thước `docker_data.vhdx` | thứ ổ D **thật sự mất** |

Số thứ ba lớn hơn hẳn số thứ hai là bình thường, và đó chính là vấn đề.

### ⭐⭐ Nhưng đừng tin số thứ nhất khi nó nói về volume

`docker system df` báo `rag-platform_qdrant_data` là **13,04 GB**. Đo lại bằng
hai cách trên cùng một volume:

```
du -smb /v   →  12 449 MB   (kích thước biểu kiến, tính cả lỗ)
du -sm  /v   →   1 207 MB   (block thật sự cấp phát)
```

Qdrant cấp phát trước từng trang `32 MB` cho mỗi segment (`page_0.dat`,
`wal/open-*`) dưới dạng **file sparse** — file khai 32 MB nhưng chỉ chiếm vài
chục KB block. `docker system df` cộng kích thước biểu kiến, nên nó phóng đại
volume này lên hơn **10×**.

⚠️ Hệ quả rất thực tế: nhìn `13.04GB` rồi kết luận "volume là thủ phạm" là sai
đường, và cái sai ấy dẫn thẳng tới `docker volume prune`. Thủ phạm thật ở đây
luôn là **build cache** và **image**. Toàn bộ 11 volume cộng lại chỉ khoảng
**1,4 GB** block thật.

⚠️ Và đừng dùng con số phồng ấy để biện minh cho luật an toàn bên dưới. Lý do
không xoá index Qdrant là **hàng giờ GPU để dựng lại**, không phải dung lượng
nó chiếm — bản đầu của tài liệu này viết "index Qdrant 12 GB" và đó là con số
đọc từ `docker system df`, tức một lý lẽ đúng dựa trên một số liệu sai.

## ⭐⭐ Hai nguyên nhân, và chúng độc lập nhau

### 1. Build cache phình im lặng

`serving/Dockerfile` đúc một image **7 GB** (torch CUDA — `TD-57`), và mỗi lần
`make up-api` build lại là thêm một lớp cache. Đo được: **18 lần build → 21,5 GB
cache**, nhiều hơn cả tổng số image trong máy.

Nó không hiện ở `docker images`, không hiện ở tab Images của Docker Desktop, và
không có gì cảnh báo. Chỉ `docker system df` mới thấy.

```bash
make docker-clean     # dọn build cache + image mồ côi
```

### 2. ⚠️ Đĩa ảo của WSL **chỉ phình ra, không bao giờ co lại**

Đây là phần khiến việc dọn ở trên trông như vô ích: xoá 21,5 GB bên trong xong,
ổ D **không nhận lại một byte nào**.

```
fsutil sparse queryflag D:\wsl\DockerDesktopWSL\disk\docker_data.vhdx
→ This file is NOT set as sparse
```

File không ở chế độ sparse ⇒ Windows không lấy lại được block đã trống. Hệ điều
hành khách *đã* báo trống từ lâu (`fstrim` chỉ ra thêm 3,1 MiB — nghĩa là ext4
đang bật `discard` và đã báo hết); vấn đề nằm ở đầu Windows.

⚠️⚠️ **Đừng bật sparse.** WSL từ chối nó có lý do, và nó nói thẳng:

```
wsl --manage docker-desktop --set-sparse true
→ Sparse VHD support is currently disabled due to potential data corruption.
```

Có cờ `--allow-unsafe` để ép. **Không dùng** trên máy này: cùng cái đĩa ấy đang
giữ `rag-platform_qdrant_data` (20.424 chunk BGE-M3 — dựng lại tốn hàng giờ GPU)
và `rag-platform_postgres_data` (lịch sử hội thoại + feedback `W5-08`).
Đổi vài GB lấy rủi ro hỏng cả hai là một cuộc đổi tồi.

## Cách lấy lại dung lượng — nén một lần, cần quyền admin

```powershell
# 1. Dọn phần bên trong trước, nếu không thì chẳng có gì để nén
make docker-clean

# 2. Tắt Docker và WSL
docker desktop stop
wsl --shutdown

# 3. Nén — CHẠY POWERSHELL/CMD VỚI QUYỀN ADMINISTRATOR
diskpart /s C:\Windows\Temp\nen-docker.txt

# 4. Bật lại
docker desktop start
```

Nội dung `nen-docker.txt`:

```
select vdisk file="D:\wsl\DockerDesktopWSL\disk\docker_data.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

`attach vdisk readonly` là điểm đáng chú ý: đĩa được gắn ở chế độ **chỉ đọc**
trong lúc nén, nên không có đường nào để thao tác này sửa dữ liệu bên trong.

**Đo được khi chạy thật (07/09/2026):** `61,61 GiB → 17,05 GiB`, ổ D từ 144,9 GB
lên **189,4 GB** trống — lấy lại **44,5 GB**. Sau đó kiểm chứng: 10/10 volume còn
nguyên, 19/19 collection Qdrant `green`, không mất một điểm nào.

## ⚠️⚠️ Những lệnh KHÔNG được gõ trên máy này

| lệnh | mất gì |
|---|---|
| `docker volume prune` | **index Qdrant** (20.424 chunk, hàng giờ GPU) + lịch sử hội thoại + feedback |
| `docker system prune -a` | mọi thứ ở trên, cộng image `rag-serving:local` (7 GB, 6 phút build lại) |
| `docker desktop` → *Reset disk image* | toàn bộ, không hoàn tác được |
| `wsl --manage … --set-sparse --allow-unsafe` | rủi ro hỏng dữ liệu, xem trên |

⭐ Và một cái bẫy đã suýt sập: `docker container prune` **không** xoá dữ liệu,
nhưng nó làm mọi volume thành "mồ côi" dưới mắt `docker system df` —
`RECLAIMABLE` nhảy từ `0B` lên `13.31GB (100%)`. Một lệnh `volume prune` gõ ngay
sau đó, với thiện chí dọn dẹp, sẽ xoá sạch index. Dựng lại container bằng
`docker compose create` là khoá chúng lại được:

```bash
docker compose -f infra/docker-compose.yml --env-file .env create
docker compose -f infra/docker-compose.metrics.yml create
docker compose -f infra/docker-compose.langfuse.yml create
```

## Còn cắt được nữa không

| bỏ đi | lấy lại | mất gì |
|---|---|---|
| Stack Langfuse (`langfuse`, `worker`, `clickhouse`, `minio`) | ~3,5 GB image | trace `W5-06`/`W5-08`; tải lại 3,5 GB khi cần |
| Stack metrics (`grafana`, `prometheus`) | ~0,8 GB image | bảng RAG Health `W5-07`; volume chỉ 30 MB, bảng có sẵn trong `infra/grafana/` |
| `rag-serving:local` | 7 GB | `make smoke` phải build lại ~6 phút |

⚠️ Không cái nào trong ba dòng trên là "rác" — chúng là bằng chứng của `W5`. Cắt
chúng là quyết định đánh đổi, không phải dọn dẹp.
