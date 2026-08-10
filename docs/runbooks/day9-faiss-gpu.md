# Day 9: FAISS GPU build trong WSL2

Day 9 giữ baseline chính xác `IndexFlatIP`; không dùng approximate index. Gói FAISS GPU
được hỗ trợ ở Linux, vì vậy chạy build CUDA trong WSL2/Linux. Môi trường Python native
Windows của dự án vẫn là `faiss-cpu`; chỉ có thể chạy FAISS GPU native Windows khi tự build
FAISS từ source với CUDA.

## Giá trị khóa

```powershell
$fingerprint = '37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f'
$lockPath = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
```

Invariant này áp dụng cho các lệnh retrieval cleanup, build và verification: chúng phải bind
đúng `$lockPath` và `$fingerprint`. Các lệnh cài WSL/Conda và GPU preflight không đọc dữ liệu
retrieval nên không dùng path khóa. Không sửa lock, gold, hay dữ liệu canonical trong khi chạy
runbook.

## Kiểm tra WSL2 và tạo môi trường Conda

Trong Windows PowerShell, kiểm tra WSL2 và GPU đã được chuyển tiếp vào Linux:

```powershell
wsl.exe --status
wsl.exe -l -v
wsl.exe nvidia-smi
```

Nếu chưa có distro WSL2, chạy `wsl.exe --install -d Ubuntu`, khởi động lại khi Windows yêu
cầu, rồi mở Ubuntu. Trong Ubuntu/WSL2, tạo môi trường Python 3.11 và cài gói Linux chính thức:

```bash
~/.local/bin/micromamba create -y -r ~/.local/share/micromamba \
  -n financial-faiss-gpu -c pytorch -c nvidia -c conda-forge \
  python=3.11 faiss-gpu=1.10.0 pip
```

`pyproject.toml` hiện khai báo `faiss-cpu` cho môi trường Windows. Không chạy `uv sync` hoặc
`pip install -e .` có dependency resolution trong environment GPU vì chúng có thể thay
`faiss-gpu` bằng `faiss-cpu`. Cài các dependency còn lại và project không dependency như sau:

```bash
cd /mnt/d/GitHub/financial-assistant
GPU_ENV=~/.local/share/micromamba/envs/financial-faiss-gpu/bin
"$GPU_ENV/python" -m pip install 'uv==0.11.28'
"$GPU_ENV/uv" export --frozen --no-dev --no-emit-project --no-hashes \
  | grep -vE '^(faiss-cpu|numpy|torch|nvidia-|triton|cuda-)' \
  > ~/.cache/financial-assistant/gpu-requirements.txt
"$GPU_ENV/python" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.13.0+cpu'
"$GPU_ENV/python" -m pip install --no-deps -r ~/.cache/financial-assistant/gpu-requirements.txt
"$GPU_ENV/python" -m pip install --no-deps -e .
```

Giữ NumPy `1.26.4` do Conda cài cùng `faiss-gpu`: binary FAISS 1.10.0 không tương thích với
NumPy 2.x. Torch chạy CPU trong Day 9; CUDA chỉ dành cho FAISS index build.

## Preflight GPU

Chạy trong WSL2 sau khi kích hoạt environment. Lệnh phải in Python, phiên bản FAISS,
`faiss.get_num_gpus()`, và driver CUDA; dừng tại đây nếu số GPU không dương.

```bash
cd /mnt/d/GitHub/financial-assistant
GPU_ENV=~/.local/share/micromamba/envs/financial-faiss-gpu/bin
"$GPU_ENV/python" -c "import sys, faiss, numpy; print('Python:', sys.version); print('FAISS:', faiss.__version__); print('NumPy:', numpy.__version__); print('FAISS GPUs:', faiss.get_num_gpus()); assert faiss.get_num_gpus() > 0"
nvidia-smi
```

## Cleanup an toàn: dry-run rồi mới apply

Chạy từ Windows PowerShell tại worktree. Cleanup không nhận release-lock làm CLI flag; biến
khóa dưới đây là bằng chứng thao tác giữ nguyên release lock/fingerprint. Lần đầu luôn dry-run.
Lệnh `--apply` chỉ quarantine candidate được phê duyệt trong `data/`, không xóa vĩnh viễn.

```powershell
$fingerprint = '37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f'
$lockPath = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
Test-Path $lockPath
uv run --frozen --no-sync financial-report-qa retrieval cleanup-day9-data `
  --repo-root . `
  --quarantine-root data/quarantine/day9-cleanup
```

Chỉ khi dry-run không có entry `blocked` ngoài dự kiến, áp dụng cùng scope:

```powershell
$fingerprint = '37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f'
$lockPath = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
Test-Path $lockPath
uv run --frozen --no-sync financial-report-qa retrieval cleanup-day9-data `
  --repo-root . `
  --quarantine-root data/quarantine/day9-cleanup `
  --apply
```

## BGE-M3 GPU build có log

Chạy từ Windows PowerShell tại repository root. BGE-M3 đã pin revision
`5617a9f61b028005a4858fdac845db406aefb181` trong code. `python -u` phát tiến độ theo batch
ngay lập tức; `Tee-Object` ghi cả stdout/stderr vào log.

```powershell
$fingerprint = '37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f'
$lockPath = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
$log = 'artifacts/evaluations/day9/bge-m3-faiss-gpu-build.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
& wsl.exe bash -lc "cd /mnt/d/GitHub/financial-assistant && \
  ~/.local/share/micromamba/envs/financial-faiss-gpu/bin/financial-report-qa retrieval build-dense-index \
  --release-lock $lockPath \
  --corpus-dir data/indexes/dense-day9-a/$fingerprint/corpus \
  --encoder bge-m3 \
  --output-root data/indexes/dense-day9-a/$fingerprint/encoders \
  --observation-path artifacts/evaluations/day9/bge-m3-faiss-gpu-build.json \
  --local-files-only --faiss-device cuda" 2>&1 | Tee-Object -FilePath $log
```

Theo dõi log trong PowerShell khác khi build đang chạy:

```powershell
Get-Content -Path artifacts/evaluations/day9/bge-m3-faiss-gpu-build.log -Wait
```

## Xác minh artifacts

Sau build, kiểm tra release-lock, corpus/index manifest, observation CUDA, và hai artifact index.

```powershell
$fingerprint = '37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f'
$lockPath = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
$corpusManifestPath = "data/indexes/dense-day9-a/$fingerprint/corpus/manifest.json"
$indexRoot = "data/indexes/dense-day9-a/$fingerprint/encoders"
if (-not (Test-Path $corpusManifestPath)) { throw 'locked dense corpus manifest is missing' }
$releaseLock = Get-Content -Raw $lockPath | ConvertFrom-Json
$releaseLockSha256 = (Get-FileHash -Algorithm SHA256 $lockPath).Hash.ToLowerInvariant()
$corpusManifest = Get-Content -Raw $corpusManifestPath | ConvertFrom-Json
$observation = Get-Content -Raw artifacts/evaluations/day9/bge-m3-faiss-gpu-build.json | ConvertFrom-Json
if ($releaseLock.dataset_fingerprint -ne $fingerprint) { throw 'release lock fingerprint mismatch' }
if ($observation.dataset_fingerprint -ne $fingerprint -or $observation.faiss_device -ne 'cuda') {
  throw 'GPU build observation mismatch'
}
if ($corpusManifest.dataset_fingerprint -ne $fingerprint -or
    $corpusManifest.release_lock_sha256 -ne $releaseLockSha256) {
  throw 'corpus manifest lock or fingerprint mismatch'
}
$indexDirectories = Get-ChildItem $indexRoot -Directory
if ($indexDirectories.Count -lt 1) { throw 'no persisted BGE-M3 index directory' }
$indexDirectories | ForEach-Object {
  $indexPath = "$($_.FullName)/index.faiss"
  $manifestPath = "$($_.FullName)/manifest.json"
  if (-not (Test-Path $indexPath) -or -not (Test-Path $manifestPath)) {
    throw "incomplete index artifact: $($_.FullName)"
  }
  $indexManifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
  $indexSha256 = (Get-FileHash -Algorithm SHA256 $indexPath).Hash.ToLowerInvariant()
  if ($indexManifest.dataset_fingerprint -ne $fingerprint -or
      $indexManifest.release_lock_sha256 -ne $releaseLockSha256 -or
      $indexManifest.index_type -ne 'IndexFlatIP' -or
      $indexManifest.document_count -ne $corpusManifest.document_count -or
      $indexManifest.artifact_sha256.'index.faiss' -ne $indexSha256) {
    throw "index manifest validation failed: $($_.FullName)"
  }
}
Write-Output 'Day 9 CUDA observation and index artifacts verified.'
```

Kết quả observation phải có `faiss_device` là `cuda`, fingerprint đúng giá trị khóa, và index
persisted vẫn là exact CPU `IndexFlatIP` để replay/loader hiện có đọc được.
