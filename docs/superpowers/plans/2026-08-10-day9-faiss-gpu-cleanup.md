# Day 9 FAISS GPU Build and Data Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit FAISS CUDA build path with progress logging and a fail-closed data cleanup/quarantine command, then document the WSL2 execution commands.

**Architecture:** Keep exact `IndexFlatIP` artifacts and the current CPU loader contract. `build_dense_index` optionally creates the index on GPU 0, adds normalized batches, and converts it back to CPU before atomic persistence; CLI observations record the operational device. A separate cleanup module inventories protected/candidate paths, blocks unreadable or referenced releases, and quarantines only validated paths inside `data/`.

**Tech Stack:** Python 3.11, FAISS GPU/CPU Python bindings, NumPy, Pydantic, argparse, pytest, PowerShell, WSL2/Linux, Conda.

## Global Constraints

- Preserve exact `IndexFlatIP` and inner-product semantics; do not introduce approximate indexes.
- CUDA mode must fail closed when GPU symbols are missing or `faiss.get_num_gpus() == 0`; no silent CPU fallback.
- Keep SentenceTransformer inference on CPU in this change; only FAISS index construction is accelerated.
- Never delete `data/raw`, `data/manifests`, the canonical release `data/processed/release_v2_37a61be7aebd`, the reviewed gold file, or user-modified QA files.
- Cleanup defaults to dry-run and moves approved candidates to a timestamped quarantine directory; no irreversible purge in the first implementation.
- All new behavior requires a failing test before production code, then focused and full verification.
- The Windows environment remains `faiss-cpu`; GPU execution is documented for WSL2/Linux with Conda `faiss-gpu`.

---

### Task 1: Add an explicit FAISS device seam to dense index construction

**Files:**
- Modify: `src/financial_report_qa/retrieval/dense_index.py`
- Test: `tests/unit/retrieval/test_dense_index.py`

**Interfaces:**
- Consumes: existing `DenseCorpus`, `DenseEncoder`, and normalized batch validation.
- Produces: `ProgressCallback = Callable[[int, int, float], None]` and `build_dense_index(corpus, encoder, *, faiss_device: Literal["cpu", "cuda"] = "cpu", progress: ProgressCallback | None = None) -> DenseIndex`.

- [ ] **Step 1: Write the failing GPU-path test.**

Add a test that monkeypatches the imported FAISS module with `StandardGpuResources`, `index_cpu_to_gpu`, and `index_gpu_to_cpu` seams. Build a three-row fixture with `faiss_device="cuda"`, assert the conversion functions were called once on device 0, assert every encoder batch was added, and assert the returned index is a CPU `IndexFlatIP` with `ntotal == 3`.

```python
def test_cuda_build_uses_gpu_then_returns_cpu_index(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: calls.append("resources") or object(), raising=False)
    monkeypatch.setattr(
        faiss,
        "index_cpu_to_gpu",
        lambda resources, device, index: calls.append(f"to_gpu:{device}") or index,
        raising=False,
    )
    monkeypatch.setattr(
        faiss,
        "index_gpu_to_cpu",
        lambda index: calls.append("to_cpu") or index,
        raising=False,
    )

    built = build_dense_index(_corpus(), _encoder(), faiss_device="cuda")

    assert calls == ["resources", "to_gpu:0", "to_cpu"]
    assert isinstance(built.faiss_index, faiss.IndexFlatIP)
    assert built.faiss_index.ntotal == 3
```

- [ ] **Step 2: Run the test and verify the expected failure.**

Run:

```powershell
uv run --frozen --no-sync pytest tests/unit/retrieval/test_dense_index.py::test_cuda_build_uses_gpu_then_returns_cpu_index -q
```

Expected: FAIL because `build_dense_index` does not accept `faiss_device`.

- [ ] **Step 3: Add the minimal device factory and conversion logic.**

Define a `Literal["cpu", "cuda"]` device type and a callback protocol. For `cpu`, keep `faiss.IndexFlatIP`. For `cuda`, check the three GPU symbols and `faiss.get_num_gpus() > 0`; raise `DenseArtifactError` with the missing API/count in the message otherwise. Use `faiss.StandardGpuResources()`, `faiss.index_cpu_to_gpu(resources, 0, cpu_index)`, add each validated batch, call `faiss.index_gpu_to_cpu` once, and return the CPU index. Call `progress(encoded, total, elapsed_seconds)` after every batch.

- [ ] **Step 4: Run focused tests and preserve the CPU regression.**

Run:

```powershell
uv run --frozen --no-sync pytest tests/unit/retrieval/test_dense_index.py -q
```

Expected: all existing persistence/vector tests plus the new CUDA seam test pass.

- [ ] **Step 5: Commit the index backend change.**

```powershell
git add src/financial_report_qa/retrieval/dense_index.py tests/unit/retrieval/test_dense_index.py
git commit -m "feat(retrieval): add explicit FAISS CUDA build path"
```

### Task 2: Expose device selection, progress, and build observations in the CLI

**Files:**
- Modify: `src/financial_report_qa/retrieval/cli.py`
- Modify: `tests/integration/retrieval/test_day9_dense_cli.py`
- Modify: `tests/unit/retrieval/test_dense_contracts.py`

**Interfaces:**
- Consumes: Task 1 `faiss_device` and progress callback.
- Produces: `build-dense-index --faiss-device {cpu,cuda}` and observation fields `faiss_device: Literal["cpu", "cuda"]`, `faiss_gpu_count: int`.

- [ ] **Step 1: Add failing CLI assertions.**

Extend the fixture lifecycle test to pass `--faiss-device cpu`, capture stdout, and assert it contains `dense-build: 3/3` and `dense-build: complete`. Read the observation and assert `faiss_device == "cpu"` and `faiss_gpu_count == 0` in the CPU fixture. Add a parser test that `--faiss-device cuda` is accepted.

- [ ] **Step 2: Run the focused CLI tests and verify failure.**

Run:

```powershell
uv run --frozen --no-sync pytest tests/integration/retrieval/test_day9_dense_cli.py -q
```

Expected: FAIL because the parser and observation model do not contain the new fields/logs.

- [ ] **Step 3: Implement the CLI option and progress callback.**

Add the `--faiss-device` argument with default `cpu`. Before building, compute the available GPU count (`faiss.get_num_gpus()` when present, otherwise `0`). Pass a callback that prints `dense-build: {encoded}/{total} vectors, {elapsed:.1f}s, {rate:.1f} vectors/s` with `flush=True`. Print `dense-build: complete` after atomic publication. Persist `faiss_device` and `faiss_gpu_count` in `_DenseBuildObservation`; update `_load_build_observation` validation so old observations fail clearly instead of being silently reused.

- [ ] **Step 4: Run the focused CLI and contract tests.**

```powershell
uv run --frozen --no-sync pytest tests/integration/retrieval/test_day9_dense_cli.py tests/unit/retrieval/test_dense_contracts.py -q
```

Expected: PASS, including corrupted-index exit code `2` and deterministic fixture lifecycle.

- [ ] **Step 5: Commit the CLI change.**

```powershell
git add src/financial_report_qa/retrieval/cli.py tests/integration/retrieval/test_day9_dense_cli.py tests/unit/retrieval/test_dense_contracts.py
git commit -m "feat(retrieval): expose FAISS device and build progress"
```

### Task 3: Implement fail-closed cleanup planning and quarantine

**Files:**
- Create: `src/financial_report_qa/retrieval/data_cleanup.py`
- Modify: `src/financial_report_qa/retrieval/cli.py`
- Create: `tests/unit/retrieval/test_data_cleanup.py`
- Modify: `tests/integration/retrieval/test_day9_dense_cli.py`

**Interfaces:**
- Consumes: repository root and the Day 9 protected/candidate path policy.
- Produces: `CleanupEntry` and `CleanupPlan` frozen dataclasses, `plan_day9_cleanup(repo_root: Path) -> CleanupPlan`, `quarantine_day9_cleanup(plan: CleanupPlan, quarantine_root: Path) -> list[Path]`, and CLI command `cleanup-day9-data --repo-root PATH --quarantine-root PATH [--apply]`.

- [ ] **Step 1: Write failing cleanup tests.**

Cover four behaviors with `tmp_path`: dry-run lists rebuildable interim directories and superseded releases; protected canonical/raw/QA paths are never candidates; a candidate containing a reference in a release/report file is blocked; `--apply` moves only approved candidates under a timestamped quarantine directory and leaves the source absent. Add an unreadable-candidate test that returns a blocked reason without attempting deletion.

- [ ] **Step 2: Run the cleanup tests and verify failure.**

```powershell
uv run --frozen --no-sync pytest tests/unit/retrieval/test_data_cleanup.py -q
```

Expected: FAIL because the cleanup module and CLI command do not exist.

- [ ] **Step 3: Implement the cleanup policy.**

Use immutable repository-relative paths. Define `CleanupEntry(path: Path, reason: str, status: Literal["approved", "blocked", "missing"], byte_count: int, detail: str)` and `CleanupPlan(entries: list[CleanupEntry], generated_at: str)`. Protect `data/raw`, `data/manifests`, `data/qa`, `data/processed/release_v2_37a61be7aebd`, and all paths outside `data/`. Inspect candidate manifests and scan repository text artifacts for candidate directory names; mark unreadable or referenced candidates blocked. Include `data/interim/week1_gate_attempts`, `data/interim/week1_gate_replay`, and the four old processed release directories as candidates. `quarantine_day9_cleanup` must resolve every source and destination, require both inside `data/`, create a timestamped quarantine directory, and use `shutil.move` only for approved entries.

- [ ] **Step 4: Add the CLI and verify dry-run/apply behavior.**

Add `cleanup-day9-data` to the retrieval parser. The default prints JSON lines for every plan entry and performs no mutation; `--apply` prints each move and returns `2` if any candidate is blocked. Run:

```powershell
uv run --frozen --no-sync financial-report-qa retrieval cleanup-day9-data --repo-root . --quarantine-root data/quarantine/day9-cleanup
uv run --frozen --no-sync pytest tests/unit/retrieval/test_data_cleanup.py tests/integration/retrieval/test_day9_dense_cli.py -q
```

Expected: dry-run is non-mutating; fixture apply moves only approved paths; all tests pass.

- [ ] **Step 5: Commit the cleanup command.**

```powershell
git add src/financial_report_qa/retrieval/data_cleanup.py src/financial_report_qa/retrieval/cli.py tests/unit/retrieval/test_data_cleanup.py tests/integration/retrieval/test_day9_dense_cli.py
git commit -m "feat(data): add fail-closed Day 9 cleanup quarantine"
```

### Task 4: Document WSL2/Conda GPU setup and logging commands

**Files:**
- Create: `docs/runbooks/day9-faiss-gpu.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2 CLI flags and Task 3 cleanup command.
- Produces: copy-pasteable setup, preflight, cleanup, build, and log-follow commands.

- [ ] **Step 1: Write the runbook content.**

Document WSL2 availability, the official Linux Conda installation using `faiss-gpu=1.15.0` from `pytorch`, `nvidia`, and `conda-forge`, the project dependency install without replacing GPU FAISS with `faiss-cpu`, and a preflight that prints Python, FAISS version, `faiss.get_num_gpus()`, and CUDA driver status. Include the explicit BGE-M3 command with the approved release lock, corpus fingerprint, output root, observation path, `--local-files-only`, `--faiss-device cuda`, unbuffered output, and `Tee-Object` log capture. Include `Get-Content -Wait` for log monitoring and an artifact verification block.

- [ ] **Step 2: Add the README entry.**

Link the runbook from the Day 9 dense-retrieval section and state that native Windows remains CPU-only unless FAISS is compiled from source with CUDA.

- [ ] **Step 3: Run documentation checks.**

```powershell
git diff --check
rg -n "faiss-device|faiss-gpu|Tee-Object|Get-Content -Wait|37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f" docs/runbooks/day9-faiss-gpu.md README.md
```

Expected: no whitespace errors and all required command anchors present.

- [ ] **Step 4: Commit the runbook.**

```powershell
git add docs/runbooks/day9-faiss-gpu.md README.md
git commit -m "docs(retrieval): document WSL FAISS GPU build"
```

### Task 5: Run full verification and the real cleanup/build gate

**Files:**
- Modify only generated artifacts under `artifacts/evaluations/day9/` and `data/indexes/`; do not stage them unless explicitly requested.

**Interfaces:**
- Consumes: all committed code/runbook changes and the locked Day 9 corpus.
- Produces: cleanup plan/quarantine evidence, GPU preflight output, BGE-M3 GPU observation/index, and reproducible log.

- [ ] **Step 1: Run static and focused gates.**

```powershell
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync mypy
uv run --frozen --no-sync pytest tests/unit/retrieval tests/integration/retrieval/test_day9_dense_cli.py -q
```

Expected: all commands exit `0`.

- [ ] **Step 2: Execute cleanup dry-run and inspect the JSON.**

```powershell
uv run --frozen --no-sync financial-report-qa retrieval cleanup-day9-data --repo-root . --quarantine-root data/quarantine/day9-cleanup | Tee-Object artifacts/evaluations/day9/data-cleanup-dry-run.jsonl
```

Confirm the canonical release, raw data, manifests, gold, and user-modified QA are protected; confirm blocked candidates are not moved. Only then run the same command with `--apply` if every approved path is expected.

- [ ] **Step 3: Provision WSL2 GPU environment and preflight.**

Run the exact commands from `docs/runbooks/day9-faiss-gpu.md`. Require `faiss.get_num_gpus() == 1`, a successful CUDA driver query, and the pinned BGE-M3 snapshot available locally before starting the long build.

- [ ] **Step 4: Build BGE-M3 with GPU FAISS and capture logs.**

Run the documented WSL wrapper. Verify that the log contains batch progress and `dense-build: complete`, the encoder directory contains `index.faiss` and `manifest.json`, the observation JSON exists, and the manifest has `IndexFlatIP`, the locked fingerprint, and the expected document count `146011`.

- [ ] **Step 5: Run replay/hash and report verification.**

Build the same BGE-M3 index from corpus B, compare `index.faiss` and manifest SHA-256 values, run the existing dense evaluator, and compare against the locked BM25 report. If WSL2 remains inaccessible or the GPU package cannot be provisioned, report the exact blocker and do not claim GPU completion.

- [ ] **Step 6: Final verification report.**

```powershell
git status --short --branch
git log -5 --oneline
Get-Content -Raw artifacts/evaluations/day9/bge-m3-faiss-gpu-build.json
Get-Content -Wait artifacts/evaluations/day9/bge-m3-faiss-gpu-build.log
```

Expected: clean source worktree except explicitly untracked generated artifacts, complete observation, and no protected data removed.
