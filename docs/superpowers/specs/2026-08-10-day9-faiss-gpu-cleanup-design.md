# Day 9 FAISS GPU build and data cleanup design

**Date:** 2026-08-10  
**Status:** Approved for implementation

## Goal

Prepare the locked Day 9 dense-retrieval corpus for a reproducible FAISS GPU build in WSL2/Linux, while removing only superseded or rebuildable data artifacts and preserving the canonical corpus, raw provenance, release lock, and reviewed gold data.

## Scope and invariants

- The canonical release remains `data/processed/release_v2_37a61be7aebd` with fingerprint `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.
- Never delete `data/raw`, `data/manifests`, the canonical processed release, `data/qa/retrieval-gold-v1.jsonl`, or user-modified QA files.
- Candidate old releases are inspected for manifests, fingerprints, references, and access errors before action:
  `release_v2_37a61be7aeba`, `release_v2_7868718f2547`, `release_v2_7fc5d5d57bf6`, and `v2_remediated`.
- Rebuildable interim attempt/replay outputs and incomplete dense-index staging directories are candidates for cleanup. The cleanup command defaults to a dry run and moves approved candidates to a timestamped quarantine directory rather than irreversibly deleting them.
- Cleanup must fail closed if a candidate is unreadable, referenced by the locked Day 9 plan/reports, or resolves outside `data/`.

## FAISS GPU architecture

`build_dense_index` receives a `faiss_device` value (`cpu` or `cuda`). CPU remains the default and preserves the current exact `IndexFlatIP` behavior. In CUDA mode:

1. Verify the imported FAISS module exposes `StandardGpuResources`, `index_cpu_to_gpu`, and `index_gpu_to_cpu`.
2. Allocate `StandardGpuResources` on GPU 0 and create an exact `IndexFlatIP` GPU index.
3. Encode and add normalized `float32` document vectors in existing encoder batches.
4. Convert the completed GPU index back to a CPU `IndexFlatIP` before atomic persistence. This keeps `index.faiss`, manifest validation, and deterministic replay compatible with the current loader.
5. Include `faiss_device` and GPU availability in the operational build observation; deterministic manifest identity remains based on corpus, encoder, and artifact hashes.

The encoder continues to run on CPU in this change. Moving SentenceTransformer inference to CUDA is deliberately out of scope because it is a separate benchmark dimension and would confound the FAISS-only comparison.

## CLI and logging

Add `--faiss-device {cpu,cuda}` to `build-dense-index`, defaulting to `cpu`. The command prints progress at batch boundaries (`encoded/total`, elapsed seconds, vectors/sec) and emits a final artifact path. The documented WSL PowerShell wrapper:

```powershell
$log = "artifacts/evaluations/day9/bge-m3-faiss-gpu-build.log"
& wsl.exe bash -lc "cd /mnt/d/GitHub/financial-assistant && \
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate financial-faiss-gpu && \
  PYTHONPATH=.worktrees/day9-dense/src python -u -m financial_report_qa.cli retrieval \
  build-dense-index --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
  --corpus-dir data/indexes/dense-day9-a/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f/corpus \
  --encoder bge-m3 \
  --output-root data/indexes/dense-day9-a/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f/encoders \
  --observation-path artifacts/evaluations/day9/bge-m3-faiss-gpu-build.json \
  --local-files-only --faiss-device cuda" 2>&1 | Tee-Object -FilePath $log
```

The final implementation will provide the complete command with explicit release lock, corpus, output root, encoder revision, observation path, and `--local-files-only` values. A preflight command must print Python, FAISS version, `get_num_gpus()`, and CUDA availability before a long build starts.

## Error handling

- `--faiss-device cuda` exits non-zero with an actionable error if the GPU-enabled FAISS API is unavailable or reports zero GPUs; it must never silently fall back to CPU.
- `--faiss-device cpu` remains usable with the current Windows `faiss-cpu` environment.
- Existing atomic artifact publication is preserved: no partial encoder directory or observation is published after a failed build.
- Cleanup errors are reported with exact paths and no partial purge is attempted.

## Testing and verification

Test-first coverage will add:

- CUDA mode uses the GPU conversion seam and returns a CPU `IndexFlatIP` suitable for persistence.
- CUDA mode fails closed when GPU symbols are absent or GPU count is zero.
- CPU mode remains byte-compatible with the current deterministic fixture.
- Progress logging contains batch counters and final totals.
- Cleanup dry-run lists only approved candidates, rejects protected canonical paths, and quarantines only paths inside `data/`.

Verification gates:

1. Focused retrieval tests and CLI integration tests.
2. Ruff and mypy on changed source.
3. GPU preflight inside WSL2, including `faiss.get_num_gpus() == 1`.
4. One BGE-M3 GPU build with observation and index manifest present.
5. A/B replay hash comparison and locked dense evaluation, without deleting the canonical release.

## Non-goals

- No change to the locked gold set or retrieval metric definition.
- No irreversible deletion in the first cleanup command.
- No FAISS approximate-index substitution; the baseline remains exact `IndexFlatIP`.
- No native Windows FAISS GPU packaging claim; official prebuilt GPU packages are Linux-only, so WSL2 is required unless FAISS is built from source.
