# Day 9 FAISS Colab T4 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Google Colab notebook that builds the locked Day 9 BGE-M3 `IndexFlatIP` artifact on a T4 GPU using Google Drive for persistent inputs, logs, and outputs.

**Architecture:** The notebook mounts Drive, checks the T4/CUDA and FAISS GPU seam, installs the project into an isolated Python 3.11 environment without replacing GPU FAISS with `faiss-cpu`, and invokes the existing `financial_report_qa` CLI. It performs fail-closed lock/fingerprint checks before the build and validates observation, manifests, document count, and `index.faiss` hash after the build.

**Tech Stack:** Google Colab, Python 3.11, micromamba/Conda, FAISS GPU 1.14.2, `google.colab.drive`, `nbformat`, existing project CLI.

## Global Constraints

- Use the locked fingerprint `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.
- Use `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json` as the release lock.
- Use encoder revision `5617a9f61b028005a4858fdac845db406aefb181` through the existing CLI contract.
- Use `--local-files-only` for model loading and `--faiss-device cuda` for the build.
- Never delete or mutate raw, manifest, QA, canonical, or locked release data from the notebook.
- Persist logs, observation, and index artifacts under the mounted Drive output root.

---

### Task 1: Create the Colab notebook structure

**Files:**
- Create: `notebooks/day9_faiss_gpu_colab_t4.ipynb`

**Interfaces:**
- Consumes: Google Drive mount, project repository, locked corpus and release lock.
- Produces: executable notebook cells with stable `REPO_ROOT`, `DRIVE_ROOT`, `LOCK_PATH`, `FINGERPRINT`, `OUTPUT_ROOT`, and `LOG_PATH` variables.

- [x] **Step 1: Add setup and Drive cells**

  Include cells that mount `/content/drive`, set `DRIVE_ROOT` to `/content/drive/MyDrive/financial-assistant-day9`, copy or access the repository at `/content/financial-assistant`, and fail if the lock path or corpus manifest is absent.

- [x] **Step 2: Add environment installation cell**

  Use micromamba/Conda to create `financial-faiss-gpu` with Python 3.11 and install `faiss-gpu=1.14.2` from `pytorch`, `nvidia`, and `conda-forge`. Export project requirements while excluding `faiss-cpu`, then install the project with `--no-deps`.

- [x] **Step 3: Add GPU preflight cell**

  Run `nvidia-smi` and `faiss.get_num_gpus()` inside the same environment. Raise a clear error unless the runtime is a T4 and FAISS reports at least one GPU.

### Task 2: Add locked build and progress cells

**Files:**
- Modify: `notebooks/day9_faiss_gpu_colab_t4.ipynb`

**Interfaces:**
- Consumes: setup variables from Task 1 and existing `financial_report_qa.cli` retrieval command.
- Produces: persistent JSONL/log output and the BGE-M3 dense index under the Drive output root.

- [x] **Step 1: Add cleanup dry-run cell**

  Invoke `cleanup-day9-data` without `--apply`, print each JSONL decision, and stop if any candidate is `blocked`. Do not invoke quarantine automatically.

- [x] **Step 2: Add build cell**

  Invoke the existing CLI with `python -u`, `--release-lock`, the locked corpus path, `--encoder bge-m3`, `--local-files-only`, `--faiss-device cuda`, and an observation path. Stream stdout/stderr to both notebook output and `bge-m3-faiss-gpu-build.log`.

- [x] **Step 3: Add progress monitoring cell**

  Parse or tail the persisted log and display `encoded/total`, elapsed seconds, and vectors/sec without changing the build process.

### Task 3: Add artifact verification and notebook QA

**Files:**
- Modify: `notebooks/day9_faiss_gpu_colab_t4.ipynb`

**Interfaces:**
- Consumes: completed build observation, corpus/index manifests, release lock, and `index.faiss`.
- Produces: explicit PASS/FAIL verification output and a final Drive artifact summary.

- [x] **Step 1: Verify identity and schema**

  Check observation `faiss_device == "cuda"`, locked fingerprint, release-lock SHA-256, `IndexFlatIP`, document count, and index artifact SHA-256.

- [x] **Step 2: Verify notebook syntax**

  Parse the notebook with `nbformat` and ensure every code cell is syntactically valid Python or an explicit shell cell.

- [x] **Step 3: Verify no unsafe operations**

  Search notebook source for `rm -rf`, `shutil.rmtree`, `--apply`, and writes outside the Drive output root; remove any unsafe occurrence before delivery.

- [x] **Step 4: Commit the notebook and plan**

  Run `git diff --check`, commit the notebook and this plan with `feat(notebooks): add Colab T4 FAISS build workflow`.
