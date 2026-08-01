# Modular Monolith Development Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the project to a WSL-first modular-monolith Python product repository without changing the safe dataset-download behavior.

**Architecture:** Build one explicitly packaged `financial_report_qa` namespace with domain modules below it, thin external scripts, and identical local/CI quality gates. Keep raw and generated data outside the package and ignored by Git.

**Tech Stack:** Python 3.11, uv, Hatchling, Pydantic Settings, pytest, Ruff, mypy, pre-commit, GNU Make, Docker, GitHub Actions.

## Global Constraints

- WSL2/Linux is the canonical development and CI environment.
- The distribution name stays `financial-assistant`; the import namespace is `financial_report_qa`.
- Dataset transfer must require `--download`; the default command performs only a dry run.
- Do not move, rewrite, or commit anything under `data/raw`, generated data directories, `models`, `artifacts`, or `submissions`.
- Do not add production cloud infrastructure, microservices, authentication, or Kubernetes.

---

### Task 1: Migrate the Python package and preserve downloader behavior

**Files:**
- Create: `src/financial_report_qa/__init__.py`
- Create: `src/financial_report_qa/data/__init__.py`
- Move: `src/financial_assistant/dataset_download.py` to `src/financial_report_qa/data/download.py`
- Create: `src/financial_report_qa/cli.py`
- Modify: `scripts/download_dataset.py`
- Modify: `tests/test_environment.py`
- Modify: `tests/unit/test_dataset_download.py`
- Modify: `pyproject.toml`
- Delete: `src/financial_assistant/__init__.py`

**Interfaces:**
- Consumes: existing `DownloadRequest`, `DownloadPlan`, `build_download_plan`, and `download_dataset` behavior.
- Produces: import namespace `financial_report_qa` and console entry point `financial-report-qa` with subcommand `download-data`.

- [ ] Change tests to import `financial_report_qa` and `financial_report_qa.data.download`.
- [ ] Add a CLI test asserting `main(["download-data", "--target", PATH])` performs only the dry run.
- [ ] Run `python -m pytest tests/test_environment.py tests/unit/test_dataset_download.py -q -p no:cacheprovider`; expect collection failure because the new package does not exist.
- [ ] Move the downloader, implement the CLI dispatcher, and update the thin script import.
- [ ] Configure `[project.scripts]`, explicit Hatch wheel packages, and coverage source in `pyproject.toml`.
- [ ] Re-run the focused tests; expect all to pass.

### Task 2: Add the product foundation and module boundaries

**Files:**
- Create: `src/financial_report_qa/core/config.py`
- Create: `src/financial_report_qa/core/errors.py`
- Create: `src/financial_report_qa/core/logging.py`
- Create package initializers for `schemas`, `ingestion`, `normalization`, `retrieval`, `planning`, `execution`, and `evaluation`.
- Create: `tests/unit/core/test_config.py`
- Create: `tests/unit/core/test_logging.py`

**Interfaces:**
- Produces: `Settings.load()` for environment-backed project settings, `FinancialReportQAError`, and `configure_logging(level)`.

- [ ] Write tests asserting environment variables override defaults and invalid log levels are rejected.
- [ ] Run the focused core tests; expect import failure.
- [ ] Implement minimal typed settings, error, and logging modules.
- [ ] Re-run focused core tests; expect all to pass.
- [ ] Add one-sentence module responsibility docstrings to future domain package initializers.

### Task 3: Add reproducible developer tooling

**Files:**
- Create: `.editorconfig`
- Create: `.pre-commit-config.yaml`
- Create: `.dockerignore`
- Create: `Makefile`
- Create: `docker/Dockerfile`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `make setup`, `make lint`, `make typecheck`, `make test`, `make check`, `make build`, and `make download-data`.

- [ ] Add `pre-commit` to the dev dependency group and update the universal lockfile.
- [ ] Define Make targets using only locked `uv run` commands.
- [ ] Add local pre-commit hooks for Ruff, Ruff format checking, mypy, and fast tests.
- [ ] Add a Python 3.11 slim Dockerfile using the pinned uv version and the built console command.
- [ ] Run `make check` from Bash when available; otherwise run each underlying command directly.

### Task 4: Add CI, dependency automation, and product documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/dependabot.yml`
- Create: `docs/architecture.md`
- Create: `docs/development.md`
- Create: `docs/decisions/0001-modular-monolith.md`
- Modify: `docs/data-download.md`
- Replace: `README.md`

**Interfaces:**
- Produces: Linux CI matching local gates and a README path from fresh checkout to dataset dry-run.

- [ ] Define CI jobs for locked dependency sync, Ruff, mypy, pytest, and package build.
- [ ] Configure weekly dependency checks for uv/pip and GitHub Actions.
- [ ] Document architecture boundaries, WSL setup, command reference, data lifecycle, and the modular-monolith decision.
- [ ] Rewrite README in valid UTF-8 Vietnamese with setup, structure, commands, and documentation links.
- [ ] Verify every README command maps to an existing Make target or CLI command.

### Task 5: Verify and establish repository baseline

**Files:**
- Verify all files above.

**Interfaces:**
- Produces: a clean, buildable baseline on branch `codex/modular-foundation`.

- [ ] Run `uv lock --check` and confirm the lockfile is current.
- [ ] Run `python -m ruff check .` and confirm zero errors.
- [ ] Run `python -m ruff format --check .` and confirm no formatting drift.
- [ ] Run `python -m mypy src` and confirm strict typing passes.
- [ ] Run `python -m pytest -q -p no:cacheprovider` and confirm all tests pass.
- [ ] Run `uv build` and confirm both wheel and source distribution are created.
- [ ] Run the CLI help and dataset dry-run unit tests without downloading real data.
- [ ] Review `git status` to confirm caches, large data, models, and build artifacts are ignored.
