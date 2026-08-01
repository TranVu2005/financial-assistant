# Modular Monolith Development Environment Design

## Goal

Turn the current Python skeleton into a reproducible product-development repository for a
Vietnamese financial-report QA system. WSL2/Linux is the canonical developer environment;
Windows remains the host and may store large data under the mounted `D:` drive.

## Architecture decision

Use a modular monolith. All installable application code belongs to one Python package named
`financial_report_qa`; domain modules live below that namespace and communicate through explicit
Python interfaces. Data, tests, configuration, operational scripts, and documentation remain
outside `src/` because they are project assets rather than importable application code.

Microservices, Kubernetes, cloud infrastructure, authentication, and production deployment are
outside this foundation. They will be considered only after an end-to-end local product exists
and measurements show a concrete need.

## Repository boundary

`D:\GitHub\financial-assistant` is an independent Git repository. Work is performed on
`codex/modular-foundation`, preventing this project from being committed together with sibling
directories under `D:\GitHub`.

## Package boundaries

```text
src/financial_report_qa/
├── core/           shared configuration, typed errors, and logging setup
├── data/           dataset acquisition and immutable raw-data inventory
├── schemas/        stable Pydantic contracts shared between domain modules
├── ingestion/      TXT/HTML-table parsing and provenance capture
├── normalization/  company, period, metric, number, and unit normalization
├── retrieval/      lexical, vector, fusion, and graph-aware table retrieval
├── planning/       question parsing and constrained query plans
├── execution/      deterministic compilation, execution, and verification
├── evaluation/     metrics, run artifacts, and error analysis
└── cli.py          product command-line entry point
```

Only `core`, `data`, and the CLI contain executable foundation code in this migration. Future
domain packages are initialized with a documented responsibility and receive implementation only
when their milestone starts.

## Data lifecycle

```text
external source
  -> data/raw (immutable, ignored by Git)
  -> data/interim (rebuildable extraction artifacts)
  -> data/processed (canonical Parquet datasets)
  -> data/indexes (rebuildable retrieval indexes)
```

Small manifests, schemas, licensed fixtures, and QA annotations may be committed. Raw reports,
generated tables, indexes, model files, secrets, and evaluation outputs are ignored. The existing
Hugging Face downloader remains revision-pinned, resumable, capacity-checked, and safe-by-default
with dry-run behavior.

## Developer interface

- Python: exactly 3.11, managed by `uv` and `.python-version`.
- Canonical shell: Bash in WSL2/Linux.
- Common commands: `make setup`, `make test`, `make lint`, `make typecheck`, `make check`,
  `make build`, and `make download-data`.
- Configuration: public defaults in YAML and `.env.example`; real secrets remain in `.env`.
- Packaging: Hatchling explicitly builds `src/financial_report_qa` and exposes the
  `financial-report-qa` console command.
- Containers: a small Dockerfile verifies that the product can be built in Linux without relying
  on the developer's global Python environment.

## Quality gates

Local and CI gates are the same:

1. Ruff linting.
2. Strict mypy checking over `src` and `tests`.
3. Pytest unit/integration suite.
4. Wheel and source-distribution build.

Pre-commit runs fast lint, formatting, type, and unit-test checks before a commit. GitHub Actions
runs the complete gate on pushes and pull requests. Dependabot monitors Python and GitHub Actions
dependencies weekly.

## Error handling and observability

Domain failures derive from one `FinancialReportQAError` base type. CLI commands translate known
domain errors into concise stderr messages and non-zero exit codes. Logging uses the standard
library with environment-controlled levels and consistent timestamps; business modules receive
module loggers rather than configuring global output themselves.

## Migration compatibility

The distribution remains named `financial-assistant`, while the import namespace changes from
`financial_assistant` to `financial_report_qa`. The existing dataset downloader behavior, tests,
Unicode paths, and WSL wrapper remain intact. Coverage, scripts, tests, and Hatchling package
selection change together so the old namespace is fully removed.

## Success criteria

- A clean WSL checkout can run `make setup && make check && make build`.
- `financial-report-qa --help` and `financial-report-qa download-data --help` work.
- Dataset dry-run remains the default; transfer still requires `--download`.
- CI expresses the same lint, type, test, and build gates used locally.
- No large data, cache, model, secret, or generated artifact becomes tracked.
