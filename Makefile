UV ?= uv
UV_RUN ?= $(UV) run --frozen --no-sync
UV_LINK_MODE ?= copy
export UV_LINK_MODE

.PHONY: setup lint format format-check typecheck test check build download-data

setup:
	$(UV) sync --frozen --extra dev
	$(UV_RUN) pre-commit install

lint:
	$(UV_RUN) ruff check .

format:
	$(UV_RUN) ruff format .

format-check:
	$(UV_RUN) ruff format --check .

typecheck:
	$(UV_RUN) mypy src tests

test:
	$(UV_RUN) pytest -q

check: lint format-check typecheck test

build:
	$(UV) build --no-build-isolation

download-data:
	$(UV_RUN) financial-report-qa download-data
