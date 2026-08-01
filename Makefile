UV ?= uv
UV_LINK_MODE ?= copy
export UV_LINK_MODE

.PHONY: setup lint format format-check typecheck test check build download-data

setup:
	$(UV) sync --extra dev
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy src

test:
	$(UV) run pytest -q -p no:cacheprovider

check: lint format-check typecheck test

build:
	$(UV) build

download-data:
	$(UV) run financial-report-qa download-data
