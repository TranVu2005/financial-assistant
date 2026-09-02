FROM ghcr.io/astral-sh/uv:0.12.9 AS uv

FROM python:3.11-slim AS builder

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --extra dev --no-install-project
RUN uv build --no-build-isolation --wheel

FROM python:3.11-slim AS runtime

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --from=builder /app/dist/*.whl /tmp/
RUN uv pip install --python /app/.venv/bin/python --no-deps /tmp/*.whl

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["financial-report-qa"]
CMD ["--help"]
