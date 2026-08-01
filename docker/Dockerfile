FROM ghcr.io/astral-sh/uv:0.11.28 AS uv

FROM python:3.11-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["financial-report-qa"]
CMD ["--help"]
