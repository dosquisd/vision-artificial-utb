# Stage 1: Builder stage
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app/

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /uvx /bin/

# Configure UV
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Create a virtual environment
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies
COPY ./pyproject.toml ./uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Install the project and its dependencies
COPY ./project/server.py ./project/main.py /app/
COPY ./project/src/ /app/src/
COPY ./project/models/ /app/models/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Stage 2: Final lightweight image
FROM python:3.12-slim-bookworm

WORKDIR /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Copy only the necessary files from the builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/server.py /app/main.py /app/
COPY --from=builder /app/src/ /app/src/
COPY --from=builder /app/models/ /app/models/

# Remove pip cache and other unnecessary files
RUN find /app/.venv -name "__pycache__" -type d -exec rm -rf {} +

EXPOSE 8000

CMD ["uvicorn", "server:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]