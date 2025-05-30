FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app/

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /uvx /bin/

# Set uv options for better performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy only requirements file first to leverage Docker caching
COPY ./requirements-docker.txt /app/

# Create virtual environment and install dependencies with caching
RUN uv venv .venv -p python3.12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir -r requirements-docker.txt

# Copy only the necessary application files
COPY ./project/src/ /app/src/
COPY ./project/models/ /app/models/
COPY ./project/static/ /app/static/
COPY ./project/server.py ./project/main.py ./project/.env /app/

RUN yolo settings datasets_dir=/app/models

# Expose the port the app runs on
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
