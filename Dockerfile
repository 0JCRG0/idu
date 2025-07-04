# Template from https://github.com/astral-sh/uv-docker-example/blob/main/multistage.Dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copy only dependency files first
COPY uv.lock pyproject.toml ./

# Install dependencies without the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project


# Stage 2: Final image
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy ONLY the virtual environment from the builder
# This layer is truly cached since builder's .venv doesn't change with source code
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    DISPLAY=:99 

RUN apt-get update && \
    apt-get install -yqq \
        tesseract-ocr \
        libtesseract-dev \
        libnss3-dev \
        wget \
        curl \
        libgtk-3-0 \
        libx11-xcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application code last - only these layers rebuild when code changes
COPY pyproject.toml ./
COPY data data
COPY chroma chroma
COPY src src

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/healthcheck || exit 1