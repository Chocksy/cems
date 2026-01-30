FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# - curl: For health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy all files needed for install (source must exist for editable install)
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies
RUN uv pip install --system -e .

# Copy deploy files
COPY deploy/ deploy/

# Create non-root user and storage directory
RUN useradd -m -u 1000 cems \
    && mkdir -p /home/cems/.cems \
    && chown -R cems:cems /app /home/cems/.cems
USER cems

# Expose MCP server port
EXPOSE 8765

# Set environment for HTTP mode
ENV CEMS_MODE=http
ENV CEMS_SERVER_HOST=0.0.0.0
ENV CEMS_SERVER_PORT=8765

# Default to OpenRouter embeddings (1536-dim)
# For llama.cpp server embeddings (768-dim), set:
#   CEMS_EMBEDDING_BACKEND=llamacpp_server
#   CEMS_EMBEDDING_DIMENSION=768
#   CEMS_LLAMACPP_BASE_URL=http://llama-embed:8081
ENV CEMS_EMBEDDING_BACKEND=openrouter
ENV CEMS_EMBEDDING_DIMENSION=1536

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run the server
CMD ["python", "-m", "cems.server"]
