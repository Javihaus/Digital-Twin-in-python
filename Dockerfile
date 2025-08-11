# Multi-stage build for production efficiency
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
RUN chown app:app /app

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

USER app

# Copy source code
COPY --chown=app:app . .

# Install package in development mode
RUN pip install -e .

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

USER app

# Copy only necessary files
COPY --chown=app:app src/ src/
COPY --chown=app:app pyproject.toml .
COPY --chown=app:app README.md .
COPY --chown=app:app LICENSE .

# Install package
RUN pip install .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import hybrid_digital_twin; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "from hybrid_digital_twin.cli import main; main()"]

# Training stage (for model training workloads)
FROM production as training

# Install additional ML dependencies
RUN pip install mlflow tensorboard

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Set working directory for training
WORKDIR /app

# Default command for training
CMD ["hybrid-twin", "--help"]

# Inference stage (for serving predictions)
FROM production as inference

# Install serving dependencies
RUN pip install fastapi uvicorn

# Copy inference server (if available)
# COPY --chown=app:app src/hybrid_digital_twin/serving/ src/hybrid_digital_twin/serving/

# Expose API port
EXPOSE 8000

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for inference
CMD ["uvicorn", "hybrid_digital_twin.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]