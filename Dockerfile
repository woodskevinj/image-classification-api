# ======================================================
# 🐳 IMAGE CLASSIFICATION API — Optimized Multi-Stage Dockerfile
# ======================================================

# ---- Stage 1: Builder (installs deps only) ----
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install build tools temporarily for heavy Python deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /root/.cache/pip /var/lib/apt/lists/*

# ---- Stage 2: Runtime ----
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# ---- Copy runtime dependencies only ----
# Copy heavy PyTorch packages in separate layers
COPY --from=builder /install/lib/python3.10/site-packages/torch /usr/local/lib/python3.10/site-packages/torch
COPY --from=builder /install/lib/python3.10/site-packages/torchvision /usr/local/lib/python3.10/site-packages/torchvision

# Copy smaller libraries explicitly (brace expansion is shell-specific, so use individual COPY lines)
RUN mkdir -p /usr/local/lib/python3.10/site-packages
COPY --from=builder /install/lib/python3.10/site-packages/fastapi /usr/local/lib/python3.10/site-packages/fastapi
COPY --from=builder /install/lib/python3.10/site-packages/uvicorn /usr/local/lib/python3.10/site-packages/uvicorn
COPY --from=builder /install/lib/python3.10/site-packages/numpy /usr/local/lib/python3.10/site-packages/numpy
COPY --from=builder /install/lib/python3.10/site-packages/pandas /usr/local/lib/python3.10/site-packages/pandas
COPY --from=builder /install/lib/python3.10/site-packages/PIL /usr/local/lib/python3.10/site-packages/PIL
COPY --from=builder /install/lib/python3.10/site-packages/matplotlib /usr/local/lib/python3.10/site-packages/matplotlib
COPY --from=builder /install/lib/python3.10/site-packages/tqdm /usr/local/lib/python3.10/site-packages/tqdm
COPY --from=builder /install/lib/python3.10/site-packages/python_multipart /usr/local/lib/python3.10/site-packages/python_multipart

# Remove metadata and test folders to shrink image
RUN find /usr/local/lib/python3.10/site-packages -type d -name "tests" -prune -exec rm -rf '{}' + \
 && find /usr/local/lib/python3.10/site-packages -type d -name "__pycache__" -prune -exec rm -rf '{}' + \
 && find /usr/local/lib/python3.10/site-packages -type d -name "*.dist-info" -prune -exec rm -rf '{}' +

# Copy binaries and app source
COPY --from=builder /install/bin /usr/local/bin
COPY api ./api
COPY models ./models
COPY ./*.py ./
COPY requirements.txt .

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
