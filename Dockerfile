# ======================================================
# 🐳 IMAGE CLASSIFICATION API — DOCKERFILE
# Author: Kevin Woods
# ======================================================

# ---- Base Image ----
FROM python:3.10-slim

# ---- Set Environment Variables ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

# ---- Create & Set Working Directory ----
WORKDIR $APP_HOME

# ---- Install System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy Project Files ----
COPY . .

# ---- Install Python Dependencies ----
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---- Expose Port ----
EXPOSE 8000

# ---- Define Entrypoint Command ----
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
