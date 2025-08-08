# path: Dockerfile
FROM python:3.11-slim

# System deps (for chroma/hnswlib and parsers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY app ./app

# Create runtime dirs (also mounted via volumes in compose)
RUN mkdir -p /data/chroma /data/files

EXPOSE 8000
CMD ["uvicorn", "app.memory_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

