# path: Dockerfile
FROM python:3.11-slim

# Build tools for any deps that need compiling
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY app ./app

# Expose port used by Uvicorn
ENV PORT=10000
EXPOSE 10000

# Start FastAPI
CMD ["uvicorn", "app.memory_api:app", "--host", "0.0.0.0", "--port", "10000"]

