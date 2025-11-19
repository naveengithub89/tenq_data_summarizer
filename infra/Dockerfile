# Lightweight base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (for building wheels etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy full project context (we'll set context to repo root in docker-compose)
COPY . .

# Upgrade pip and install project with dev extras (for tests / tools)
RUN pip install --upgrade pip && \
    pip install ".[dev]"

# Expose API port
EXPOSE 8000

# Default command: run FastAPI via uvicorn
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
