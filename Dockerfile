FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables (override at deploy time)
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import requests; requests.get('http://localhost:8080/')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "crucible.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
