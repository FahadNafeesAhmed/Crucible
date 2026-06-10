FROM python:3.11-slim

# Install Node.js for npx (required to spawn @arizeai/phoenix-mcp)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn mcp pandas

# Copy code
COPY crucible/ crucible/

# Pre-install the Phoenix MCP package so first call is fast
RUN npx -y @arizeai/phoenix-mcp@latest --version || true

EXPOSE 8080

# Run the OpenAPI backend
CMD ["uvicorn", "crucible.obs.crucible_api:app", "--host", "0.0.0.0", "--port", "8080"]
