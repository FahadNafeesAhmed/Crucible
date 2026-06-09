# Cloud Run Deployment Instructions

## What Changed

The MCP wrapper was completely rewritten to be **self-contained**. The previous version crashed on startup because it imported the full Crucible agent stack (`google-genai`, `torch`, etc.) which requires API keys and heavy dependencies not present in the MCP container.

The new version:
- Has **zero imports** from `crucible.agent.*`
- Only installs minimal deps: `mcp`, `starlette`, `uvicorn`, `httpx`
- Directly proxies to `@arizeai/phoenix-mcp` via `npx`
- Includes a health check endpoint at `/` and `/health`

---

## Deploy Steps (run in Google Cloud Shell)

### 1. Pull the latest code
```bash
cd ~/Crucible
git pull
```

### 2. Copy the MCP Dockerfile as the main Dockerfile
```bash
cp Dockerfile.mcp Dockerfile
```

### 3. Deploy to Cloud Run
```bash
gcloud run deploy mcp-wrapper \
  --source . \
  --set-env-vars PHOENIX_API_KEY="YOUR_KEY_HERE" \
  --allow-unauthenticated \
  --region us-central1 \
  --memory 1Gi \
  --timeout 300
```

Replace `YOUR_KEY_HERE` with your actual Phoenix API key.

### 4. Connect to Agent Builder
Once deployed, Cloud Run will give you a URL like:
```
https://mcp-wrapper-XXXXX-uc.a.run.app
```

In Google Agent Builder:
1. Click the **MCP Server [+]** button
2. Paste the URL with `/sse` appended:
   ```
   https://mcp-wrapper-XXXXX-uc.a.run.app/sse
   ```
3. The tools from Phoenix MCP should appear automatically

---

## Troubleshooting

If the deploy fails, check the build logs:
```bash
gcloud builds log $(gcloud builds list --limit=1 --format="value(id)")
```

If the container starts but tools don't work, check runtime logs:
```bash
gcloud run services logs read mcp-wrapper --region us-central1 --limit 50
```
