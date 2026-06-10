import os
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

import asyncio
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crucible-api")

app = FastAPI(
    title="Crucible API",
    description="Backend API for Crucible Agent Builder, exposing Phoenix observability and datasets.",
    version="1.0.0",
)

async def _call_phoenix_mcp(tool_name: str, arguments: dict) -> str:
    """Spawn the real @arizeai/phoenix-mcp via npx and call a tool.
    We proxy via the Phoenix MCP CLI because it already implements the complex GraphQL/REST calls to Phoenix.
    """
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    base_url = "https://app.phoenix.arize.com"
    env = {**os.environ, "PHOENIX_API_KEY": api_key, "PHOENIX_HOST": base_url}

    # Instead of full MCP stdio bridging, we can just run the CLI directly for simple commands if possible,
    # or we can use the mcp python client to call it over stdio internally.
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession, StdioServerParameters

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@arizeai/phoenix-mcp@latest"],
        env=env,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.content:
                    return result.content[0].text
                return json.dumps({"status": "ok", "data": []})
    except Exception as e:
        logger.error(f"Error calling Phoenix MCP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/phoenix/traces")
async def list_traces(project_identifier: str = "crucible", limit: int = 10):
    """List the latest traces from Arize Phoenix."""
    result = await _call_phoenix_mcp("list-traces", {"project_identifier": project_identifier, "limit": limit})
    try:
        return json.loads(result)
    except:
        return {"raw": result}

@app.get("/api/phoenix/traces/{trace_id}")
async def get_trace_details(trace_id: str, project_identifier: str = "crucible"):
    """Get detailed information about a specific trace."""
    result = await _call_phoenix_mcp("get-trace-details", {"trace_id": trace_id, "project_identifier": project_identifier})
    try:
        return json.loads(result)
    except:
        return {"raw": result}

@app.get("/api/dataset/sample")
def get_sample_review():
    """Fetch a random sample review from the Ott benchmark dataset to evaluate."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
        from crucible.data.load_ott import load_ott_data
        df = load_ott_data()
        sample = df.sample(1).iloc[0]
        return {
            "text": sample["text"],
            "hotel": sample["hotel"],
            "true_label": "fake" if sample["deceptive"] == "deceptive" else "real"
        }
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Could not load sample review")

class BlindspotUpdate(BaseModel):
    new_rule: str

@app.post("/api/phoenix/blindspots")
def update_blindspots(update: BlindspotUpdate):
    """Save a new rule/blindspot learned by the Reflector."""
    # In a real app, this would save to a database. For Agent Builder, we return it so the orchestration agent can pass it to the Forger.
    logger.info(f"New blindspot learned: {update.new_rule}")
    return {"status": "success", "rule_added": update.new_rule}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
