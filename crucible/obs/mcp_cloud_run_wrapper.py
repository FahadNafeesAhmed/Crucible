from fastapi import FastAPI, HTTPException
import asyncio
import os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import BaseModel

app = FastAPI(title="Crucible MCP Cloud Run Wrapper")

class TraceRequest(BaseModel):
    project_name: str = "crucible"
    limit: int = 50

@app.post("/api/traces")
async def get_traces(req: TraceRequest):
    """
    REST endpoint that wraps the stdio MCP client.
    Google Cloud Agent Builder will call this endpoint.
    """
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@arizeai/phoenix-mcp"],
        env={
            **os.environ,
            "PHOENIX_API_KEY": os.environ.get("PHOENIX_API_KEY", ""),
            "PHOENIX_HOST": os.environ.get("PHOENIX_HOST", "https://app.phoenix.arize.com")
        }
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "list-traces",
                    arguments={
                        "project_name": req.project_name,
                        "limit": req.limit
                    }
                )
                
                return {"traces": result.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Typically Cloud Run runs on port 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
