import os
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from crucible.obs.mcp_client import PhoenixMCPClient

# 1. Initialize MCP server
mcp_server = Server("crucible-phoenix-mcp-proxy")

@mcp_server.list_tools()
async def list_tools():
    return [
        {
            "name": "list-traces",
            "description": "Fetch the latest observability traces from Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of traces to fetch"}
                }
            }
        }
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "list-traces":
        limit = arguments.get("limit", 50)
        client = PhoenixMCPClient()
        traces = await client._fetch_from_mcp(limit=limit)
        return [{"type": "text", "text": traces}]
    raise ValueError(f"Tool {name} not found")

# 2. Setup SSE Transport
sse = SseServerTransport("/messages/")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            mcp_server.create_initialization_options()
        )
    return Response()

# 3. Configure Starlette routes
routes = [
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse.handle_post_message),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
