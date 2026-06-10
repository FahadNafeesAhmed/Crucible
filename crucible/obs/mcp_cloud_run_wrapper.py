"""
Self-contained MCP SSE Server for Cloud Run.

This wraps the @arizeai/phoenix-mcp stdio server as an SSE endpoint
so Google Agent Builder can connect to it via the MCP Server button.

IMPORTANT: This file must NOT import any crucible.agent.* modules.
"""
import os
import json
import logging
import uvicorn

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-wrapper")

# --- MCP Server (what Agent Builder connects to) ---
mcp_server = Server("crucible-phoenix-proxy")


async def _call_phoenix_mcp(tool_name: str, arguments: dict) -> str:
    """Spawn the real @arizeai/phoenix-mcp via npx and call a tool."""
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    base_url = "https://app.phoenix.arize.com"

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@arizeai/phoenix-mcp@latest"],
        env={**os.environ, "PHOENIX_API_KEY": api_key, "PHOENIX_HOST": base_url},
    )

    logger.info(f"[Proxy] Calling phoenix-mcp tool '{tool_name}' with args: {arguments}")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            if result.content:
                return result.content[0].text
            return json.dumps({"status": "ok", "data": []})


@mcp_server.list_tools()
async def list_tools():
    """Proxy the tool list from the real Phoenix MCP server."""
    return [
        {
            "name": "get-trace-details",
            "description": "Get detailed information about a specific trace from Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string", "description": "The trace ID to look up"},
                    "project_identifier": {"type": "string", "description": "The project name or ID"},
                },
                "required": ["trace_id"],
            },
        },
        {
            "name": "list-traces",
            "description": "List the latest traces from Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_identifier": {
                        "type": "string",
                        "description": "The project name or ID",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of traces to return",
                        "default": 10,
                    },
                },
                "required": ["project_identifier"],
            },
        },
        {
            "name": "list-experiments",
            "description": "List experiments for a dataset in Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "The dataset ID"},
                },
                "required": ["dataset_id"],
            },
        },
        {
            "name": "list-datasets",
            "description": "List all datasets in Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "list-projects",
            "description": "List all projects in Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "list-prompts",
            "description": "List all prompts in Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get-prompt",
            "description": "Get a specific prompt from Arize Phoenix",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt_identifier": {"type": "string", "description": "The prompt name or ID"},
                },
                "required": ["prompt_identifier"],
            },
        },
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Forward every tool call to the real Phoenix MCP server."""
    try:
        result_text = await _call_phoenix_mcp(name, arguments)
        return [{"type": "text", "text": result_text}]
    except Exception as e:
        logger.error(f"[Proxy] Error calling tool '{name}': {e}")
        return [{"type": "text", "text": f"Error: {str(e)}"}]


# --- SSE Transport (what Agent Builder connects over) ---
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            mcp_server.create_initialization_options(),
        )


# --- Health check endpoint ---
async def health(request):
    return JSONResponse({"status": "healthy", "service": "crucible-phoenix-mcp-proxy"})


# --- Starlette App ---
routes = [
    Route("/", endpoint=health),
    Route("/health", endpoint=health),
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse.handle_post_message),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting MCP SSE proxy on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
