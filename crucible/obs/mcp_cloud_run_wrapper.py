"""
Self-contained MCP Streamable HTTP Server for Cloud Run.

Uses FastMCP with streamable-http transport so Google ADK's
StreamableHTTPConnectionParams can connect to it directly.

IMPORTANT: This file must NOT import any crucible.agent.* modules.
"""
import os
import json
import logging

from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-wrapper")

mcp = FastMCP("crucible-phoenix-proxy")


async def _call_phoenix_mcp(tool_name: str, arguments: dict) -> str:
    """Spawn the real @arizeai/phoenix-mcp via npx and call a tool."""
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    base_url = "https://app.phoenix.arize.com"

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@arizeai/phoenix-mcp@latest"],
        env={
            **os.environ,
            "PHOENIX_API_KEY": api_key,
            "PHOENIX_HOST": base_url,
        },
    )

    logger.info(f"[Proxy] Calling phoenix-mcp tool '{tool_name}' with args: {arguments}")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            if result.content:
                return result.content[0].text
            return json.dumps({"status": "ok", "data": []})


# --- Tool definitions (proxied to real Phoenix MCP) ---

@mcp.tool()
async def list_traces(project_identifier: str = "crucible", limit: int = 10) -> str:
    """List the latest traces from Arize Phoenix."""
    return await _call_phoenix_mcp(
        "list-traces",
        {"project_identifier": project_identifier, "limit": limit},
    )


@mcp.tool()
async def get_trace_details(trace_id: str, project_identifier: str = "crucible") -> str:
    """Get detailed information about a specific trace from Arize Phoenix."""
    return await _call_phoenix_mcp(
        "get-trace-details",
        {"trace_id": trace_id, "project_identifier": project_identifier},
    )


@mcp.tool()
async def list_projects() -> str:
    """List all projects in Arize Phoenix."""
    return await _call_phoenix_mcp("list-projects", {})


@mcp.tool()
async def list_datasets() -> str:
    """List all datasets in Arize Phoenix."""
    return await _call_phoenix_mcp("list-datasets", {})


@mcp.tool()
async def list_experiments(dataset_id: str) -> str:
    """List experiments for a dataset in Arize Phoenix."""
    return await _call_phoenix_mcp("list-experiments", {"dataset_id": dataset_id})


@mcp.tool()
async def list_prompts() -> str:
    """List all prompts in Arize Phoenix."""
    return await _call_phoenix_mcp("list-prompts", {})


@mcp.tool()
async def get_prompt(prompt_identifier: str) -> str:
    """Get a specific prompt from Arize Phoenix."""
    return await _call_phoenix_mcp(
        "get-prompt", {"prompt_identifier": prompt_identifier}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting MCP Streamable HTTP proxy on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
