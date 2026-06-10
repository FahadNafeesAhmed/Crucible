import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# We need to set PHOENIX_API_KEY and PHOENIX_HOST (read from environment — never hard-code secrets)
api_key = os.environ.get("PHOENIX_API_KEY", "")
base_url = os.environ.get("PHOENIX_HOST") or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or "https://app.phoenix.arize.com"

async def run():
    server_params = StdioServerParameters(
        command="npx.cmd",  # Use npx.cmd on Windows
        args=[
            "-y",
            "@arizeai/phoenix-mcp@latest",
        ],
        env={**os.environ, "PHOENIX_API_KEY": api_key, "PHOENIX_HOST": base_url}
    )

    print("Connecting to Phoenix MCP...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools = await session.list_tools()
            print("AVAILABLE TOOLS:")
            for tool in tools.tools:
                if tool.name in ["list-traces", "get-spans"]:
                    print(f"- {tool.name}: {tool.inputSchema}")

if __name__ == "__main__":
    asyncio.run(run())
