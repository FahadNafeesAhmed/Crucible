import os
from functools import cached_property

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.genai import Client

from mcp.client.stdio import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from dotenv import load_dotenv
load_dotenv()


class GlobalGemini(Gemini):
    @cached_property
    def api_client(self) -> Client:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[ADK] WARNING: GEMINI_API_KEY not set.")
        return Client(api_key=api_key)


REFLECTOR_INSTRUCTION = '''You are the Reflector Agent for a deceptive review detection pipeline.
Your job is to analyze the traces of recent evaluation failures to understand what adversarial fakes slipped past our detector.

Follow these steps exactly:
1. Use the `list-traces` tool for the project `crucible`.
2. Look for traces where the detector failed. If you need to see the exact text of the attack, use `get-trace-details`.
3. Analyze the adversarial strategy used to fool the detector.
4. Formulate exactly ONE short, generalized rule/blindspot that will catch this adversarial strategy in the future.

OUTPUT FORMAT: You MUST output ONLY the rule itself as a plain string. Do not include any reasoning, conversational text, or formatting.
Example: "Rule: Watch out for reviews that heavily praise the lobby but are vague about the rooms."
'''


def _build_mcp_toolset() -> McpToolset:
    """Create a FRESH Arize Phoenix MCP toolset.

    A new toolset (and therefore a new stdio connection) must be built per
    asyncio.run() invocation: the MCP session binds to the event loop that
    first uses it, and reusing a module-level toolset across loops raises
    "Event loop is closed" on the second round.
    """
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    base_url = (
        os.environ.get("PHOENIX_HOST")
        or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or "https://app.phoenix.arize.com"
    )
    env = {**os.environ, "PHOENIX_API_KEY": api_key, "PHOENIX_HOST": base_url}

    server_params = StdioServerParameters(
        command="npx.cmd" if os.name == "nt" else "npx",
        args=["-y", "@arizeai/phoenix-mcp@latest"],
        env=env,
    )
    return McpToolset(
        connection_params=StdioConnectionParams(server_params=server_params, timeout=30.0)
    )


def _build_reflector_agent(mcp_toolset: McpToolset) -> LlmAgent:
    """Build a fresh Reflector LlmAgent wired to the given MCP toolset."""
    return LlmAgent(
        name="Crucible_Reflector_Agent",
        model=GlobalGemini(model="gemini-2.5-flash-lite"),
        description="Agent specialized in introspecting observability telemetry and creating detection rules.",
        instruction=REFLECTOR_INSTRUCTION,
        tools=[mcp_toolset],
    )


# Module-level agent kept for `adk web` / standalone discovery. The pipeline
# itself builds fresh agents per call via reflect_via_adk_async().
crucible_reflector_agent = _build_reflector_agent(_build_mcp_toolset())


async def reflect_via_adk_async(failed_reviews=None) -> str:
    """
    Run the Google ADK Reflector Agent. The agent autonomously calls the
    Arize Phoenix MCP server (list-traces / get-trace-details) to inspect the
    detector's failure traces and returns exactly ONE new detection rule.

    A fresh MCP toolset + agent are built per call so repeated invocations
    across separate event loops do not hit "Event loop is closed".
    """
    import uuid
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    message = (
        "Please analyze the latest Phoenix traces for the project 'crucible' "
        "using your MCP tools and output exactly ONE new detection rule."
    )
    if failed_reviews:
        message += f"\n\nFor additional context, here are the exact failing reviews:\n{failed_reviews}"

    toolset = _build_mcp_toolset()
    agent = _build_reflector_agent(toolset)
    new_rule = ""
    try:
        sess_id = str(uuid.uuid4())
        runner = InMemoryRunner(agent=agent, app_name="crucible_app")
        await runner.session_service.create_session(
            user_id="crucible", session_id=sess_id, app_name="crucible_app"
        )

        async for event in runner.run_async(
            user_id="crucible",
            session_id=sess_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            ),
        ):
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        new_rule += part.text
    finally:
        # Tear down the MCP stdio connection bound to this event loop.
        try:
            await toolset.close()
        except Exception:
            pass

    return new_rule.strip()


def reflect_via_adk(failed_reviews=None) -> str:
    """Synchronous wrapper around the ADK Reflector Agent."""
    import asyncio

    return asyncio.run(reflect_via_adk_async(failed_reviews=failed_reviews))
