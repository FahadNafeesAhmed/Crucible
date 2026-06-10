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


REFLECTOR_INSTRUCTION = '''You are the Reflector Agent for a deceptive hotel-review detection pipeline.
The Detector keeps getting fooled by AI-Forged fake reviews, and your job is to write the ONE
new detection rule that would have caught the most recent failures.

KNOWN FAILURE MODE: the Forger defeats the Detector by stuffing fakes with *specific, plausible
details* (named theaters, restaurants, room numbers, minor complaints about AC or Wi-Fi, a reason
for the trip). The Detector wrongly treats "specific detail" as proof of authenticity. Your rules
must break that assumption.

Follow these steps exactly:
1. Call the `list-traces` tool for the project `crucible`.
2. For the failed evaluations, call `get-trace-details` to read the exact fake text and the
   Detector's reasoning.
3. Identify the CONCRETE textual signal the fakes share that the Detector missed.
4. Write exactly ONE new rule that is:
   - CONCRETE and TESTABLE: name the specific signal to check (e.g. "treat brand-name landmark +
     generic complaint combos as suspicious unless the detail is internally verifiable"), NOT vague
     advice like "be more careful".
   - GENERAL enough to catch the whole strategy, not just one review.
   - DIFFERENT from any rule already in the current rule set (do not restate or rephrase).
   - A counter to the specificity trick: remind the Detector that fabricated specifics are cheap,
     so it should weigh internal consistency, plausibility, and verifiability over mere detail.

OUTPUT FORMAT: Output ONLY the rule as a single plain sentence. No preamble, no numbering, no
quotes, no reasoning. Start with an imperative verb.
Example: Flag reviews that pair a famous nearby landmark with a vague, generic complaint, since
forgers use real place-names to fake authenticity.
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


async def reflect_via_adk_async(failed_reviews=None, current_rules="None") -> str:
    """
    Run the Google ADK Reflector Agent. The agent autonomously calls the
    Arize Phoenix MCP server (list-traces / get-trace-details) to inspect the
    detector's failure traces and returns exactly ONE new detection rule.

    A fresh MCP toolset + agent are built per call so repeated invocations
    across separate event loops do not hit "Event loop is closed".

    `current_rules` lets the agent avoid duplicating rules the Detector already has.
    """
    import uuid
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    message = (
        "Analyze the latest Phoenix traces for the project 'crucible' using your MCP tools, "
        "then output exactly ONE new, concrete detection rule that counters the failures."
    )
    if current_rules and current_rules != "None":
        message += f"\n\nRules the Detector ALREADY has (do NOT repeat or rephrase these):\n{current_rules}"
    if failed_reviews:
        message += f"\n\nThe exact reviews that just fooled the Detector:\n{failed_reviews}"

    toolset = _build_mcp_toolset()
    agent = _build_reflector_agent(toolset)
    # Keep only the agent's FINAL text turn — not its intermediate tool-call
    # narration / scratchpad — so we store just the rule, never the reasoning.
    last_text = ""
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
            if hasattr(event, "content") and event.content and event.content.parts:
                cur = "".join(
                    p.text for p in event.content.parts
                    if hasattr(p, "text") and p.text
                )
                if cur.strip():
                    last_text = cur  # overwrite — final non-empty turn wins
    finally:
        # Tear down the MCP stdio connection bound to this event loop.
        try:
            await toolset.close()
        except Exception:
            pass

    return last_text.strip()


def reflect_via_adk(failed_reviews=None, current_rules="None") -> str:
    """Synchronous wrapper around the ADK Reflector Agent."""
    import asyncio

    return asyncio.run(
        reflect_via_adk_async(failed_reviews=failed_reviews, current_rules=current_rules)
    )
