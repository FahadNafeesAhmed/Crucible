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

# Configure the local Phoenix MCP via Stdio
api_key = os.environ.get("PHOENIX_API_KEY", "")
base_url = "https://app.phoenix.arize.com"
env = {**os.environ, "PHOENIX_API_KEY": api_key, "PHOENIX_HOST": base_url}

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@arizeai/phoenix-mcp@latest"],
    env=env,
)

mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=server_params,
        timeout=30.0
    ),
)

crucible_reflector_agent = LlmAgent(
    name='Crucible_Reflector_Agent',
    model=GlobalGemini(model='gemini-2.5-flash-lite'),
    description=('Agent specialized in introspecting observability telemetry and creating detection rules.'),
    instruction='''You are the Reflector Agent for a deceptive review detection pipeline.
Your job is to analyze the traces of recent evaluation failures to understand what adversarial fakes slipped past our detector.

Follow these steps exactly:
1. Use the `list-traces` tool for the project `crucible`.
2. Look for traces where the detector failed. If you need to see the exact text of the attack, use `get-trace-details`.
3. Analyze the adversarial strategy used to fool the detector.
4. Formulate exactly ONE short, generalized rule/blindspot that will catch this adversarial strategy in the future.

OUTPUT FORMAT: You MUST output ONLY the rule itself as a plain string. Do not include any reasoning, conversational text, or formatting.
Example: "Rule: Watch out for reviews that heavily praise the lobby but are vague about the rooms."
''',
    tools=[mcp_toolset],
)
