"""
MCP Client to connect to the @arizeai/phoenix-mcp server.
This allows the Detector agent to query its own failure traces securely from the cloud.
The Reflector Agent logic lives here — it analyzes failures and generates new rules.
"""
import os
import json
import logging
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from opentelemetry import trace

from crucible.agent.prompts import REFLECTOR_SYSTEM_PROMPT
from crucible.agent.llm_client import LLMClient

logger = logging.getLogger("crucible.reflector")
tracer = trace.get_tracer("crucible.reflector")


class PhoenixMCPClient:
    def __init__(self):
        self.llm = LLMClient()

    async def _fetch_from_mcp(self, limit=5):
        """Connects to the official Arize Phoenix MCP server to pull trace metadata."""
        api_key = os.getenv("PHOENIX_API_KEY", "")
        base_url = os.getenv("PHOENIX_HOST") or os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or "https://app.phoenix.arize.com"

        server_params = StdioServerParameters(
            command="npx.cmd" if os.name == 'nt' else "npx",
            args=["-y", "@arizeai/phoenix-mcp@latest"],
            env={**os.environ, "PHOENIX_API_KEY": api_key, "PHOENIX_HOST": base_url}
        )
        
        mcp_context = ""
        try:
            logger.info("[MCP] Connecting to official @arizeai/phoenix-mcp server...")
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # Officially call the MCP tool
                    result = await session.call_tool(
                        "list-traces", 
                        arguments={"project_identifier": "crucible", "limit": limit}
                    )
                    
                    if result.content:
                        # Extract the trace summaries provided by MCP
                        mcp_context = result.content[0].text
                        logger.info("[MCP] Successfully retrieved traces via MCP Protocol.")
        except Exception as e:
            logger.error(f"[MCP] Failed to connect to MCP server: {e}")
            mcp_context = "Could not fetch remote MCP traces."
            
        return mcp_context

    def format_failure_traces(self, failed_reviews=None):
        """Formats the in-memory failure dict into prompt strings."""
        if failed_reviews and isinstance(failed_reviews, dict):
            traces = []
            fps = failed_reviews.get("false_positives", [])
            fns = failed_reviews.get("false_negatives", [])
            
            if fps:
                traces.append("=== FALSE POSITIVES (Real reviews the detector incorrectly flagged as Fake) ===")
                for r in fps:
                    traces.append(f"Text: {r['text']}\\nDetector Reasoning: {r.get('reasoning', 'N/A')}\\n")
            
            if fns:
                traces.append("=== FALSE NEGATIVES (Fake/Deceptive reviews the detector incorrectly flagged as Real) ===")
                for r in fns:
                    traces.append(f"Text: {r['text']}\\nDetector Reasoning: {r.get('reasoning', 'N/A')}\\n")
                    
            return "\\n".join(traces)
        return "No recent failures."

    def _call_llm(self, prompt: str) -> str:
        """Call LLM backend (Gemini or Ollama)."""
        return self.llm.generate_content(
            model='gemini-2.5-flash',
            prompt=prompt
        )

    def summarize_blindspots(self, failed_reviews=None, current_rules="None"):
        """
        Reflector Agent logic: Analyze the failure traces using Gemini to return a new rule.
        Uses asyncio.run() to fetch MCP context without breaking synchronous orchestration loop.
        """
        # 1. Get the explicit failed reviews from local eval memory
        local_traces = self.format_failure_traces(failed_reviews)
        
        # 2. Get remote trace context via official MCP Server integration
        mcp_traces = asyncio.run(self._fetch_from_mcp())
        
        if not local_traces.strip():
            return "No recent failures found. The detector is performing well."
        
        logger.info("[Reflector] Analyzing failures to generate a new rule...")
        
        with tracer.start_as_current_span("ReflectorAgent.generate_rule") as span:
            # Combine the MCP context and local traces
            combined_traces = f"[MCP Trace Metadata from Phoenix Cloud]\\n{mcp_traces}\\n\\n[Exact Failed Review Payloads]\\n{local_traces}"
            
            prompt = REFLECTOR_SYSTEM_PROMPT.format(traces=combined_traces, current_rules=current_rules)
            span.set_attribute("failure_count", len(failed_reviews) if failed_reviews else 0)
            
            try:
                new_rule = self._call_llm(prompt)
            except Exception as e:
                logger.error(f"[Reflector] Gemini call failed after retries: {e}")
                new_rule = "Be more cautious with negative reviews — they are often genuine."

            span.set_attribute("generated_rule", new_rule)
            
            # OpenInference semantic conventions
            try:
                from openinference.semconv.trace import SpanAttributes
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                span.set_attribute(SpanAttributes.INPUT_VALUE, prompt[:500])
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, new_rule)
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "gemini-2.5-flash")
            except Exception:
                pass

        logger.info(f"[Reflector] New rule: {new_rule}")
        return new_rule
