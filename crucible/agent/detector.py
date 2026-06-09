"""
The Detector Agent — production-grade with retry logic, structured output, parallel execution, and full tracing.
"""
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from opentelemetry import trace
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from .prompts import DETECTOR_SYSTEM_PROMPT
from crucible.agent.llm_client import LLMClient

logger = logging.getLogger("crucible.detector")
tracer = trace.get_tracer("crucible.detector")


class DetectorAgent:
    def __init__(self, mcp_client=None):
        self.mcp_client = mcp_client
        self.blindspots = "None"
        self.llm = LLMClient()
        
    def update_blindspots(self, failed_reviews=None):
        """
        Uses the MCP client to fetch Phoenix failure traces and update the agent's context.
        Accumulates rules as a numbered list.
        """
        with tracer.start_as_current_span("DetectorAgent.update_blindspots") as span:
            if self.mcp_client:
                new_rule = self.mcp_client.summarize_blindspots(failed_reviews, current_rules=self.blindspots)
                if self.blindspots == "None":
                    self.blindspots = f"1. {new_rule}"
                else:
                    count = len(self.blindspots.strip().split("\n")) + 1
                    self.blindspots += f"\n{count}. {new_rule}"
                span.set_attribute("new_blindspots", self.blindspots)
                logger.info(f"[Detector] Accumulated blindspots via MCP:\n{self.blindspots}")

    def get_system_prompt(self):
        return DETECTOR_SYSTEM_PROMPT.format(blindspots=self.blindspots)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend (Gemini or Ollama)."""
        return self.llm.generate_content(
            model='gemini-2.5-flash-lite',
            prompt=prompt,
            system_instruction=self.get_system_prompt(),
            temperature=0.2
        )

    def _parse_verdict(self, raw_response: str) -> Dict[str, str]:
        """Parse the structured JSON response from Gemini."""
        try:
            # Try to extract JSON from the response
            text = raw_response.strip()
            # Handle cases where Gemini wraps in ```json blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            
            parsed = json.loads(text)
            verdict = parsed.get("verdict", "real").lower().strip()
            reasoning = parsed.get("reasoning", "No reasoning provided.")
            
            if "fake" in verdict:
                verdict = "fake"
            else:
                verdict = "real"
                
            return {"verdict": verdict, "reasoning": reasoning}
        except (json.JSONDecodeError, AttributeError):
            # Fallback: try to extract just the verdict word
            lower = raw_response.lower()
            verdict = "fake" if "fake" in lower else "real"
            return {"verdict": verdict, "reasoning": f"(Raw response: {raw_response[:100]})"}

    def _process_single_review(self, r: tuple) -> Dict[str, Any]:
        """Helper to process a single review, useful for parallel execution."""
        rev_id, prod_id, text, rating, date, reviewer, verified = r
        
        with tracer.start_as_current_span("gemini_evaluation") as child_span:
            child_span.set_attribute("review_text", text)
            child_span.set_attribute("rating", float(rating))
            
            prompt = (
                f"Review Text: '{text}'\n"
                f"Rating: {rating} Stars\n"
                f"Reviewer Profile: {reviewer}\n\n"
                f"Based on the evidence and your current blindspots, "
                f"is this review 'real' or 'fake'? "
                f"Respond with JSON only."
            )
            
            try:
                raw_response = self._call_llm(prompt)
                parsed = self._parse_verdict(raw_response)
                verdict = parsed["verdict"]
                reasoning = parsed["reasoning"]
            except Exception as e:
                logger.error(f"[Detector] FATAL: Gemini call failed after retries: {e}\nExiting immediately to avoid wasting tokens.")
                import os
                os._exit(1)
                
            child_span.set_attribute("verdict", verdict)
            child_span.set_attribute("reasoning", reasoning)
            
            # OpenInference semantic conventions for Arize Phoenix
            try:
                from openinference.semconv.trace import SpanAttributes
                child_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                child_span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
                child_span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{verdict} | {reasoning}")
                child_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "gemini-2.5-flash-lite")
            except Exception:
                pass
    
        return {
            "id": rev_id,
            "text": text,
            "rating": rating,
            "reviewer": reviewer,
            "verdict": verdict,
            "reasoning": reasoning,
            "confidence": 0.90,
            "original_tuple": r
        }

    def analyze_reviews(self, reviews: List[tuple]) -> List[Dict[str, Any]]:
        """
        Analyzes a list of reviews and returns a list of verdicts using Gemini in parallel.
        """
        results = []
        with tracer.start_as_current_span("DetectorAgent.analyze_reviews") as span:
            span.set_attribute("num_reviews", len(reviews))
            
            # Use ThreadPoolExecutor to run API calls in parallel
            import os
            backend = os.getenv("LLM_BACKEND", "gemini").lower()
            workers = 1 if backend == "huggingface" else 5
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Map the reviews to the executor
                future_to_review = {executor.submit(self._process_single_review, r): r for r in reviews}
                
                try:
                    for future in as_completed(future_to_review):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as exc:
                            logger.error(f"[Detector] Review processing generated an exception: {exc}")
                except KeyboardInterrupt:
                    print("\n\n[Crucible] Pipeline forcibly stopped by user (Ctrl+C). Exiting immediately.")
                    import os
                    os._exit(1)
                        
        # Because as_completed doesn't preserve order, we need to sort results back to match the input order
        order_dict = {r[0]: idx for idx, r in enumerate(reviews)}
        results.sort(key=lambda x: order_dict.get(x["id"], 0))
        
        return results
