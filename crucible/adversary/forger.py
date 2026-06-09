"""
The Forger Agent — production-grade with retry logic and full tracing.
This adversarial agent uses Gemini to generate fake hotel reviews designed to fool the Detector.
"""
import logging

from opentelemetry import trace

from crucible.agent.prompts import FORGER_SYSTEM_PROMPT
from crucible.agent.llm_client import LLMClient

logger = logging.getLogger("crucible.forger")
tracer = trace.get_tracer("crucible.forger")


class ForgerAgent:
    def __init__(self):
        self.iteration = 0
        self.llm = LLMClient()

    def _call_llm(self, prompt: str, detector_rules: str) -> str:
        """Call LLM backend (Gemini or Ollama)."""
        return self.llm.generate_content(
            model='gemini-2.5-flash-lite',
            prompt=prompt,
            system_instruction=FORGER_SYSTEM_PROMPT.format(detector_rules=detector_rules),
            temperature=0.8
        )

    def generate_fakes(self, hotel_name: str = "Conrad Chicago", detector_blindspots: str = "None", count: int = 2) -> list:
        """
        Generates fake reviews using Gemini. If it knows the detector's blindspots, it exploits them.
        """
        self.iteration += 1
        fakes = []
        
        logger.info(f"[Forger] Generating {count} adversarial fakes for '{hotel_name}' (Round {self.iteration})...")
        
        for idx in range(count):
            prompt = f"Hotel Name: {hotel_name}\nDetector's Current Blindspots: {detector_blindspots}\n\nWrite a completely fake, deceptive review."
            
            with tracer.start_as_current_span("ForgerAgent.generate_fake") as span:
                span.set_attribute("hotel_name", hotel_name)
                span.set_attribute("iteration", self.iteration)
                span.set_attribute("fake_index", idx)
                
                try:
                    fake_text = self._call_llm(prompt, detector_rules=detector_blindspots)
                except KeyboardInterrupt:
                    print("\n\n[Crucible] Pipeline forcibly stopped by user (Ctrl+C). Exiting immediately.")
                    import os
                    os._exit(1)
                except Exception as e:
                    logger.error(f"[Forger] FATAL: Gemini call failed after retries: {e}\nExiting immediately to avoid wasting tokens.")
                    import os
                    os._exit(1)

                span.set_attribute("generated_text", fake_text[:500])
                
                # OpenInference semantic conventions for Arize Phoenix
                try:
                    from openinference.semconv.trace import SpanAttributes
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                    span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, fake_text)
                    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "gemini-2.5-flash-lite")
                except Exception:
                    pass
                    
            fakes.append(fake_text)
            
        return fakes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forger = ForgerAgent()
    hotel = "The Palmer House Hilton"
    print(f"Testing Forger Agent on {hotel}...")
    
    fakes = forger.generate_fakes(hotel_name=hotel, count=1)
    print(f"\n--- Round 1 Fake Review ---")
    print(fakes[0])
    
    blindspot = "The detector flags reviews that are too perfectly formatted and have no typos."
    fakes_smart = forger.generate_fakes(hotel_name=hotel, detector_blindspots=blindspot, count=1)
    print(f"\n--- Round 2 Smart Fake Review (Exploiting Typo Blindspot) ---")
    print(fakes_smart[0])
