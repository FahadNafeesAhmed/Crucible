"""
Abstracted LLM Client for Crucible.
Switches between Google Gemini API and Local Ollama (Gemma 3) based on LLM_BACKEND env var.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import APIError, ClientError

logger = logging.getLogger("crucible.llm")

# Silence noisy HTTP logs from the SDK
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# Global Gemini Client initialized exactly ONCE at import time to prevent thread race conditions
GEMINI_CLIENT = genai.Client()

# Try to import ollama for local testing
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import HuggingFace transformers
try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    HF_AVAILABLE = True
    
    # We will initialize the HF model and processor lazily so it doesn't crash on boot if the GPU is busy
    HF_MODEL_OBJ = None
    HF_PROCESSOR = None
except ImportError:
    HF_AVAILABLE = False

class LLMClient:
    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "gemini").lower()
        if self.backend == "ollama" and not OLLAMA_AVAILABLE:
            logger.warning("[LLMClient] LLM_BACKEND=ollama but 'ollama' package is not installed. Falling back to gemini.")
            self.backend = "gemini"
        elif self.backend == "huggingface" and not HF_AVAILABLE:
            logger.warning("[LLMClient] LLM_BACKEND=huggingface but 'transformers'/'torch' is not installed. Falling back to gemini.")
            self.backend = "gemini"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"[LLM] Gemini 429 — retrying in {retry_state.next_action.sleep:.0f}s (attempt {retry_state.attempt_number}/5)"
        ),
    )
    def _call_gemini(self, model: str, prompt: str, system_instruction: str = None, temperature: float = 0.2) -> str:
        """Single Gemini call with retry logic."""
        global GEMINI_CLIENT
            
        config_kwargs = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
            
        response = GEMINI_CLIENT.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs)
        )
        return response.text.strip()

    def _call_ollama(self, model: str, prompt: str, system_instruction: str = None, temperature: float = 0.2) -> str:
        """Call local Ollama."""
        # For local testing, we override the Gemini model name with the Ollama model
        local_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=local_model,
            messages=messages,
            options={"temperature": temperature}
        )
        return response['message']['content'].strip()

    def _call_huggingface(self, prompt: str, system_instruction: str = None, temperature: float = 0.2) -> str:
        """Call local HuggingFace model."""
        global HF_MODEL_OBJ, HF_PROCESSOR
        model_id = os.getenv("HF_MODEL", "google/gemma-3-4b-it")
        
        if HF_MODEL_OBJ is None:
            logger.info(f"[LLM] Initializing HuggingFace Gemma 4 model for {model_id}...")
            try:
                HF_PROCESSOR = AutoProcessor.from_pretrained(model_id)
                HF_MODEL_OBJ = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            except Exception as e:
                err_str = str(e).lower()
                if "401" in err_str or "gated repo" in err_str or "unauthorized" in err_str or "access to model" in err_str:
                    print(f"\n\n[FATAL ERROR] You do not have access to {model_id}.")
                    print("This is a gated model. Please run 'huggingface-cli login' in your terminal and provide your access token.")
                    os._exit(1)
                raise e
            
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        inputs = HF_PROCESSOR.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(HF_MODEL_OBJ.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        outputs = HF_MODEL_OBJ.generate(**inputs, max_new_tokens=1024, temperature=temperature, do_sample=(temperature > 0.0))
        response = HF_PROCESSOR.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response.strip()

    def generate_content(self, model: str, prompt: str, system_instruction: str = None, temperature: float = 0.2) -> str:
        if self.backend == "ollama":
            try:
                return self._call_ollama(model, prompt, system_instruction, temperature)
            except Exception as e:
                logger.error(f"[LLM] Ollama call failed ({e}). Ensure Ollama is running and model '{os.getenv('OLLAMA_MODEL', 'gemma3:4b')}' is pulled.")
                raise e
        elif self.backend == "huggingface":
            return self._call_huggingface(prompt, system_instruction, temperature)
        else:
            return self._call_gemini(model, prompt, system_instruction, temperature)
