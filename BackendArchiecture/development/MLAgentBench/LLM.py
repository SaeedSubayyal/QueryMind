""" This file contains the code for calling Gemini API. """

import os
import google.generativeai as genai
import json
import time
from typing import Optional

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "models/gemini-2.0-flash"

def complete_text(prompt: str, model: str = MODEL, temperature: float = 0.7, max_tokens: int = 4000, log_file: Optional[str] = None, **kwargs) -> str:
    """Complete text using Gemini model."""
    print("\nCalling complete_text with parameters:")
    print(f"model: {model}")
    print(f"temperature: {temperature}")
    print(f"max_tokens: {max_tokens}")
    print(f"additional kwargs: {kwargs}")
    
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in complete_text: {e}")
        return ""

def complete_text_fast(prompt: str, model: str = MODEL, temperature: float = 0.7, max_tokens: int = 4000, log_file: Optional[str] = None, **kwargs) -> str:
    """Fast version of complete_text with retries."""
    max_retries = 3
    for i in range(max_retries):
        try:
            return complete_text(prompt, model, temperature, max_tokens, log_file, **kwargs)
        except Exception as e:
            if i == max_retries - 1:
                print(f"Failed after {max_retries} retries: {e}")
                return ""
            time.sleep(1)

# Constants for compatibility
HUMAN_PROMPT = "Human: "
AI_PROMPT = "Assistant: "
STATISTICAL_DIR = "./logs"
FAST_MODEL = MODEL

