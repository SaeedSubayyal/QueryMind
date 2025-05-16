from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import psutil
import uvicorn
from typing import List, Optional

cache_dir = "C:/models"

app = FastAPI(title="Qwen 2.5 Coder API for Research & Data Science")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def print_memory_usage():
    memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory:.2f} MB")

print("Loading model and tokenizer... This may take a few minutes.")
print_memory_usage()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", cache_dir=cache_dir)

# Use 8-bit quantization for CPU
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B",
    cache_dir=cache_dir,
    device_map="cpu",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)

print("Model and tokenizer loaded successfully!")
print_memory_usage()

class GenerationConfig:
    MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    DEFAULT_DO_SAMPLE = True

    @staticmethod
    def limit_tokens(tokens: int) -> int:
        return min(tokens, GenerationConfig.MAX_NEW_TOKENS)

class GenerationRequest(BaseModel):
    prompt: str
    template: Optional[str] = "default"
    max_new_tokens: int = Field(default=512, le=GenerationConfig.MAX_NEW_TOKENS)
    temperature: float = GenerationConfig.DEFAULT_TEMPERATURE
    top_p: float = GenerationConfig.DEFAULT_TOP_P
    do_sample: bool = GenerationConfig.DEFAULT_DO_SAMPLE
    custom_template: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = None
    max_new_tokens: int = Field(default=512, le=GenerationConfig.MAX_NEW_TOKENS)
    temperature: float = GenerationConfig.DEFAULT_TEMPERATURE
    top_p: float = GenerationConfig.DEFAULT_TOP_P
    do_sample: bool = GenerationConfig.DEFAULT_DO_SAMPLE

@app.post("/chat")
async def chat(request: ChatRequest):
    formatted_prompt = ""
    if request.system_prompt:
        formatted_prompt += f"System: {request.system_prompt}\n"
    for msg in request.messages:
        if msg.role.lower() == "user":
            formatted_prompt += f"User: {msg.content}\n"
        elif msg.role.lower() == "assistant":
            formatted_prompt += f"Assistant: {msg.content}\n"
        elif msg.role.lower() == "system" and not request.system_prompt:
            formatted_prompt += f"System: {msg.content}\n"
    formatted_prompt += "Assistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GenerationConfig.limit_tokens(request.max_new_tokens),
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
            use_cache=True,
            num_beams=1
        )
    full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = full_result[len(formatted_prompt):].strip()
    print(assistant_response)
    return {
        "response": assistant_response,
        "full_text": full_result
    }

if __name__ == "__main__":
    uvicorn.run(
        "model:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        limit_concurrency=1,
        timeout_keep_alive=30
    )