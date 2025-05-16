from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import uvicorn
import os
import json
from typing import List, Optional, Dict, Any, Union

# Define the cache directory path
cache_dir = "C:/models"

# Create FastAPI app
app = FastAPI(title="Qwen 2.5 Coder API for Research & Data Science")

# Add CORS middleware to allow requests from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and tokenizer once when the app starts
print("Loading model and tokenizer... This may take a few minutes.")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B",
    cache_dir=cache_dir,
    device_map="cpu",  # Force CPU usage
    torch_dtype=torch.float32,  # Use standard precision for CPU
    low_cpu_mem_usage=True
)
print("Model and tokenizer loaded successfully!")

# Store optimized prompts for different use cases
PROMPT_TEMPLATES = {
    "default": "{input}",
    
    "data_science": """As an expert data scientist, analyze the following problem and provide a comprehensive solution with code:

Problem: {input}

Please provide:
1. Analysis of the problem
2. Complete Python code solution using best practices
3. Explanation of algorithms/techniques used
4. How to interpret the results
5. Potential improvements or limitations

Solution:
""",
    
    "code_improvement": """Analyze and improve the following code for performance, readability, and best practices:

```python
{input}
```

Provide a detailed response including:
1. Identified issues and potential optimizations
2. Refactored code with improvements
3. Explanation of changes made
4. Big O analysis before and after
5. Additional suggestions for further improvement

Improved solution:
""",
    
    "data_cleaning": """As a data preprocessing expert, review this data preparation approach and provide the most efficient solution:

{input}

Your comprehensive solution should include:
1. Data validation and quality checks
2. Cleaning strategies for missing values, outliers, and inconsistencies
3. Feature engineering recommendations
4. Complete code implementation with pandas/numpy
5. Validation approach to verify the cleaning process

Solution:
""",

    "ml_model": """Design a machine learning solution for the following problem:

{input}

Provide a detailed implementation including:
1. Problem framing and approach selection
2. Data preparation steps
3. Model architecture with justification
4. Complete implementation code
5. Evaluation strategy and metrics
6. Fine-tuning approach
7. Production considerations

Implementation:
""",

    "data_visualization": """Create comprehensive data visualization code for the following scenario:

{input}

Provide a complete solution including:
1. Visualization strategy and choice of plots
2. Complete implementation code (matplotlib/seaborn/plotly)
3. Customization for clarity and insight extraction
4. Interpretation guidelines
5. Alternative visualization approaches

Code solution:
""",

    "statistical_analysis": """Perform a statistical analysis for the following problem:

{input}

Your response should include:
1. Statistical approach justification
2. Hypothesis formulation (if applicable)
3. Complete implementation code
4. Interpretation of results
5. Assumptions and limitations
6. Recommendations based on findings

Analysis:
"""
}

class GenerationRequest(BaseModel):
    prompt: str
    template: Optional[str] = "default"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    custom_template: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = None
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class DataScienceRequest(BaseModel):
    problem_description: str
    data_description: Optional[str] = None
    specific_requirements: Optional[str] = None
    techniques_to_use: Optional[List[str]] = None
    template: str = "data_science"
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@app.post("/generate")
async def generate(request: GenerationRequest):
    # Select template or use custom
    template = request.custom_template if request.custom_template else PROMPT_TEMPLATES.get(request.template, PROMPT_TEMPLATES["default"])
    
    # Format the prompt with the template
    formatted_prompt = template.format(input=request.prompt)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
    
    # Decode and return the result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # For templated prompts, remove the template prefix from result if present
    if request.template != "default" and request.template in PROMPT_TEMPLATES:
        prefix = template.format(input=request.prompt)
        if result.startswith(prefix):
            result = result[len(prefix):]
    
    return {"result": result}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Format messages for the model with optional system prompt
    formatted_prompt = ""
    
    # Add system prompt if provided
    if request.system_prompt:
        formatted_prompt += f"System: {request.system_prompt}\n"
    
    # Add message history
    for msg in request.messages:
        if msg.role.lower() == "user":
            formatted_prompt += f"User: {msg.content}\n"
        elif msg.role.lower() == "assistant":
            formatted_prompt += f"Assistant: {msg.content}\n"
        elif msg.role.lower() == "system" and not request.system_prompt:
            # Only use inline system messages if no separate system prompt was provided
            formatted_prompt += f"System: {msg.content}\n"
    
    formatted_prompt += "Assistant:"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
    
    # Decode the result
    full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (after the last prompt)
    assistant_response = full_result[len(formatted_prompt):].strip()
    
    return {
        "response": assistant_response,
        "full_text": full_result
    }

@app.post("/data-science")
async def data_science(request: DataScienceRequest):
    # Build a comprehensive prompt for data science tasks
    problem_content = request.problem_description
    
    # Add data description if provided
    if request.data_description:
        problem_content += f"\n\nData description: {request.data_description}"
    
    # Add specific requirements if provided
    if request.specific_requirements:
        problem_content += f"\n\nRequirements: {request.specific_requirements}"
    
    # Add techniques to use if provided
    if request.techniques_to_use:
        techniques_str = ", ".join(request.techniques_to_use)
        problem_content += f"\n\nPlease incorporate these techniques: {techniques_str}"
    
    # Use the template to format the prompt
    template = PROMPT_TEMPLATES.get(request.template, PROMPT_TEMPLATES["data_science"])
    formatted_prompt = template.format(input=problem_content)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
    
    # Decode the result
    full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the result
    if full_result.startswith(formatted_prompt):
        result = full_result[len(formatted_prompt):]
    else:
        result = full_result
    
    return {
        "result": result.strip(),
        "full_text": full_result
    }

# Specialized endpoints for different data science tasks
@app.post("/ml-model")
async def ml_model(
    problem: str,
    data_description: Optional[str] = None,
    requirements: Optional[str] = None,
    max_new_tokens: int = 4096
):
    request = DataScienceRequest(
        problem_description=problem,
        data_description=data_description,
        specific_requirements=requirements,
        template="ml_model",
        max_new_tokens=max_new_tokens
    )
    return await data_science(request)

@app.post("/data-visualization")
async def data_visualization(
    problem: str,
    data_description: Optional[str] = None,
    requirements: Optional[str] = None,
    max_new_tokens: int = 4096
):
    request = DataScienceRequest(
        problem_description=problem,
        data_description=data_description,
        specific_requirements=requirements,
        template="data_visualization",
        max_new_tokens=max_new_tokens
    )
    return await data_science(request)

@app.post("/data-cleaning")
async def data_cleaning(
    problem: str,
    data_description: Optional[str] = None,
    requirements: Optional[str] = None,
    max_new_tokens: int = 4096
):
    request = DataScienceRequest(
        problem_description=problem,
        data_description=data_description,
        specific_requirements=requirements,
        template="data_cleaning",
        max_new_tokens=max_new_tokens
    )
    return await data_science(request)

@app.post("/statistical-analysis")
async def statistical_analysis(
    problem: str,
    data_description: Optional[str] = None,
    requirements: Optional[str] = None,
    max_new_tokens: int = 4096
):
    request = DataScienceRequest(
        problem_description=problem,
        data_description=data_description,
        specific_requirements=requirements,
        template="statistical_analysis",
        max_new_tokens=max_new_tokens
    )
    return await data_science(request)

@app.post("/code-improvement")
async def code_improvement(
    code: str,
    requirements: Optional[str] = None,
    max_new_tokens: int = 4096
):
    problem = code
    if requirements:
        problem += f"\n\nAdditional requirements: {requirements}"
    
    request = DataScienceRequest(
        problem_description=problem,
        template="code_improvement",
        max_new_tokens=max_new_tokens
    )
    return await data_science(request)

# Add template listing endpoint
@app.get("/templates")
async def list_templates():
    return {
        "available_templates": list(PROMPT_TEMPLATES.keys()),
        "descriptions": {
            "default": "Basic prompt with no special formatting",
            "data_science": "General data science problem-solving",
            "code_improvement": "Analyze and optimize existing code",
            "data_cleaning": "Data preprocessing and cleaning tasks",
            "ml_model": "Machine learning model design and implementation",
            "data_visualization": "Creating effective data visualizations",
            "statistical_analysis": "Statistical testing and analysis"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "model": "Qwen/Qwen2.5-Coder-1.5B",
        "max_supported_tokens": 8192,
        "available_endpoints": [
            "/generate", 
            "/chat", 
            "/data-science", 
            "/ml-model", 
            "/data-visualization", 
            "/data-cleaning", 
            "/statistical-analysis", 
            "/code-improvement",
            "/templates",
            "/health"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("model:app", host="0.0.0.0", port=8000, reload=False)
