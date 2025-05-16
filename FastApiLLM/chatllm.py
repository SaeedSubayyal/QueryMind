from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import psutil

cache_dir = "C:/models"  # Update this path if needed

def print_memory_usage():
    memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory:.2f} MB")

print("Loading base Qwen2.5-Coder-1.5B model... Please wait.")
print_memory_usage()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", cache_dir=cache_dir)

# Load the base (non-quantized) model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B",
    cache_dir=cache_dir,
    device_map="auto",  # Uses GPU if available
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("Model and tokenizer loaded successfully.")
print_memory_usage()

def format_prompt(user_input):
    # Simple prompt template for base model
    return f"### Instruction:\n{user_input}\n### Response:\n"

def chat():
    print("\nStart chatting with the base model (type 'exit' to quit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        prompt = format_prompt(user_input)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip prompt from output
        reply = response[len(prompt):].strip()
        print(f"Assistant: {reply}")

chat()
