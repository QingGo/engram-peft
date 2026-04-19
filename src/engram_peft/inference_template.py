import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram_peft.model import EngramModel

# 1. Setup paths
model_id = "{{MODEL_NAME}}"
adapter_path = "."  # Current directory where this script is located

print(f"[*] Loading model and Engram adapter from {adapter_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Standard practice for Llama/TinyLlama models which don't have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# 2. Load Engram-augmented model
model = EngramModel.from_pretrained(base_model, adapter_path, tokenizer=tokenizer)
model.eval()

# 3. Quick test
prompt = "What is the secret of life?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)

print(f"\n[*] Prompt: '{prompt}'")
print("[*] Generating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode only the NEWLY generated tokens
input_len = inputs.input_ids.shape[-1]
response_tokens = outputs[0][input_len:]
response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

# Ensure response_text is a string for mypy
if isinstance(response_text, list):
    response_text = "".join(response_text)

print("\nResponse:")
print("-" * 20)
print(response_text.strip())
print("-" * 20)
