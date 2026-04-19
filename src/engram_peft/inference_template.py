import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram_peft.model import EngramModel

# 1. Setup paths
model_id = "{{MODEL_NAME}}"
adapter_path = "."  # Current directory where this script is located

print(f"[*] Loading model and Engram adapter from {adapter_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# 2. Load Engram-augmented model
model = EngramModel.from_pretrained(base_model, adapter_path, tokenizer=tokenizer)
model.eval()

# 3. Quick test
prompt = "What is the secret of life?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)

print(f"[*] Generating response for: '{prompt}'")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print("\nResponse:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
