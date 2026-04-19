from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram_peft.model import EngramModel

# 1. Setup paths
model_id = "{{MODEL_NAME}}"
# Automatically find the adapter directory where this script is located
adapter_path = Path(__file__).parent

print(f"[*] Loading model and adapters from {adapter_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Standard practice for Llama/TinyLlama models which don't have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Base Model and PEFT (LoRA) if exists
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# Check if there is a LoRA adapter (PeftModel) in the same directory
if (adapter_path / "adapter_config.json").exists():
    print("[*] Detected LoRA adapter. Loading PEFT model...")
    base_model = cast("Any", PeftModel.from_pretrained(base_model, str(adapter_path)))

# 3. Load Engram-augmented model
model = EngramModel.from_pretrained(base_model, str(adapter_path), tokenizer=tokenizer)
model.eval()

# 4. Quick test
prompt = "What is the secret of life?"

# Use Chat Template if available (recommended for Chat models)
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    input_ids = cast(
        "torch.Tensor",
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ),
    ).to(model.base_model.device)
    print(f"\n[*] Prompt: '{prompt}' (Applied Chat Template)")
else:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        model.base_model.device
    )
    print(f"\n[*] Prompt: '{prompt}'")

print("[*] Generating response...")

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,  # Test KV-Cache compatibility
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode only the NEWLY generated tokens
input_len = input_ids.shape[-1]
response_tokens = outputs[0][input_len:]

if len(response_tokens) == 0:
    print(
        f"\n[!] Warning: No tokens were generated. Output length: {len(outputs[0])}, Input length: {input_len}"
    )
    print(f"[*] Raw output IDs: {outputs[0].tolist()}")

response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

# Ensure response_text is a string for mypy
if isinstance(response_text, list):
    response_text = "".join(response_text)

print("\nResponse:")
print("-" * 20)
print(response_text.strip())
print("-" * 20)
