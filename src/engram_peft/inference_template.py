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
    # apply_chat_template can return BatchEncoding or Tensor depending on version
    gen_inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    print(f"\n[*] Prompt: '{prompt}' (Applied Chat Template)")
else:
    gen_inputs = tokenizer(prompt, return_tensors="pt")
    print(f"\n[*] Prompt: '{prompt}'")

# Robustly extract input_ids if the tokenizer returned a BatchEncoding/dict
if isinstance(gen_inputs, dict) or hasattr(gen_inputs, "input_ids"):
    input_ids = (
        gen_inputs["input_ids"]
        if isinstance(gen_inputs, dict)
        else gen_inputs.input_ids
    )
else:
    input_ids = gen_inputs

# Move to correct device
input_ids = input_ids.to(model.base_model.device)
# Capture other inputs (like attention_mask) if they exist
gen_kwargs: dict[str, Any] = (
    {
        k: v.to(model.base_model.device)
        for k, v in gen_inputs.items()
        if k != "input_ids" and torch.is_tensor(v)
    }
    if isinstance(gen_inputs, dict)
    else {}
)

print("[*] Generating response...")

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        **gen_kwargs,
        max_new_tokens=50,
        max_length=None,  # Suppress warning when max_new_tokens is set
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
