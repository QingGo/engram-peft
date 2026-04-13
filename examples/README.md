# Engram-PEFT Examples

This directory contains examples demonstrating how to use the Engram-PEFT library for efficient Parameter-Efficient Fine-Tuning.

## End-to-End Example: `end_to_end.py`

The `end_to_end.py` script provides a complete walkthrough of the Engram lifecycle, from dataset preparation to dynamic weight switching.

### Features
- **Official Config**: Uses `EngramConfig` aligned with the official paper specifications.
- **Engram Injection**: Demonstrates `get_engram_model` for non-invasive layer injection into a base Transformer model.
- **Optimized Training**: Uses the custom `MixedOptimizer` (SparseAdam + Adam) and `LambdaLR` scheduler.
- **Hugging Face Integration**: Fully compatible with the `Trainer` API.
- **Dynamic Loading**: Shows how to `load_engram` and `unload_engram` at runtime.

### Prerequisites
Ensure you have the dependencies installed. This project uses `uv` for package management.

### Running the Example
To run the full end-to-end example (training + inference):

```bash
uv run python examples/end_to_end.py
```

### Script Breakdown
1. **Phase 1: Training**
   - Loads a subset of the `TinyStories` dataset.
   - Loads `TinyLlama-1.1B` as the base model.
   - Injects Engram layers and freezes the base model.
   - Trains for 1 epoch.
   - Saves only the Engram weights (`engram_weights.pt`) and config.

2. **Phase 2: Inference**
   - Loads a clean base model.
   - Attaches the trained Engram weights.
   - Generates text.

3. **Phase 3: Dynamic Switching**
   - Demonstrates unloading Engram layers to revert to base model behavior.
   - Demonstrates reloading the weights dynamically.

### Memory Usage
For `TinyLlama-1.1B`, the training typically requires less than 8GB of VRAM with the provided settings (batch size 4, gradient accumulation 2), making it suitable for consumer GPUs like the RTX 3060/3070 and above.
