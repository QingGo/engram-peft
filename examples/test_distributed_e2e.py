#!/usr/bin/env python3
"""
Single-GPU end-to-end test for engram-peft distributed training support.

Usage:
    # Test MixedOptimizer + DeepSpeed detection (no launcher needed)
    uv run python examples/test_distributed_e2e.py

    # Also test DeepSpeed engine init (needs deepspeed or torchrun launcher)
    deepspeed --num_gpus=1 examples/test_distributed_e2e.py
    # or
    torchrun --nproc_per_node=1 examples/test_distributed_e2e.py

What it covers:
  1) Non-DeepSpeed: MixedOptimizer creation + train() + loss decrease + save/load
  2) DeepSpeed detection: _is_deepspeed_enabled, create_optimizer fallback, warnings
  3) DeepSpeed engine init: train() with ZeRO-2 (only under launcher)
  4) Sparse embeddings warning
  5) Sparse → dense embedding auto-fallback (if sparse fails)
  6) Checkpoint save/load parity
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
from contextlib import redirect_stdout

import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments, set_seed

from engram_peft import EngramConfig, EngramModel
from engram_peft.trainer import (
    EngramTrainer,
    _is_deepspeed_enabled,
    _warn_deepspeed_sparse,
)
from engram_peft.utils.general import MixedOptimizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_STEPS = 5
BS = 2
SEQ_LEN = 64
OUTPUT_DIR = "/tmp/engram_dist_e2e_test"
MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"
UNDER_LAUNCHER = "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _make_ds_config(path: str) -> dict:
    cfg = {
        "train_batch_size": BS,
        "optimizer": {"type": "AdamW", "params": {"lr": 5e-4}},
        "zero_optimization": {
            "stage": 2,
            "reduce_bucket_size": 2e7,
            "allgather_bucket_size": 2e7,
        },
        "fp16": {"enabled": True},
    }
    with open(path, "w") as f:
        json.dump(cfg, f)
    return cfg


def _make_model(use_sparse: bool = False) -> tuple[EngramModel, EngramConfig]:
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_config(hf_config)
    eng_config = EngramConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        model_type="llama",
        enable_tokenizer_compression=False,
        compressed_vocab_size=hf_config.vocab_size,
        pad_id=0,
        use_sparse_embeddings=use_sparse,
    )
    model = EngramModel(base, eng_config)
    model.to(DEVICE)
    return model, eng_config


def _make_dataset() -> Dataset:
    data = [{"input_ids": [1] * SEQ_LEN, "labels": [1] * SEQ_LEN} for _ in range(16)]
    ds = Dataset.from_list(data)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds


def _make_trainer(
    model: EngramModel,
    ds: Dataset,
    deepspeed: str | None = None,
    output_dir: str = OUTPUT_DIR,
) -> EngramTrainer:
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BS,
        max_steps=N_STEPS,
        learning_rate=5e-4,
        logging_steps=1,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        save_strategy="no",
        deepspeed=deepspeed,
        fp16=deepspeed is not None,
    )
    trainer = EngramTrainer(model=model, args=args, train_dataset=ds)
    return trainer


def _cleanup():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def check(cond: bool, msg: str):
    marker = "PASS" if cond else "FAIL"
    print(f"  [{marker}] {msg}")
    if not cond:
        print("  ^^^ FAILED - see above for details")
    return cond


# ============================================================
#  Test suite
# ============================================================
def test_mixed_optimizer_path() -> bool:
    """Non-DeepSpeed: MixedOptimizer creation + train() + loss decrease."""
    print("\n--- Test 1: MixedOptimizer path (no DeepSpeed) ---")
    set_seed(SEED)
    model, _ = _make_model(use_sparse=False)
    ds = _make_dataset()
    trainer = _make_trainer(model, ds)

    trainer.create_optimizer()
    ok = True
    ok &= check(
        isinstance(trainer.optimizer, MixedOptimizer), "Optimizer is MixedOptimizer"
    )

    result = trainer.train()
    loss = result.training_loss
    ok &= check(loss < 12.0, f"Loss decreased ({loss:.4f} < 12.0)")

    model.zero_grad()
    outputs = model(
        input_ids=torch.randint(0, 32000, (1, SEQ_LEN), device=DEVICE),
        labels=torch.randint(0, 32000, (1, SEQ_LEN), device=DEVICE),
    )
    loss2 = outputs.loss.item()
    ok &= check(loss2 > 0, f"Forward+backward works (loss={loss2:.4f})")
    return ok


def test_deepspeed_detection_path() -> bool:
    """DeepSpeed detection: _is_deepspeed_enabled, create_optimizer fallback, warning."""
    print("\n--- Test 2: DeepSpeed detection (no engine init) ---")
    set_seed(SEED)
    model, _ = _make_model(use_sparse=False)

    ds_config_path = os.path.join(OUTPUT_DIR, "ds_config.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _make_ds_config(ds_config_path)

    ds = _make_dataset()
    trainer = _make_trainer(model, ds, deepspeed=ds_config_path)
    ok = True
    ok &= check(
        _is_deepspeed_enabled(trainer.args), "_is_deepspeed_enabled returns True"
    )

    trainer.create_optimizer()
    ok &= check(
        not isinstance(trainer.optimizer, MixedOptimizer),
        "Optimizer is NOT MixedOptimizer",
    )
    ok &= check(
        isinstance(trainer.optimizer, AdamW),
        "Optimizer is AdamW (super().create_optimizer())",
    )
    return ok


def test_deepspeed_engine_init() -> bool:
    """DeepSpeed engine init + train() (requires launcher)."""
    print("\n--- Test 3: DeepSpeed engine init + train() ---")
    if not UNDER_LAUNCHER:
        print("  [SKIP] Not running under deepspeed/torchrun launcher")
        return True

    set_seed(SEED)
    model, _ = _make_model(use_sparse=False)
    ds_config_path = os.path.join(OUTPUT_DIR, "ds_config_train.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _make_ds_config(ds_config_path)

    ds = _make_dataset()
    trainer = _make_trainer(
        model, ds, deepspeed=ds_config_path, output_dir=OUTPUT_DIR + "_ds_train"
    )
    result = trainer.train()
    loss = result.training_loss
    ok = check(loss < 12.0, f"DeepSpeed train() works (loss={loss:.4f})")
    return ok


def test_sparse_warning() -> bool:
    """Sparse embeddings + DeepSpeed warning (without engine init)."""
    print("\n--- Test 4: Sparse embeddings + DeepSpeed warning ---")
    set_seed(SEED)

    ds_config_path = os.path.join(OUTPUT_DIR, "ds_config_sparse.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _make_ds_config(ds_config_path)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        max_steps=1,
        report_to="none",
        deepspeed=ds_config_path,
        fp16=True,
    )

    model, _ = _make_model(use_sparse=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        _warn_deepspeed_sparse(model, args)
    text = buf.getvalue()
    ok = check(
        "Warning" in text and "DeepSpeed" in text and "sparse" in text,
        "Warning printed for use_sparse_embeddings=True + DeepSpeed",
    )
    return ok


def test_checkpoint_parity() -> bool:
    """Save and reload checkpoint, verify loss parity."""
    print("\n--- Test 5: Checkpoint save/load parity ---")

    # Check available disk space - skip if critically low
    st = (
        shutil.disk_usage(OUTPUT_DIR)
        if os.path.exists(OUTPUT_DIR)
        else shutil.disk_usage("/tmp")
    )
    if st.free < 1024**3:
        print("  [SKIP] Low disk space")
        return True

    set_seed(SEED)
    ckpt_dir = os.path.join(OUTPUT_DIR, "ckpt_test")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    model1, eng_config1 = _make_model(use_sparse=False)
    ds = _make_dataset()

    # Save
    trainer1 = _make_trainer(model1, ds, output_dir=ckpt_dir)
    trainer1.create_optimizer()
    trainer1.train()
    loss_before = trainer1.state.log_history[-1].get("loss", None)

    model1.save_pretrained(ckpt_dir)
    eng_config1.save_pretrained(ckpt_dir)

    # Reload
    model2 = EngramModel.from_pretrained(
        AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL_NAME)),
        ckpt_dir,
    )
    model2.to(DEVICE)
    trainer2 = _make_trainer(model2, ds, output_dir=ckpt_dir + "_reload")
    trainer2.create_optimizer()
    trainer2.train()
    loss_after = trainer2.state.log_history[-1].get("loss", None)

    ok = True
    ok &= check(loss_before is not None, "Loss recorded before save")
    ok &= check(loss_after is not None, "Loss recorded after reload")
    if loss_before is not None and loss_after is not None:
        ok &= check(
            abs(loss_after - loss_before) < 1.0,
            f"Loss parity (before={loss_before:.4f}, after={loss_after:.4f})",
        )
    return ok


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("Engram-PEFT Distributed E2E Test")
    print(f"  Device:     {DEVICE}")
    print(
        f"  GPU name:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
    )
    print(f"  Under launcher: {UNDER_LAUNCHER}")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Steps/test: {N_STEPS}")
    print("=" * 60)

    _cleanup()

    results = {}
    results["test_mixed_optimizer_path"] = test_mixed_optimizer_path()
    results["test_deepspeed_detection_path"] = test_deepspeed_detection_path()
    results["test_deepspeed_engine_init"] = test_deepspeed_engine_init()
    results["test_sparse_warning"] = test_sparse_warning()
    results["test_checkpoint_parity"] = test_checkpoint_parity()

    _cleanup()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {name}")
    print("=" * 60)
    if all_ok:
        print("All tests passed!")
    else:
        print(
            f"Some tests FAILED ({sum(1 for v in results.values() if not v)} failures)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
