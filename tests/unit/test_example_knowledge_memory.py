"""Unit tests for examples/engram_knowledge_memory.py logic.

Tests pure functions by duplicating minimal logic (no module import needed).
For heavy integration, uses mock-based approach via function-level imports.

All tests run in <1s on CPU and require no GPU.
"""

from __future__ import annotations

import argparse
import importlib.util as _util
import json
import logging
import os
import re
import sys
import tempfile
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# ──────────────────────────────────────────────────────────────────────
#  Pure: normalize_answer
# ──────────────────────────────────────────────────────────────────────


def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class TestNormalizeAnswer:
    def test_lowercase(self) -> None:
        assert _normalize_answer("Hello World") == "hello world"

    def test_strip_punctuation(self) -> None:
        assert _normalize_answer("Paris!") == "paris"
        assert _normalize_answer("What's up?") == "whats up"

    def test_collapse_whitespace(self) -> None:
        assert _normalize_answer("  hello   world  ") == "hello world"

    def test_empty_string(self) -> None:
        assert _normalize_answer("") == ""

    def test_numbers(self) -> None:
        assert _normalize_answer("42") == "42"

    def test_mixed(self) -> None:
        assert _normalize_answer("  The Eiffel Tower!!  ") == "the eiffel tower"


# ──────────────────────────────────────────────────────────────────────
#  Pure: format_qa_text
# ──────────────────────────────────────────────────────────────────────


def _format_qa_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


class TestFormatQAText:
    def test_basic(self) -> None:
        assert (
            _format_qa_text("Where is Paris?", "France")
            == "Question: Where is Paris?\nAnswer: France"
        )

    def test_empty_answer_trailing_space(self) -> None:
        """f-string ``f'Answer: {answer}'`` with empty answer yields trailing space."""
        result = _format_qa_text("Where?", "")
        assert result == "Question: Where?\nAnswer: "


# ──────────────────────────────────────────────────────────────────────
#  Pure: print_comparison
# ──────────────────────────────────────────────────────────────────────


def _print_comparison(results: dict[str, dict[str, Any]]) -> None:
    if not results:
        return
    base_acc = results.get("Base", {}).get("accuracy", 0.0)
    logging.info("=" * 60)
    logging.info("  PopQA Benchmark Results")
    logging.info("=" * 60)
    logging.info("  %-25s %10s %10s", "Config", "Accuracy", "Δ vs Base")
    logging.info("  " + "-" * 47)
    for name, res in results.items():
        acc = res.get("accuracy", 0.0)
        if name == "Base":
            delta_str = "—"
        else:
            delta = acc - base_acc
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        logging.info("  %-25s %8.1f%% %10s", name, acc, delta_str)
    logging.info("=" * 60)


class TestPrintComparison:
    def test_empty(self, caplog: Any) -> None:
        _print_comparison({})
        assert len(caplog.records) == 0

    def test_single_result(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO)
        _print_comparison({"Base": {"correct": 80, "total": 100, "accuracy": 80.0}})
        assert any("80.0%" in rec.message for rec in caplog.records)

    def test_delta_shown(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO)
        _print_comparison(
            {
                "Base": {"correct": 80, "total": 100, "accuracy": 80.0},
                "+Engram": {"correct": 90, "total": 100, "accuracy": 90.0},
            }
        )
        combined = "\n".join(rec.message for rec in caplog.records)
        assert "+10.0%" in combined or "10.0%" in combined

    def test_base_delta_dash(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO)
        _print_comparison({"Base": {"correct": 50, "total": 100, "accuracy": 50.0}})
        combined = "\n".join(rec.message for rec in caplog.records)
        assert "—" in combined


# ──────────────────────────────────────────────────────────────────────
#  parse_args
# ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="test")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--eval_max_samples", type=int, default=200)
    p.add_argument("--use_deepspeed", action="store_true")
    p.add_argument("--output_dir", default="outputs/popqa_benchmark")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engram", action="store_true", default=True)
    p.add_argument("--no-engram", action="store_false", dest="engram")
    p.add_argument("--lora", action="store_true", default=False)
    p.add_argument("--joint", action="store_true", default=False)
    p.add_argument("--engram_path", type=str, default=None)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--embedding_dim", type=int, default=1280)
    p.add_argument("--target_layers", type=int, nargs="+", default=[2, 15])
    p.add_argument("--entropy_loss_weight", type=float, default=0.01)
    return p.parse_args(argv)


class TestParseArgs:
    def test_defaults(self) -> None:
        args = _parse_args([])
        assert args.mode == "train"
        assert args.engram is True
        assert args.lora is False
        assert args.joint is False
        assert args.max_samples is None
        assert args.eval_max_samples == 200

    def test_lora(self) -> None:
        args = _parse_args(["--lora"])
        assert args.lora is True

    def test_joint(self) -> None:
        args = _parse_args(["--joint"])
        assert args.joint is True

    def test_no_engram(self) -> None:
        args = _parse_args(["--no-engram"])
        assert args.engram is False

    def test_eval_mode(self) -> None:
        args = _parse_args(["--mode", "eval", "--engram_path", "/p"])
        assert args.mode == "eval"
        assert args.engram_path == "/p"

    def test_lora_params(self) -> None:
        args = _parse_args(["--lora_r", "32", "--lora_alpha", "64"])
        assert args.lora_r == 32
        assert args.lora_alpha == 64

    def test_target_layers(self) -> None:
        args = _parse_args(["--target_layers", "0", "1", "2"])
        assert args.target_layers == [0, 1, 2]

    def test_deepspeed_flag(self) -> None:
        args = _parse_args(["--use_deepspeed"])
        assert args.use_deepspeed is True


# ──────────────────────────────────────────────────────────────────────
#  write_default_ds_config
# ──────────────────────────────────────────────────────────────────────


def _write_default_ds_config(output_dir: str) -> str:
    config = {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "gradient_clipping": 1.0,
        "bf16": {"enabled": "auto"},
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ds_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


class TestWriteDefaultDSConfig:
    def test_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_default_ds_config(tmp)
            with open(path) as f:
                cfg = json.load(f)
            assert cfg["zero_optimization"]["stage"] == 2
            assert cfg["bf16"]["enabled"] == "auto"

    def test_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_default_ds_config(tmp)
            assert os.path.isfile(path)
            assert path.endswith("ds_config.json")


# ──────────────────────────────────────────────────────────────────────
#  load_popqa — import example module once
# ──────────────────────────────────────────────────────────────────────

# Cache the example module so import cost is paid once

_EXAMPLE_MODULE: Any = None
_IMPORTED = False


def _get_example() -> Any:
    global _EXAMPLE_MODULE, _IMPORTED
    if not _IMPORTED:
        _path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "examples",
            "engram_knowledge_memory.py",
        )
        _spec = _util.spec_from_file_location(
            "engram_knowledge_memory", os.path.abspath(_path)
        )
        assert _spec and _spec.loader
        _EXAMPLE_MODULE = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_EXAMPLE_MODULE)
        _IMPORTED = True
    return _EXAMPLE_MODULE


# Warm the import cache so per-test overhead is <1s
_get_example()


class TestLoadPopQA:
    def _patch_load_dataset(self) -> Any:
        """Helper: mock datasets.load_dataset within the example module."""
        return patch.object(_get_example(), "load_dataset")

    def test_80_20_split(self) -> None:
        from datasets import Dataset

        mock_ds = Dataset.from_dict(
            {
                "question": [f"Q{i}" for i in range(100)],
                "possible_answers": [[f"A{i}"] for i in range(100)],
            }
        )
        with self._patch_load_dataset() as mock_load:
            mock_load.return_value = mock_ds
            train, test = _get_example().load_popqa()
        assert len(train) == 80
        assert len(test) == 20

    def test_max_samples(self) -> None:
        from datasets import Dataset

        mock_ds = Dataset.from_dict(
            {
                "question": [f"Q{i}" for i in range(100)],
                "possible_answers": [[f"A{i}"] for i in range(100)],
            }
        )
        with self._patch_load_dataset() as mock_load:
            mock_load.return_value = mock_ds
            train, test = _get_example().load_popqa(max_samples=10)
        assert len(train) + len(test) == 10

    def test_text_column(self) -> None:
        from datasets import Dataset

        mock_ds = Dataset.from_dict(
            {
                "question": [f"Q{i}" for i in range(10)],
                "possible_answers": [[f"Answer{i}", f"Alt{i}"] for i in range(10)],
            }
        )
        with self._patch_load_dataset() as mock_load:
            mock_load.return_value = mock_ds
            train, _ = _get_example().load_popqa(max_samples=10)
        assert "Question: Q0" in train[0]["text"]
        assert "Answer0" in train[0]["text"]


# ──────────────────────────────────────────────────────────────────────
#  evaluate_em — mocked model, tokenizer, dataset
# ──────────────────────────────────────────────────────────────────────


class TestEvaluateEM:
    def _module(self) -> Any:
        return _get_example()

    def _make_dataset(self) -> Any:
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "question": ["Q0", "Q1"],
                "possible_answers": [["ans_zero", "alt_zero"], ["ans_one", "alt_one"]],
            }
        )

    def _make_mock_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.padding_side = "right"
        tok.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        tok.decode.return_value = "some output"
        return tok

    def test_accuracy_dict_structure(self) -> None:
        fn = self._module().evaluate_em
        ds = self._make_dataset()
        model = MagicMock()
        model.device = torch.device("cpu")
        model.eval.return_value = None
        model.generate.return_value = torch.tensor([[1]])
        tok = self._make_mock_tokenizer()
        result = fn(model, tok, ds, max_samples=2)
        assert isinstance(result, dict)
        assert "correct" in result
        assert "total" in result
        assert "accuracy" in result
        assert result["total"] == 2

    def test_padding_side_restored(self) -> None:
        fn = self._module().evaluate_em
        ds = self._make_dataset()
        model = MagicMock()
        model.device = torch.device("cpu")
        model.eval.return_value = None
        model.generate.return_value = torch.tensor([[1]])
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.padding_side = "right"
        tok.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        tok.decode.return_value = "output"
        fn(model, tok, ds, max_samples=1)
        assert tok.padding_side == "right"

    def test_logging_progress(self, caplog: Any) -> None:
        fn = self._module().evaluate_em
        from datasets import Dataset

        size = 50
        ds = Dataset.from_dict(
            {
                "question": [f"Q{i}" for i in range(size)],
                "possible_answers": [[f"ans_{i}"] for i in range(size)],
            }
        )
        model = MagicMock()
        model.device = torch.device("cpu")
        model.eval.return_value = None
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.padding_side = "right"
        tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        tok.decode.return_value = "output"
        caplog.set_level(logging.INFO)
        fn(model, tok, ds, max_samples=size)
        assert any("progress" in r.message.lower() for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────
#  build_engram_model — heavy mocking
# ──────────────────────────────────────────────────────────────────────


class TestBuildEngramModel:
    def _module(self) -> Any:
        return _get_example()

    def test_basic_flow(self) -> None:
        mod = self._module()
        from engram_peft import EngramConfig

        with (
            patch.object(mod, "load_4bit_backbone"),
            patch.object(mod, "prepare_model_for_kbit_training"),
            patch.object(mod, "get_engram_model") as mock_get,
            patch.object(mod, "wash_tokenizer"),
        ):
            mock_model = MagicMock()
            mock_get.return_value = mock_model
            config = EngramConfig(target_layers=[0], embedding_dim=16)
            result = mod.build_engram_model(
                "dummy", MagicMock(), use_deepspeed=False, engram_config=config
            )
            assert result == mock_model
            mock_model.print_trainable_parameters.assert_called_once()

    def test_skips_prepare_when_deepspeed(self) -> None:
        mod = self._module()
        from engram_peft import EngramConfig

        with (
            patch.object(mod, "load_4bit_backbone"),
            patch.object(mod, "get_engram_model"),
            patch.object(mod, "wash_tokenizer"),
            patch.object(mod, "prepare_model_for_kbit_training") as mock_prep,
        ):
            config = EngramConfig(target_layers=[0], embedding_dim=16)
            mod.build_engram_model(
                "dummy", MagicMock(), use_deepspeed=True, engram_config=config
            )
            mock_prep.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
#  build_lora_model — heavy mocking
# ──────────────────────────────────────────────────────────────────────


class TestBuildLoRAModel:
    def _module(self) -> Any:
        return _get_example()

    def test_basic_flow(self) -> None:
        mod = self._module()
        with (
            patch.object(mod, "load_4bit_backbone"),
            patch.object(mod, "prepare_model_for_kbit_training"),
            patch.object(mod, "get_peft_model") as mock_get,
        ):
            mock_peft = MagicMock()
            mock_get.return_value = mock_peft
            result = mod.build_lora_model("dummy", r=8, lora_alpha=16)
            assert result == mock_peft
            mock_peft.print_trainable_parameters.assert_called_once()

    def test_default_r_alpha(self) -> None:
        mod = self._module()
        with (
            patch.object(mod, "load_4bit_backbone"),
            patch.object(mod, "prepare_model_for_kbit_training"),
            patch.object(mod, "get_peft_model") as mock_get,
        ):
            mock_get.return_value = MagicMock()
            mod.build_lora_model("dummy")
            lora_config = mock_get.call_args[0][1]
            assert lora_config.r == 16
            assert lora_config.lora_alpha == 32


# ──────────────────────────────────────────────────────────────────────
#  load_4bit_backbone — mocked AutoModel
# ──────────────────────────────────────────────────────────────────────


class TestLoad4BitBackbone:
    def _module(self) -> Any:
        return _get_example()

    def test_calls_from_pretrained(self) -> None:
        mod = self._module()
        with (
            patch.object(mod, "AutoModelForCausalLM") as mock_auto,
            patch.object(mod, "BitsAndBytesConfig"),
        ):
            mock_auto.from_pretrained.return_value = MagicMock()
            result = mod.load_4bit_backbone("dummy/model")
            mock_auto.from_pretrained.assert_called_once()
            assert result is not None

    def test_deepspeed_device_map(self) -> None:
        mod = self._module()
        with (
            patch.object(mod, "AutoModelForCausalLM") as mock_auto,
            patch.object(mod, "BitsAndBytesConfig"),
        ):
            mock_auto.from_pretrained.return_value = MagicMock()
            mod.load_4bit_backbone("dummy/model", use_deepspeed=True)
            kwargs = mock_auto.from_pretrained.call_args.kwargs
            assert kwargs["device_map"] is None


# ──────────────────────────────────────────────────────────────────────
#  main() — fully mocked smoke test
# ──────────────────────────────────────────────────────────────────────


class TestMain:
    def _module(self) -> Any:
        return _get_example()

    def test_eval_mode_base_only(self) -> None:
        mod = self._module()
        from datasets import Dataset

        ds = Dataset.from_dict(
            {"text": ["Q\nA"], "question": ["Q"], "possible_answers": [["A"]]}
        )

        with (
            patch.object(mod, "load_popqa", return_value=(ds, ds)),
            patch.object(mod, "AutoTokenizer") as mock_tok_cls,
            patch.object(mod, "set_seed"),
            patch.object(mod, "logging"),
            patch.object(mod, "load_4bit_backbone") as mock_load,
        ):
            mock_tok = MagicMock()
            mock_tok.pad_token = None
            mock_tok.eos_token = 2
            mock_tok.pad_token_id = 0
            mock_tok.eos_token_id = 2
            mock_tok.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tok.decode.return_value = "answer"
            mock_tok_cls.from_pretrained.return_value = mock_tok

            mock_base = MagicMock()
            mock_base.device = torch.device("cpu")
            mock_base.eval.return_value = None
            mock_base.generate.return_value = torch.tensor([[1]])
            mock_load.return_value = mock_base

            with patch.object(
                sys, "argv", ["prog", "--mode", "eval", "--eval_max_samples", "1"]
            ):
                mod.main()

        mock_load.assert_called()

    def test_train_mode_smoke(self) -> None:
        mod = self._module()
        from datasets import Dataset

        from engram_peft import EngramConfig

        ds = Dataset.from_dict(
            {"text": ["Q\nA"], "question": ["Q"], "possible_answers": [["A"]]}
        )

        with (
            patch.object(mod, "load_popqa", return_value=(ds, ds)),
            patch.object(mod, "AutoTokenizer") as mock_tok_cls,
            patch.object(mod, "set_seed"),
            patch.object(mod, "logging"),
            patch.object(mod, "load_4bit_backbone"),
            patch.object(mod, "get_engram_model") as mock_get,
            patch.object(mod, "EngramTrainer") as mock_tr,
            patch.object(mod, "tokenize_dataset") as mock_tokenize,
            patch("os.makedirs"),
            patch.object(mod, "EngramModel"),
            patch.object(mod, "evaluate_em") as mock_eval,
        ):
            mock_tok = MagicMock()
            mock_tok.pad_token = None
            mock_tok.eos_token = 2
            mock_tok.pad_token_id = 0
            mock_tok.eos_token_id = 2
            mock_tok.return_value = {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
            }
            mock_tok.decode.return_value = "answer"
            mock_tok_cls.from_pretrained.return_value = mock_tok

            mock_model = MagicMock()
            mock_model.print_trainable_parameters = MagicMock()
            mock_model.config = EngramConfig(
                embedding_dim=32,
                target_layers=[0],
                compressed_vocab_size=1000,
                pad_id=0,
            )
            mock_get.return_value = mock_model

            mock_tr.return_value.train.return_value = None
            mock_tokenize.return_value = ds
            mock_eval.return_value = {"correct": 50, "total": 100, "accuracy": 50.0}

            with patch.object(
                sys, "argv", ["prog", "--max_samples", "10", "--max_steps", "1"]
            ):
                mod.main()

        mock_tr.assert_called_once()
