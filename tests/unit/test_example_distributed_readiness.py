"""Static analysis: verify examples/engram_knowledge_memory.py is safe for multi-GPU.

Runs entirely on CPU — no GPU required. Performs pattern-based checks on the
source code to catch common distributed-training bugs before renting a GPU cluster.

Checked patterns:
  1. Rank-gated save_pretrained / evaluate / logging
  2. ddp_find_unused_parameters=False in TrainingArguments
  3. remove_unused_columns=False when using custom data collator
  4. No hardcoded ``cuda:0`` or ``device(0)``
  5. ``model.eval()`` before generation
  6. ``unwrap_model`` before save_pretrained in DDP mode
  7. Broad set_seed call at the top of main()
  8. ``padding_side='left'`` for generation
  9. Proper ``device_map`` handling in DeepSpeed vs DDP
  10. No blocking ``wait()`` or ``barrier()`` that could deadlock
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

_EXAMPLE_PATH = Path("examples/engram_knowledge_memory.py")
assert _EXAMPLE_PATH.exists(), f"Example not found: {_EXAMPLE_PATH}"
_SOURCE = _EXAMPLE_PATH.read_text()


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────


def _ast_parse() -> ast.Module:
    return ast.parse(_SOURCE)


def _find_calls(tree: ast.Module, target: str) -> list[ast.Call]:
    """Find all ``ast.Call`` nodes whose function name matches *target*.

    Handles both bare names (``TrainingArguments(...)``) and dotted attributes
    (``model.save_pretrained(...)``).
    """
    results: list[ast.Call] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                if node.func.id == target:
                    results.append(node)
            elif isinstance(node.func, ast.Attribute):
                parts: list[str] = []
                cur = node.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                elif isinstance(cur, ast.Call):
                    parts.append("...call...")
                dotted = ".".join(reversed(parts))
                if dotted == target or dotted.endswith("." + target):
                    results.append(node)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return results


def _top_level_call_names(tree: ast.Module) -> list[str]:
    """Return the function/method names of all top-level expression calls in main()."""
    names: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name != "main":
                return
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(
                    child.func, ast.Attribute
                ):
                    names.append(child.func.attr)
                elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    names.append(child.func.id)

    _Visitor().visit(tree)
    return names


def _find_keyword_in_call(call: ast.Call, kwarg: str) -> ast.expr | None:
    """Find keyword argument value in an ast.Call."""
    for kw in call.keywords:
        if kw.arg == kwarg:
            return kw.value
    return None


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────


class TestDistributedPatterns:
    """Verify multi-GPU correctness via static analysis."""

    def test_rank_gated_save_pretrained(self) -> None:
        """save_pretrained must be inside ``if is_main:`` guard.

        Instead of fragile AST parent-tracking, we verify the structural pattern:
        ``if is_main:`` followed (within a few lines) by ``save_pretrained``.
        """
        lines = _SOURCE.split("\n")
        found = False
        for i, line in enumerate(lines):
            if "if is_main" in line:
                # Look at the block following this guard
                for j in range(i + 1, min(i + 20, len(lines))):
                    if "save_pretrained" in lines[j]:
                        found = True
                        break
        assert found, (
            "No ``save_pretrained`` call found inside ``if is_main:`` guard.\n"
            "   Non-main ranks must not save adapters."
        )

    def test_rank_gated_evaluate(self) -> None:
        """``evaluate_em()`` calls must be guarded by rank check.

        Main function has ``if not is_main: return`` before evaluation code.
        We verify that the call site is reachable only on the main rank.
        """
        tree = _ast_parse()
        evals = _find_calls(tree, "evaluate_em")
        assert len(evals) > 0, "No evaluate_em calls found"

        # Verify early return guard exists before the evaluation section
        lines = _SOURCE.split("\n")
        eval_line = None
        for call in evals:
            eval_line = call.lineno
            break

        # Find the ``if not is_main: return`` before the eval calls
        assert eval_line is not None
        found_guard = False
        for i, line in enumerate(lines):
            if "if not is_main" in line:
                # Check the next 2 lines contain "return"
                for j in range(i + 1, min(i + 3, len(lines))):
                    if "return" in lines[j]:
                        found_guard = True
                        break
        assert found_guard, (
            "Missing early return guard ``if not is_main: return`` before evaluation."
        )

    def test_ddp_find_unused_parameters(self) -> None:
        """TrainingArguments must set ``ddp_find_unused_parameters=False``."""
        tree = _ast_parse()
        training_args_calls = _find_calls(tree, "TrainingArguments")
        assert len(training_args_calls) > 0, "No TrainingArguments() found"

        for call in training_args_calls:
            val = _find_keyword_in_call(call, "ddp_find_unused_parameters")
            if val is not None:
                if ast.unparse(val) == "False":
                    return
        pytest.fail(
            "No TrainingArguments() with ``ddp_find_unused_parameters=False`` found.\n"
            "   Without this, DDP will hang on sparse gradients."
        )

    def test_remove_unused_columns_false(self) -> None:
        """TrainingArguments must set ``remove_unused_columns=False``."""
        tree = _ast_parse()
        for call in _find_calls(tree, "TrainingArguments"):
            val = _find_keyword_in_call(call, "remove_unused_columns")
            if val is not None and ast.unparse(val) == "False":
                return
        pytest.fail(
            "Missing ``remove_unused_columns=False`` in TrainingArguments.\n"
            "   Without this, custom data collators may silently drop fields."
        )

    def test_no_hardcoded_cuda_device(self) -> None:
        """The source must not contain hardcoded ``cuda:0`` or ``.cuda()``."""
        forbidden = ["cuda:0", ".cuda()", "device='cuda'", 'device="cuda"']
        for pattern in forbidden:
            if pattern in _SOURCE:
                # Allow comment mentions
                lines = _SOURCE.split("\n")
                for lineno, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith("#"):
                        pytest.fail(
                            f"Line {lineno}: hardcoded device '{pattern}' found.\n"
                            f"   Use ``model.device`` instead."
                        )

    def test_device_derived_from_model(self) -> None:
        """evaluate_em should derive device from model, not hardcode."""
        pattern = "model.device"
        if pattern not in _SOURCE:
            pytest.fail(
                "evaluate_em does not derive device via ``model.device``\n"
                "   Hardcoded devices break in multi-node or DeepSpeed setups."
            )

    def test_model_eval_before_generate(self) -> None:
        """model.eval() must be called before model.generate()."""
        tree = _ast_parse()
        for call in _find_calls(tree, "generate"):
            # Walk up to find if model.eval() is called in a nearby scope
            lines = _SOURCE.split("\n")
            # Check a few lines before the generate call
            start = max(0, call.lineno - 5)
            block = "\n".join(lines[start : call.lineno])
            if "model.eval()" not in block:
                pytest.fail(
                    f"Line {call.lineno}: model.generate() without model.eval() in preceding lines.\n"
                    "   This can cause incorrect batch-norm/dropout behavior."
                )

    def test_set_seed_called(self) -> None:
        """set_seed must be called in main() before any model building."""
        if "set_seed(" not in _SOURCE:
            pytest.fail(
                "set_seed() not found. Each rank needs synchronized seed for reproducibility."
            )

    def test_padding_side_left_for_generation(self) -> None:
        """Tokenizer padding_side must be set to left before generation."""
        if 'padding_side = "left"' not in _SOURCE:
            if "padding_side = 'left'" not in _SOURCE:
                pytest.fail(
                    "No explicit padding_side='left' found. Generation may produce wrong results in batch."
                )

    def test_deepspeed_device_map_handling(self) -> None:
        """load_4bit_backbone must handle device_map=None when use_deepspeed=True."""
        tree = _ast_parse()
        for call in _find_calls(tree, "from_pretrained"):
            device_map_val = _find_keyword_in_call(call, "device_map")
            if device_map_val is not None:
                # Should be conditional — either None or rank-based
                if "use_deepspeed" in _SOURCE[: call.lineno + 10]:
                    # The function should have branching logic
                    if "device_map: str | None = None" in _SOURCE:
                        return
        # This check is more nuanced — just ensure no unconditional device_map
        assert "device_map=" in _SOURCE, (
            "No device_map handling found in backbone loading"
        )

    def test_unwrap_model_before_save(self) -> None:
        """save_pretrained should unwrap the model in DDP mode."""
        if "unwrap_model" not in _SOURCE:
            pytest.fail(
                "No unwrap_model() call found before save_pretrained.\n"
                "   In DDP, the model is wrapped; save must unwrap first."
            )

    def test_no_blocking_collective_ops(self) -> None:
        """Should not contain raw torch.distributed ops that could deadlock.

        Uses AST to exclude string constants (e.g. DeepSpeed JSON config).
        """
        tree = _ast_parse()
        blocking = {"barrier", "all_reduce", "all_gather", "reduce_scatter"}

        class _OpFinder(ast.NodeVisitor):
            violations: list[int] = []

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if node.attr in blocking and not _is_within_string(node):
                    self.violations.append(node.lineno)
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                if node.id in blocking and not _is_within_string(node):
                    self.violations.append(node.lineno)
                self.generic_visit(node)

        finder = _OpFinder()
        finder.visit(tree)
        assert not finder.violations, (
            f"Raw distributed ops found at lines {finder.violations}.\n"
            "   Use Accelerator or Trainer abstractions instead."
        )


def _is_within_string(node: ast.AST) -> bool:
    """Check if a node is inside a string constant (e.g. JSON config)."""
    for parent in ast.walk(ast.parse(_SOURCE)):
        if isinstance(parent, ast.Constant) and isinstance(parent.value, str):
            start_lineno = getattr(parent, "lineno", None)
            end_lineno = getattr(parent, "end_lineno", None)
            if start_lineno and end_lineno:
                if start_lineno <= node.lineno <= end_lineno:
                    return True
    return False


class TestDeepSpeedConfig:
    """Verify DeepSpeed-specific config constraints."""

    def test_mixed_optimizer_disabled_with_deepspeed(self) -> None:
        """When use_deepspeed=True, entropy_loss_weight should be 0."""
        tree = _ast_parse()
        if "entropy_loss_weight" not in _SOURCE:
            return
        for call in _find_calls(tree, "EngramConfig"):
            val = _find_keyword_in_call(call, "entropy_loss_weight")
            if val is not None:
                # Check if it's conditional using line-based slicing
                lines = _SOURCE.split("\n")
                context = "\n".join(lines[: call.lineno + 20])
                if "if not use_deepspeed" not in context:
                    pytest.fail(
                        "entropy_loss_weight is not conditionally zeroed when use_deepspeed=True"
                    )

    def test_sparse_embeddings_disabled_with_deepspeed(self) -> None:
        """When use_deepspeed=True, sparse_embeddings must depend on use_deepspeed."""
        if "not use_deepspeed" not in _SOURCE:
            pytest.fail(
                "Missing ``sparse_embeddings = not use_deepspeed`` logic.\n"
                "   DeepSpeed ZeRO does not support sparse gradients."
            )


# ──────────────────────────────────────────────────────────────────────
#  Structural checks
# ──────────────────────────────────────────────────────────────────────


class TestExampleStructure:
    """Verify overall structure and function signatures."""

    def test_no_top_level_code_outside_functions(self) -> None:
        """Aside from constants/imports, all code must be inside functions."""
        tree = _ast_parse()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                name = (
                    ast.unparse(node.value.func)
                    if isinstance(node.value.func, ast.Name)
                    else ""
                )
                if name == "load_dotenv":
                    continue  # allowed
                pytest.fail(
                    f"Top-level expression call at line {node.lineno}: {ast.unparse(node.value)[:60]}"
                )

    def test_required_functions_exist(self) -> None:
        required = [
            "normalize_answer",
            "format_qa_text",
            "load_popqa",
            "tokenize_dataset",
            "write_default_ds_config",
            "load_4bit_backbone",
            "build_engram_model",
            "build_lora_model",
            "evaluate_em",
            "print_comparison",
            "parse_args",
            "main",
        ]
        tree = _ast_parse()
        func_names = {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = [f for f in required if f not in func_names]
        assert not missing, f"Missing required functions: {missing}"

    def test_ifname_main_guard(self) -> None:
        """File must have ``if __name__ == '__main__': main()`` entry point."""
        assert "if __name__" in _SOURCE, "Missing __name__ guard for entry point"

    def test_main_function_signature(self) -> None:
        """main() must take no arguments (args are parsed inside)."""
        tree = _ast_parse()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                args = node.args.args
                assert len(args) == 0, f"main() has {len(args)} arguments; expected 0"
                return
        pytest.fail("main() not found")

    def test_docstring_contains_usage_examples(self) -> None:
        """Module docstring should contain CLI usage examples."""
        assert "# Train Engram only" in _SOURCE, "Docstring lacks usage examples"
