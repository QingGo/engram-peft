# Contributing to Engram-PEFT

Thank you for contributing! To maintain high code quality and performance alignment with the Engram paper, we follow a strictly tiered development workflow.

## 🛠 Tiered Development Workflow (L1-L4)

All contributors (including AI Agents) should follow these layers of verification:

### L1: Real-time Feedback (IDE)
- **Tool**: Pyright / Pylance
- **Goal**: Catch obvious type errors and syntax bugs while typing.
- **Config**: Tests are excluded from strict checking; focus your strict energy on `src/`.

### L2: Automated Pre-commit (Standardization)
- **Tool**: Git Hooks (via `pre-commit`)
- **Commands**: Automatically runs on `git commit`.
- **Actions**: 
  - **Ruff (Format)**: Auto-formats code (88 chars, Black-compatible).
  - **Ruff (Lint)**: Auto-sorts imports and fixes simple logic errors (unused variables, etc.).

### L3: Deep Verification (Deep Logic)
- **Tools**: Mypy & sprintest (Unit)
- **Command**: `make type-check && make test-unit`
- **Goal**: Ensure 100% type safety in `src/` and verify core logic in <2 seconds.
- **Philosophy**: Use the `tiny_tokenizer` and `tiny_compressor` fixtures for fast unit testing.

### L4: Integration & Regression (Full Fidelity)
- **Tool**: sprintest (Integration)
- **Command**: `make test-integ`
- **Goal**: Verify algorithmic correctness using real-world models (GPT2, DeepSeek) and 32-layer weight migrations.
- **Frequency**: Run before submitting a PR or releasing a version.

---

## 🧪 Testing Guidelines

1. **Unit Tests (`tests/unit/`)**: Should involve mocked dependencies. They MUST run in under 1 second per test case.
2. **Integration Tests (`tests/integration/`)**: Used for high-fidelity verification. These are allowed to be heavy and slow.
3. **Mocking**: Use `MockTokenizer` from `tests/conftest.py` for unit tests to avoid Hub/Network overhead.

## 📏 Coding Style
- Follow PEP 8 (handled by Ruff).
- Docstrings are encouraged for all public APIs.
- Type hints are **mandatory** for all code in `src/`. For `tests/`, return types are optional but recommended for clarity.

---

## ⚡ Sprintest (Test Acceleration)

Sprintest is a C/S (Client/Server) architecture test runner specifically designed for heavy AI projects. By keeping large models and datasets in memory, it eliminates the test startup latency caused by slow loading.

### Usage

Before using `sprintest` to run tests, you **MUST** start the Daemon in the background or in a new terminal:

```bash
# Start the test acceleration server
make test-daemon
```

### Makefile Shortcuts

For a better development experience, use the provided `Makefile` targets:

| Command | Action |
| :--- | :--- |
| `make lint` | Check style |
| `make format` | Fix and format code |
| `make type-check` | Incremental type checking |
| `make test-unit` | Run fast unit tests |
| `make test-integ` | Run slow integration tests |
| `make all` | Run everything before PR |

### Core Advantages

- **Pre-loading**: Loads heavy dependencies (e.g., PyTorch, Transformers, or large datasets) into the daemon process, reducing test startup time from minutes to seconds.
- **Powerful Hot-reloading**: Automatically detects and clears modified modules in the current directory, ensuring tests run on the latest code without restarting the daemon.
- **Agent Friendly**: Designed for AI coding assistants—providing fast feedback loops, clean output (no ANSI characters), and stable communication.

---

## 🚀 Environment Setup
```bash
# Install development dependencies
uv sync --all-groups

# Initialize pre-commit hooks
uv run pre-commit install
```
