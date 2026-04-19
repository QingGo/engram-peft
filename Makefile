# Makefile for engram-peft development
# This file provides shortcuts for common development tasks such as linting,
# formatting, type checking, and testing using the project's preferred tools.

.PHONY: help lint format type-check test clean all daemon-stop daemon-status

# Default target: display help information for all available targets
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  lint           Check code style and formatting without making changes"
	@echo "  format         Auto-fix linting issues and reformat code"
	@echo "  type-check     Run static type analysis using dmypy (mypy daemon)"
	@echo "  test           Run all tests (unit tests) with coverage"
	@echo "  test-unit      Run only lightweight unit tests (< 1s)"
	@echo "  test-integ     Run integration tests (< 1m)"
	@echo "  test-daemon    Start development test acceleration daemon"
	@echo "  clean          Remove all temporary files, caches, and stop the daemon"
	@echo "  all            Run format, type-check, and test in sequence"
	@echo "  daemon-status  Check if the dmypy daemon is currently running"
	@echo "  daemon-stop    Gracefully stop the dmypy daemon"

# Configuration: Paths to be analyzed by linting and type checking tools
PATHS = src/ tests/ examples/
# Configuration: Core source directory for coverage reporting
SRC_DIR = src/engram_peft

# Run ruff check and format in 'check' mode to identify style violations
lint:
	uv run ruff check $(PATHS)
	uv run ruff format --check $(PATHS)

# Run ruff with --fix to resolve auto-fixable issues and reformat the entire codebase
format:
	uv run ruff check --fix $(PATHS)
	uv run ruff format $(PATHS)

# Run static type checking. 'dmypy run' will start the daemon if it's not running
# and perform an incremental check, which is significantly faster for subsequent runs.
type-check:
	uv run dmypy run -- $(PATHS)

# Run unit tests using sprintest, which provides optimized test execution.
# Includes coverage calculation and identifies the slowest test cases.
test: test-unit

test-unit:
	@echo "[*] Running lightweight unit tests..."
	uv run sprintest tests/unit --cov=$(SRC_DIR) --cov-report=term-missing --durations=5

test-integ:
	@echo "[*] Running integration tests..."
	uv run sprintest tests/integration --durations=5

# Start the sprintest daemon for test acceleration.
# This keeps the model and weights in memory to eliminate startup latency.
test-daemon:
	SPRINTEST_TARGET_PKG=engram-peft uv run sprintest-daemon

# Stop the type-check daemon and delete all tool-generated cache directories
clean: daemon-stop
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/ build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Display the current status of the dmypy background process
daemon-status:
	uv run dmypy status

# Stop the dmypy background process to free up memory or resolve sync issues
daemon-stop:
	uv run dmypy stop

# Comprehensive check: ensures the code is formatted, type-safe, and passes all tests
all: format type-check test
