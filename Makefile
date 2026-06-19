# Makefile for PKG v2

.PHONY: help install test lint type format check clean benchmark docs

help:
	@echo "PKG v2 - Port-Hamiltonian Digital Twins"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install package in development mode with all extras"
	@echo "  install-core Install only core dependencies (numpy + scipy)"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run tests excluding slow ones"
	@echo "  lint         Run ruff linter"
	@echo "  type         Run mypy type checker (strict)"
	@echo "  format       Format code with black"
	@echo "  check        Run all checks (lint + type + test)"
	@echo "  clean        Remove generated files"
	@echo "  benchmark    Run benchmarks and generate reports"
	@echo "  docs         Build documentation (TODO)"

install:
	pip install -e ".[dev,torch,viz]"

install-core:
	pip install -e .

test:
	PYTHONPATH=src pytest tests/ -v

test-fast:
	PYTHONPATH=src pytest tests/ -v -m "not slow"

test-coverage:
	PYTHONPATH=src pytest tests/ --cov=PKG --cov-report=term-missing --cov-report=html

lint:
	ruff check src/PKG/

type:
	mypy --strict src/PKG/

format:
	black src/PKG/ tests/
	ruff check --fix src/PKG/

check: lint type test-fast
	@echo "✅ All checks passed!"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage

benchmark:
	@echo "Running benchmarks..."
	PYTHONPATH=src python benchmarks/run_benchmarks.py
	@echo "✅ Benchmarks complete. Results in benchmarks/results/"

verify-structure:
	@echo "Verifying v2 structure..."
	@test -f pyproject_v2.toml || (echo "❌ pyproject_v2.toml missing" && exit 1)
	@test -f CITATIONS.md || (echo "❌ CITATIONS.md missing" && exit 1)
	@test -f CHANGELOG.md || (echo "❌ CHANGELOG.md missing" && exit 1)
	@test -f NOTICE_v1_to_v2.md || (echo "❌ NOTICE_v1_to_v2.md missing" && exit 1)
	@test -d src/PKG || (echo "❌ src/PKG missing" && exit 1)
	@test -f src/PKG/py.typed || (echo "❌ py.typed marker missing" && exit 1)
	@echo "✅ Structure verified"

.PHONY: verify-structure
