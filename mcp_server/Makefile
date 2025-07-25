.PHONY: install install-dev format lint test all check clean

# Define variables
PYTHON = python3
UV = uv
PYTEST = $(UV) run pytest
RUFF = $(UV) run ruff
PYRIGHT = $(UV) run pyright
BLACK = $(UV) run black

# Default target
all: format lint test

# Install production dependencies
install:
	$(UV) sync

# Install development dependencies
install-dev:
	$(UV) sync --extra dev

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check
	$(PYRIGHT) .

# Run tests
test:
	$(PYTEST)

# Run format, lint, and test
check: format lint test

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f .coverage
	rm -f coverage.xml

# Show help
help:
	@echo "Available targets:"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  format      - Format code with ruff"
	@echo "  lint        - Lint code with ruff and pyright"
	@echo "  test        - Run tests with pytest"
	@echo "  check       - Run format, lint, and test"
	@echo "  clean       - Clean up generated files"
	@echo "  all         - Run format, lint, and test (default)"
	@echo "  help        - Show this help message"
