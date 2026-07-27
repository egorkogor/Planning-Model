.PHONY: install validate-spec test lint check clean-caches status

install:
	python -m pip install -e '.[dev]'

validate-spec:
	python validation/validate_bundle.py

test:
	python -m pytest -q

lint:
	python -m ruff check src tests

check: validate-spec test

clean-caches:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +

status:
	@sed -n '1,40p' RUN_STATUS.md
