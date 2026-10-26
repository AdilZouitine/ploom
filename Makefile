.PHONY: sync hooks format lint typecheck test check build docs benchmark

sync:
	uv sync --locked

hooks:
	uv run --locked prek install

format:
	uv run --locked ruff format .
	uv run --locked ruff check --fix .

lint:
	uv run --locked ruff format --check .
	uv run --locked ruff check .

typecheck:
	uv run --locked ty check

test:
	uv run --locked pytest --cov=ploom --cov-report=term-missing

check: lint typecheck test

build:
	uv build

docs:
	uv run --locked --group docs mkdocs build --strict

benchmark:
	uv run --locked --group benchmarks pytest benchmarks --codspeed
