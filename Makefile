.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: help
help:			
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: lint
lint:			
	@echo "Running linters"
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: test-unit
test-unit:			
	@echo "Running Unit tests"
	uv run pytest tests/unit

.PHONY: pre-push
pre-push:			
	@echo "Running pre-push checks"
	uv run ruff check .
	make type-check
	make test_unit

.PHONY: test-infrastructure
test-infrastructure:			
	@echo "Running infrastructure tests"
	uv run pytest tests/infrastructure

.PHONY: local-run
local-run:
	@echo "Running idu for local development"
	uv run manage.py runserver 0.0.0.0:8000

.PHONY: docker-up
docker-up:
	@echo "Starting idu container"
	docker compose up --build

.PHONY: docker-down
docker-down:
	@echo "Stopping idu container"
	docker compose down
