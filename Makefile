.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: populate-vectordb
populate-vectordb:			
	@echo "Running command to populate vectordb"
	uv run manage.py populate_vectordb

.PHONY: populate-vectordb-test
populate-vectordb-test:			
	@echo "Running command to populate vectordb with test data"
	uv run manage.py populate_vectordb --dataset-path data/test

.PHONY: lint
lint:			
	@echo "Running linters"
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: unit-test
unit-test:			
	@echo "Running Unit tests"
	uv run pytest tests/unit -v

.PHONY: pre-push
pre-push:			
	@echo "Running pre-push checks"
	uv run ruff check .
	make unit-test

.PHONY: integration-test
integration-test:			
	@echo "Running integration tests"
	uv run pytest tests/integration -v

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
