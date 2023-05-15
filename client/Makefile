
.PHONY: setup
setup:
	poetry install

.PHONY: lint
lint:
	poetry run black .
	poetry run isort .
	poetry run flake8 . || true
	poetry run mypy --show-error-codes --pretty .

.PHONY: test
test:
	poetry run pytest

.PHONY: build
build:
	poetry build
