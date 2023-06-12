PACKAGE_NAME    := $(firstword $(shell poetry version))
PACKAGE_DIR     := $(subst -,_,$(PACKAGE_NAME))
PACKAGE_VERSION := $(shell poetry version --short)

generate_sources:
	cp -f ./../enums.py "$(PACKAGE_DIR)/enums.py"

.PHONY: setup
setup:
	poetry install

.PHONY: lint
lint: generate_sources
	poetry run black .
	poetry run isort .
	poetry run flake8 "$(PACKAGE_DIR)" "tests" || true
	poetry run mypy --show-error-codes --pretty .

.PHONY: test
test: generate_sources
	poetry run pytest

.PHONY: build
build: generate_sources
	poetry build
