POETRY_INSTALL_DIR := $(abspath ./.poetry)
POETRY_BIN         := $(POETRY_INSTALL_DIR)/bin/poetry

PACKAGE_NAME    = $(firstword $(shell $(POETRY_BIN) version))
PACKAGE_DIR     = $(subst -,_,$(PACKAGE_NAME))
PACKAGE_VERSION = $(shell $(POETRY_BIN) version --short)

$(POETRY_BIN):
	@echo "Installing Poetry into '$(POETRY_INSTALL_DIR)' ..."
	curl -sSL https://install.python-poetry.org | POETRY_HOME="$(POETRY_INSTALL_DIR)" python3 - --force --version 1.5.1

.PHONY: _generate_sources
_generate_sources:
	cp -f ./../enums.py "$(PACKAGE_DIR)/_enums.py"

.PHONY: clean
clean:
	rm -rf dist
	rm "$(PACKAGE_DIR)/_enums.py"

.PHONY: clean_deep
clean_deep: clean
	rm -rf "$(POETRY_INSTALL_DIR)"
	rm -rf ".venv"

.PHONY: setup
setup: $(POETRY_BIN)
	$(POETRY_BIN) install

.PHONY: setup_test
setup_test:
	$(POETRY_BIN) install --only=test

.PHONY: lint
lint: _generate_sources
	$(POETRY_BIN) run black .
	$(POETRY_BIN) run isort .
	$(POETRY_BIN) run flake8 "$(PACKAGE_DIR)" "tests" || true
	$(POETRY_BIN) run mypy --show-error-codes --pretty .

.PHONY: test
test: _generate_sources
	$(POETRY_BIN) run pytest -r=A

.PHONY: build
build: _generate_sources
	$(POETRY_BIN) build
