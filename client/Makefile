POETRY_INSTALL_DIR := $(abspath ./.poetry)
POETRY_BIN         := $(POETRY_INSTALL_DIR)/bin/poetry

PACKAGE_NAME    = $(firstword $(shell $(POETRY_BIN) version))
PACKAGE_DIR     = $(subst -,_,$(PACKAGE_NAME))
PACKAGE_VERSION = $(shell $(POETRY_BIN) version --short)

# Space separated list of file path that needs to be copied from h2oGPT.
FILES_FROM_H2OGPT := enums.py
NAME_PREFIX_FOR_FILES_COPIED_FROM_H2OGPT = _h2ogpt_

$(POETRY_BIN):
	@echo "Installing Poetry into '$(POETRY_INSTALL_DIR)' ..."
	curl -sSL https://install.python-poetry.org | POETRY_HOME="$(POETRY_INSTALL_DIR)" python3 - --force --version 1.5.1

.PHONY: copy_files_from_h2ogpt
copy_files_from_h2ogpt:
	for file in $(FILES_FROM_H2OGPT); do \
		dst="$(PACKAGE_DIR)/$(NAME_PREFIX_FOR_FILES_COPIED_FROM_H2OGPT)$(notdir $$file)"; \
		echo "Copying '$$file' to '$$dst' ..."; \
		cp -f "./../src/$$file" "$$dst"; \
	done

.PHONY: clean
clean:
	rm -rf dist
	find "$(PACKAGE_DIR)" -name "$(NAME_PREFIX_FOR_FILES_COPIED_FROM_H2OGPT)*" -delete

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
lint: copy_files_from_h2ogpt
	$(POETRY_BIN) run black .
	$(POETRY_BIN) run isort .
	$(POETRY_BIN) run flake8 "$(PACKAGE_DIR)" "tests" || true
	$(POETRY_BIN) run mypy --show-error-codes --pretty .

.PHONY: test
test: copy_files_from_h2ogpt
	$(POETRY_BIN) run pytest -r=A

.PHONY: build
build: copy_files_from_h2ogpt
	$(POETRY_BIN) build
