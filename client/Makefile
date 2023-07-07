POETRY_INSTALL_DIR := $(abspath ./.poetry)
POETRY_BIN         := $(POETRY_INSTALL_DIR)/bin/poetry

PACKAGE_NAME    = $(firstword $(shell $(POETRY_BIN) version))
PACKAGE_DIR     = $(subst -,_,$(PACKAGE_NAME))
PACKAGE_VERSION = $(shell $(POETRY_BIN) version --short)

H2OGPT_FILE_TO_CLIENT := enums.py:_enums.py

define CREATE_COPY_FILE_TARGET
$(eval TARGET_CLIENT_DIR := h2ogpt_client)

.PHONY: copy_$(1)_to_$(2)
copy_$(1)_to_$(2):
	cp -f "./../$(1)" "$(TARGET_CLIENT_DIR)/$(2)"

copy_files_from_h2ogpt: copy_$(1)_to_$(2)

.PHONY: clean_$(2)
clean_$(2):
	rm "$(TARGET_CLIENT_DIR)/$(2)"

clean: clean_$(2)
endef
$(foreach i,$(H2OGPT_FILE_TO_CLIENT),$(eval $(call CREATE_COPY_FILE_TARGET,$(word 1,$(subst :, ,$i)),$(word 2,$(subst :, ,$i)))))

copy_files_from_h2ogpt: ;

$(POETRY_BIN):
	@echo "Installing Poetry into '$(POETRY_INSTALL_DIR)' ..."
	curl -sSL https://install.python-poetry.org | POETRY_HOME="$(POETRY_INSTALL_DIR)" python3 - --force --version 1.5.1

.PHONY: clean
clean:
	rm -rf dist

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
