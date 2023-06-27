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

.PHONY: test_with_h2ogpt
test_with_h2ogpt: H2OGPT_START=python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --load_8bit=True
test_with_h2ogpt: H2OGPT_SERVER=http://0.0.0.0:7860
test_with_h2ogpt:
	cd .. && . venv/bin/activate && \
	$(H2OGPT_START) &
	sleep 20 && wget --retry-connrefused --waitretry=30 --timeout=60 --tries=50 --output-document=/dev/null "$(H2OGPT_SERVER)"
	$(MAKE) H2OGPT_SERVER="$(H2OGPT_SERVER)" test
	pkill -f "$(H2OGPT_START)" &

.PHONY: build
build: lint
	poetry build
