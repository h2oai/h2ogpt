PYTHON_VERSION ?= 3.10
PYTHON ?= python$(PYTHON_VERSION)
VENV?=venv

all: reqs_optional/req_constraints.txt

.PHONY: reqs_optional/req_constraints.txt
reqs_optional/req_constraints.txt:
	grep -v '#\|peft\|transformers\|accelerate' requirements.txt > $@

.PHONY: venv
venv: ## Create VENV
	$(PYTHON) -m venv $(VENV)

.PHONY: setup
setup: venv ## Setup VENV with the requirements.txt
	./$(VENV)/bin/python -m pip install -r requirements.txt

.PHONY: clean
clean: ## Clean VENV
	rm -rf $(VENV)
