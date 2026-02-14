#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL := /bin/bash
PROJECT_NAME = poetry-runner #fast-o3-iv
PYTHON_VERSION = 3.11
CONDA = conda
POETRY = poetry

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create conda environment (Poetry + conda-forge regridding stack with esmpy)
.PHONY: create_environment
create_environment:
	$(CONDA) create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	$(CONDA) install -n $(PROJECT_NAME) -c conda-forge poetry esmf esmpy xesmf -y
	@echo ">>> conda env created. Activate with:"
	@echo "conda activate $(PROJECT_NAME)"

## Install all dependencies 
.PHONY: requirements
requirements:
	$(CONDA) run -n $(PROJECT_NAME) $(POETRY) config virtualenvs.create false --local
	$(CONDA) run -n $(PROJECT_NAME) $(POETRY) install

## One-shot setup
.PHONY: bootstrap
bootstrap: create_environment requirements

## Delete compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff
.PHONY: lint
lint:
	$(CONDA) run -n $(PROJECT_NAME) $(POETRY) run ruff format --check
	$(CONDA) run -n $(PROJECT_NAME) $(POETRY) run ruff check

## Format source code with ruff
.PHONY: format
format:
	$(CONDA) run -n $(PROJECT_NAME) $(POETRY) run ruff check --fix
	$(CONDA) run -n $(PROJECT_NAME) $(POETRY) run ruff format

## Run tests
.PHONY: test
test:
	$(CONDA) run -n $(PROJECT_NAME) $(POETRY) run pytest --forked tests/

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
