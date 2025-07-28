#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = artifacts-conext25-ipv6-scanner
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	$(shell rm notebooks/*.html)
	$(shell rm reports/figures/*.pdf)
	$(shell rm reports/figures/*.png)

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ipv6_scanner
	isort --check --diff --profile black ipv6_scanner
	black --check --config pyproject.toml ipv6_scanner

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml ipv6_scanner


## Convert notebooks to html
notebooks=$(shell ls notebooks/*.ipynb)
notebooks_html:=$(subst .ipynb,.html,$(notebooks))

%.html: %.ipynb
	jupyter nbconvert $(NBCONVERT_PARAMS) --to html $< 

nbconvert: $(notebooks_html)

nbconvert-clean-execute: NBCONVERT_PARAMS=--execute
nbconvert-clean-execute: $(shell rm notebooks/*.html)
nbconvert-clean-execute: $(notebooks_html)

python_env:
	python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) ipv6_scanner/dataset.py

.PHONY: plots 
plots: requirements
	$(PYTHON_INTERPRETER) ipv6_scanner/plots.py

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
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
