.PHONY: install install-dev test lint format run eval seed data real-data clean help

PY ?= python
PIP ?= $(PY) -m pip
APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
NIH_DIR ?= $(CURDIR)/../Dataset/images
NIH_CSV ?= $(CURDIR)/../Dataset/Data_Entry_2017_v2020.csv
NIH_N ?= 50

help:
	@echo "Targets:"
	@echo "  install     Create .venv and install runtime deps"
	@echo "  install-dev Create .venv, install runtime + dev deps, and pip install -e ."
	@echo "  test        Run pytest"
	@echo "  lint        Run ruff + mypy"
	@echo "  format      Run black + ruff --fix"
	@echo "  run         Run uvicorn on $(APP_HOST):$(APP_PORT)"
	@echo "  data        Build a stub manifest (first-time setup only)"
	@echo "  real-data   Build the real NIH manifest (default ..\\Dataset\\images + Data_Entry_2017_v2020.csv)"
	@echo "  eval        Run scripts/run_eval.py"
	@echo "  seed        Seed the first admin user (scripts/seed_admin.py)"
	@echo "  clean       Remove caches and the local data dir"

install:
	$(PY) -m venv .venv
	. .venv/bin/activate && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt && $(PIP) install -e .

install-dev:
	$(PY) -m venv .venv
	. .venv/bin/activate && $(PIP) install --upgrade pip && $(PIP) install -r requirements-dev.txt && $(PIP) install -e .

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src

format:
	$(PY) -m black src tests
	$(PY) -m ruff check --fix src tests

run:
	$(PY) -m uvicorn cxt.interfaces.api.app:app --reload --host $(APP_HOST) --port $(APP_PORT)

data:
	$(PY) -m scripts.download_nih_subset --source stub --n-per-finding 5
	$(PY) -m scripts.build_manifest --n-per-finding 5 --n-negatives 5

real-data:
	rm -rf data/raw data/processed
	$(PY) -m scripts.download_nih_subset --source local --local-dir "$(NIH_DIR)" --metadata-csv "$(NIH_CSV)" --n-per-finding $(NIH_N)
	$(PY) -m scripts.build_manifest --n-per-finding $(NIH_N) --n-negatives $(NIH_N)

eval:
	$(PY) scripts/run_eval.py

seed:
	$(PY) scripts/seed_admin.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov cxt.db data/raw data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
