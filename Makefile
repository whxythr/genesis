# Python: the project's .venv when it exists (local runs), otherwise `python` (Docker image,
# where requirements.lock is installed into the image's interpreter). Override with PY=...
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
export PYTHONPATH := src

.PHONY: check setup test generate validate verify baselines invariance demo

check:          ## everything: tests, data validation, file-level verify, 4 baselines, paired invariance
	$(PY) -m audit_env check

setup:          ## local only: create .venv and install the pinned dependencies (needs internet once)
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.lock

test:
	$(PY) -m pytest -q

generate:       ## regenerate the 10 train + 10 dev episodes
	$(PY) -m audit_env generate --split train
	$(PY) -m audit_env generate --split dev

validate:
	$(PY) -m audit_env validate --split train
	$(PY) -m audit_env validate --split dev

verify:         ## independent file-level check of every episode (Block 2 conditions)
	$(PY) -m audit_env verify --split train
	$(PY) -m audit_env verify --split dev

baselines:      ## train in open mode, dev in closed eval mode
	$(PY) -m audit_env baselines --split train --mode train
	$(PY) -m audit_env baselines --split dev --mode eval

invariance:
	$(PY) -m audit_env invariance

demo:           ## scripted auditor step by step + final P/R/F1/reward
	$(PY) -m audit_env demo
