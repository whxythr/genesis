PY ?= python
export PYTHONPATH := src

.PHONY: check test generate validate baselines demo

check:
	$(PY) -m audit_env check

test:
	$(PY) -m pytest -q

generate:
	$(PY) -m audit_env generate --split train
	$(PY) -m audit_env generate --split dev

validate:
	$(PY) -m audit_env validate --split train
	$(PY) -m audit_env validate --split dev

baselines:
	$(PY) -m audit_env baselines --split dev

demo:
	$(PY) -m audit_env demo
