PY ?= python
export PYTHONPATH := src

.PHONY: check test generate validate baselines invariance demo

check:          ## everything: tests, data validation, 4 baselines, paired invariance
	$(PY) -m audit_env check

test:
	$(PY) -m pytest -q

generate:       ## regenerate the 10 train + 10 dev episodes
	$(PY) -m audit_env generate --split train
	$(PY) -m audit_env generate --split dev

validate:
	$(PY) -m audit_env validate --split train
	$(PY) -m audit_env validate --split dev

baselines:      ## train in open mode, dev in closed eval mode
	$(PY) -m audit_env baselines --split train --mode train
	$(PY) -m audit_env baselines --split dev --mode eval

invariance:
	$(PY) -m audit_env invariance

demo:           ## scripted auditor step by step + final P/R/F1/reward
	$(PY) -m audit_env demo
