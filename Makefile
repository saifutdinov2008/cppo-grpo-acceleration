# Convenience targets. All of them assume the virtualenv is active
# (see `bash scripts/setup.sh`).
.PHONY: help setup lint types test check smoke train-grpo train-cppo report clean

help:
	@echo "setup       create .venv and install the project"
	@echo "check       lint + types + tests (what CI runs)"
	@echo "smoke       full CPU end-to-end validation"
	@echo "train-grpo  train the GRPO baseline"
	@echo "train-cppo  train CPPO at P = 0.75"
	@echo "report      build report/report.pdf"

setup:
	bash scripts/setup.sh

lint:
	python -m pylint src/cppo tests benchmarks

types:
	python -m mypy

test:
	python -m pytest tests -q

check: lint types test

smoke:
	bash scripts/smoke_test.sh

train-grpo:
	bash scripts/train.sh configs/grpo.yaml

train-cppo:
	bash scripts/train.sh configs/cppo_p75.yaml

report:
	$(MAKE) -C report

clean:
	rm -rf outputs .pytest_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
