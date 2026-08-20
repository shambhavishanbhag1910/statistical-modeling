.PHONY: setup data train test lint api all batch drift docker

setup:
	python -m pip install -r requirements-dev.txt

data:
	python scripts/generate_data.py

train:
	python -m failure_risk.train

test:
	pytest -q

lint:
	ruff check failure_risk tests scripts

api:
	uvicorn failure_risk.api:app --host 0.0.0.0 --port 8000 --reload

all:
	python scripts/run_all.py

batch:
	python scripts/create_scoring_sample.py
	python -m failure_risk.batch data/scoring_sample.csv reports/batch_scored.csv

drift:
	python scripts/drift_demo.py

docker:
	docker build -t vehicle-failure-risk:latest .
