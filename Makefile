.PHONY: install install-serving install-mlops validate prepare train evaluate predict gradcam serve tune test lint

install:
	pip install -r requirements.txt
	pip install -e .

install-serving:
	pip install -r requirements-serving.txt
	pip install -e .

install-mlops:
	pip install -r requirements-mlops.txt
	pip install -e .

validate:
	python -m skin_cancer.data.validation --config configs/train_config.yaml

prepare:
	python -m skin_cancer.data.preparation --config configs/train_config.yaml

train:
	python -m skin_cancer.training.train --config configs/train_config.yaml --use-weighted-sampler

evaluate:
	python -m skin_cancer.evaluation.evaluate --config configs/train_config.yaml

predict:
	python -m skin_cancer.inference.predict --config configs/train_config.yaml --image $(IMAGE)

gradcam:
	python -m skin_cancer.explainability.gradcam --config configs/train_config.yaml --image $(IMAGE)

serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000

tune:
	python mlops/ray/tune.py --config configs/train_config.yaml

test:
	pytest -q

lint:
	ruff check .
