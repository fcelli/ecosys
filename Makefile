init:
	pip install -r requirements.txt

test:
	pytest

train:
	python train.py

run:
	python run.py

.PHONY: init test train run
