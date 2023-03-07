init:
	pip install -r requirements.txt

test:
	pytest

train:
	python app/train.py

run:
	python app/run.py

.PHONY: init test train run
