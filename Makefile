CMD:=poetry run

all: test lint

lint:
	poetry check
	$(CMD) isort --check .
	$(CMD) black --check .
	$(CMD) flake8
	$(CMD) mypy

test:
	$(CMD) pytest