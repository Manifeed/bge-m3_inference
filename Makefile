COMPOSE := $(shell if docker compose version >/dev/null 2>&1; then echo "docker compose"; elif docker-compose version >/dev/null 2>&1; then echo "docker-compose"; else echo "docker compose"; fi)
DC := $(COMPOSE) -f docker-compose.yml
PYTORCH_TEST_IMAGE ?= pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
PYTEST_ARGS ?= tests -vv --color=yes --tb=short -ra

.DEFAULT_GOAL := help

.PHONY: help build up down clean test

help:
	@printf '%s\n' 'Available targets:'
	@printf '%s\n' '  make build'
	@printf '%s\n' '  make up'
	@printf '%s\n' '  make down'
	@printf '%s\n' '  make clean'
	@printf '%s\n' '  make test'
	@printf '\n%s\n' 'Notes:'
	@printf '%s\n' '  - copy .env.example to .env before build/up.'
	@printf '%s\n' '  - make test runs pytest in a disposable container.'

build:
	$(DC) build

up:
	$(DC) up --build -d

down:
	$(DC) down

clean:
	$(DC) down --remove-orphans --rmi local -v

test:
	docker run --rm \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		$(PYTORCH_TEST_IMAGE) \
		bash -lc "python3 -m pip install --disable-pip-version-check -r requirements.txt -r requirements-dev.txt && python3 -m pytest $(PYTEST_ARGS)"

