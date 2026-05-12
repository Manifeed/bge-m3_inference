COMPOSE := $(shell if docker compose version >/dev/null 2>&1; then echo "docker compose"; elif docker-compose version >/dev/null 2>&1; then echo "docker-compose"; else echo "docker compose"; fi)
DC := $(COMPOSE) -f docker-compose.yml
RELEASE_DC := $(COMPOSE) -f docker-compose.release.yml
PYTORCH_TEST_IMAGE ?= pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
PYTEST_ARGS ?= tests -vv --color=yes --tb=short -ra
PYTHON ?= python3
LOAD_DOTENV := set -a; if [ -f .env ]; then . ./.env; fi; set +a;
GPU_PRECHECK := if [ "$${BGE_M3_INFERENCE_SKIP_GPU_CHECK:-0}" != "1" ]; then \
		if ! command -v nvidia-smi >/dev/null 2>&1; then \
			printf '%s\n' 'nvidia-smi is not available on the host. Install the NVIDIA driver/toolkit or rerun with BGE_M3_INFERENCE_SKIP_GPU_CHECK=1 if you know the container runtime is already configured.'; \
			exit 2; \
		fi; \
		if ! nvidia-smi >/dev/null 2>&1; then \
			printf '%s\n' 'The NVIDIA driver is not loaded on the host. Start/fix the NVIDIA driver before running make up.'; \
			exit 2; \
		fi; \
	fi
CANONICALIZED_ENV := export BGE_M3_INFERENCE_API_KEY="$${BGE_M3_INFERENCE_API_KEY:-$${EMBEDDING_SERVICE_API_KEY:-}}"; \
	export BGE_M3_INFERENCE_PORT="$${BGE_M3_INFERENCE_PORT:-$${EMBEDDING_SERVICE_PORT:-8000}}"; \
	export BGE_M3_INFERENCE_CONTAINER_NAME="$${BGE_M3_INFERENCE_CONTAINER_NAME:-$${EMBEDDING_SERVICE_CONTAINER_NAME:-bge-m3_inference}}"; \
	export BGE_M3_INFERENCE_IMAGE_TAG="$${BGE_M3_INFERENCE_IMAGE_TAG:-latest}"; \
	export BGE_M3_INFERENCE_BATCH_MAX_ITEMS="$${BGE_M3_INFERENCE_BATCH_MAX_ITEMS:-64}"; \
	export BGE_M3_INFERENCE_BATCH_MAX_TOKENS="$${BGE_M3_INFERENCE_BATCH_MAX_TOKENS:-16384}"; \
	export BGE_M3_INFERENCE_BATCH_MAX_WAIT_MS="$${BGE_M3_INFERENCE_BATCH_MAX_WAIT_MS:-5}"; \
	export BGE_M3_INFERENCE_QUEUE_MAX_ITEMS="$${BGE_M3_INFERENCE_QUEUE_MAX_ITEMS:-1024}"; \
	export BGE_M3_INFERENCE_MAX_LENGTH="$${BGE_M3_INFERENCE_MAX_LENGTH:-512}"; \
	export BGE_M3_INFERENCE_REQUEST_TIMEOUT_SECONDS="$${BGE_M3_INFERENCE_REQUEST_TIMEOUT_SECONDS:-300}"; \
	export BGE_M3_INFERENCE_SHUTDOWN_GRACE_SECONDS="$${BGE_M3_INFERENCE_SHUTDOWN_GRACE_SECONDS:-30}"; \
	export BGE_M3_INFERENCE_USE_FP16="$${BGE_M3_INFERENCE_USE_FP16:-true}"; \
	export HF_HUB_OFFLINE="$${HF_HUB_OFFLINE:-1}"; \
	export TRANSFORMERS_OFFLINE="$${TRANSFORMERS_OFFLINE:-1}";

.DEFAULT_GOAL := help

.PHONY: help build up down clean test benchmark release-pull release-up

help:
	@printf '%s\n' 'Available targets:'
	@printf '%s\n' '  make build'
	@printf '%s\n' '  make up'
	@printf '%s\n' '  make down'
	@printf '%s\n' '  make clean'
	@printf '%s\n' '  make test'
	@printf '%s\n' '  make benchmark'
	@printf '%s\n' '  make release-pull'
	@printf '%s\n' '  make release-up'
	@printf '\n%s\n' 'Notes:'
	@printf '%s\n' '  - copy .env.example to .env before build/up.'
	@printf '%s\n' '  - make test runs pytest in a disposable container.'
	@printf '%s\n' '  - make benchmark expects the service to already be running.'
	@printf '%s\n' '  - make release-pull / release-up use GHCR images via docker-compose.release.yml.'
	@printf '%s\n' '  - make up requires a working NVIDIA driver unless BGE_M3_INFERENCE_SKIP_GPU_CHECK=1 is set.'

build:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(DC) build

up:
	@$(LOAD_DOTENV) $(GPU_PRECHECK); $(CANONICALIZED_ENV) $(DC) up --build -d

down:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(DC) down

clean:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(DC) down --remove-orphans --rmi local -v

test:
	docker run --rm \
		-v "$(CURDIR):/workspace" \
		-w /workspace \
		$(PYTORCH_TEST_IMAGE) \
		bash -lc "python3 -m pip install --disable-pip-version-check -r requirements.txt -r requirements-dev.txt && python3 -m pytest $(PYTEST_ARGS)"

benchmark:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(PYTHON) scripts/benchmark_inference.py \
		--base-url "$${BENCHMARK_BASE_URL:-http://127.0.0.1:$${BGE_M3_INFERENCE_PORT}}" \
		--api-key "$${BGE_M3_INFERENCE_API_KEY:?Set BGE_M3_INFERENCE_API_KEY or EMBEDDING_SERVICE_API_KEY}" \
		--repeats "$${BENCHMARK_REPEATS:-5}" \
		--ready-timeout-seconds "$${BENCHMARK_READY_TIMEOUT_SECONDS:-60}" \
		$${BENCHMARK_EXTRA_ARGS:---dense}

release-pull:
	@$(LOAD_DOTENV) $(CANONICALIZED_ENV) $(RELEASE_DC) pull

release-up:
	@$(LOAD_DOTENV) $(GPU_PRECHECK); $(CANONICALIZED_ENV) $(RELEASE_DC) up -d
