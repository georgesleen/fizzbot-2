IMAGE_NAME ?= fizzbot-llm
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKERFILE ?= $(ROOT_DIR)/llm/Dockerfile

.PHONY: help docker-build docker-train-gpu docker-smoke docker-train-cpu docker-smoke-cpu local-smoke gen-training-data test-latest

help:
	@echo "Targets:"
	@echo "  gen-training-data Generate training examples JSONL"
	@echo "  local-smoke       Run a tiny local smoke test"
	@echo "  test-latest       Run inference against latest trained model"
	@echo "  docker-build      Build the training Docker image"
	@echo "  docker-train-gpu  Run GPU training in Docker (no build)"
	@echo "  docker-smoke      Run GPU smoke test in Docker (no build)"
	@echo "  docker-train-cpu  Run CPU training in Docker (no build)"
	@echo "  docker-smoke-cpu  Run CPU smoke test in Docker (no build)"

gen-training-data:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/gen_training_data.py

test-latest:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --latest --decode --speaker "<S0>" --content "hello everyone"

fix-uv-cache:
	sudo rm -rf $$HOME/.cache/uv

fix-venv-perms:
	sudo chown -R "$$USER:$$USER" .venv

docker-build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) $(ROOT_DIR)

docker-train-gpu:
	docker run --rm -it --gpus all \
		-e UV_PROJECT_ENVIRONMENT=/tmp/uv-venv \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py

docker-smoke:
	docker run --rm -it --gpus all \
		-e UV_PROJECT_ENVIRONMENT=/tmp/uv-venv \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py --smoke-test

docker-train-gpu-build: docker-build docker-train-gpu

docker-smoke-build: docker-build docker-smoke

docker-train-cpu:
	docker run --rm -it \
		-e UV_PROJECT_ENVIRONMENT=/tmp/uv-venv \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py --config llm/train_config_cpu.yaml

docker-smoke-cpu:
	docker run --rm -it \
		-e UV_PROJECT_ENVIRONMENT=/tmp/uv-venv \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py --smoke-test --config llm/train_config_cpu.yaml

docker-train-cpu-build: docker-build docker-train-cpu

docker-smoke-cpu-build: docker-build docker-smoke-cpu

local-smoke:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/train.py --smoke-test
