IMAGE_NAME ?= fizzbot-llm
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKERFILE ?= $(ROOT_DIR)/llm/Dockerfile

.PHONY: help docker-build docker-train-gpu docker-train-gpu-build docker-smoke docker-smoke-build docker-train-cpu docker-train-cpu-build docker-smoke-cpu docker-smoke-cpu-build local-smoke gen-training-data

help:
	@echo "Targets:"
	@echo "  gen-training-data Generate training examples JSONL"
	@echo "  local-smoke       Run a tiny local smoke test"
	@echo "  docker-build      Build the training Docker image"
	@echo "  docker-train-gpu  Run GPU training in Docker (no build)"
	@echo "  docker-smoke      Run GPU smoke test in Docker (no build)"
	@echo "  docker-train-gpu-build  Build + run GPU training in Docker"
	@echo "  docker-smoke-build      Build + run GPU smoke test in Docker"
	@echo "  docker-train-cpu  Run CPU training in Docker (no build)"
	@echo "  docker-smoke-cpu  Run CPU smoke test in Docker (no build)"
	@echo "  docker-train-cpu-build  Build + run CPU training in Docker"
	@echo "  docker-smoke-cpu-build  Build + run CPU smoke test in Docker"

gen-training-data:
	uv run llm/gen_training_data.py

docker-build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) $(ROOT_DIR)

docker-train-gpu:
	docker run --rm -it --gpus all \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py

docker-smoke:
	docker run --rm -it --gpus all \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py --smoke-test

docker-train-gpu-build: docker-build docker-train-gpu

docker-smoke-build: docker-build docker-smoke

docker-train-cpu:
	docker run --rm -it \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py

docker-smoke-cpu:
	docker run --rm -it \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py --smoke-test

docker-train-cpu-build: docker-build docker-train-cpu

docker-smoke-cpu-build: docker-build docker-smoke-cpu

local-smoke:
	uv run llm/train.py --smoke-test
