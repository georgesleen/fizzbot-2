IMAGE_NAME ?= fizzbot-llm
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKERFILE ?= $(ROOT_DIR)/llm/Dockerfile

.PHONY: help docker-build docker-train-gpu docker-smoke local-smoke gen-training-data

help:
	@echo "Targets:"
	@echo "  gen-training-data Generate training examples JSONL"
	@echo "  local-smoke       Run a tiny local smoke test"
	@echo "  docker-build      Build the training Docker image"
	@echo "  docker-train-gpu  Build + run GPU training in Docker"
	@echo "  docker-smoke      Build + run GPU smoke test in Docker"

gen-training-data:
	uv run llm/gen_training_data.py

docker-build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) $(ROOT_DIR)

docker-train-gpu: docker-build
	docker run --rm -it --gpus all \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py

docker-smoke: docker-build
	docker run --rm -it --gpus all \
		-v $(ROOT_DIR):/workspace \
		-w /workspace \
		$(IMAGE_NAME) \
		uv run llm/train.py --smoke-test

local-smoke:
	uv run llm/train.py --smoke-test
