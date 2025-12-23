IMAGE_NAME ?= fizzbot-llm
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKERFILE ?= $(ROOT_DIR)/llm/Dockerfile

.PHONY: help docker-build docker-train-gpu docker-smoke local-smoke

help:
	@echo "Targets:"
	@echo "  local-smoke       Run a tiny local smoke test"
	@echo "  docker-build      Build the training Docker image"
	@echo "  docker-train-gpu  Build + run GPU training in Docker"
	@echo "  docker-smoke      Build + run GPU smoke test in Docker"

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
