IMAGE_NAME ?= fizzbot-llm
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKERFILE ?= $(ROOT_DIR)/llm/Dockerfile

.PHONY: help docker-build docker-train-gpu docker-smoke docker-train-cpu docker-smoke-cpu local-smoke gen-training-data test-latest fizzbot fizzbot-cpu fizzbot-cpu-once rsync-root

help:
	@echo "Targets:"
	@echo "  fizzbot                Run inference using fizzbot_mistral_7b in llm/runs/"
	@echo "  fizzbot-cpu            Run inference against latest fizzbot_cpu run"
	@echo "  fizzbot-cpu-once       Run one CPU inference (SPEAKER, CONTENT)"
	@echo "  gen-training-data      Generate training examples JSONL"
	@echo "  docker-build           Build the training Docker image"
	@echo "  docker-train-gpu       Run GPU training in Docker (no build)"
	@echo "  docker-train-gpu-build Build and run GPU training in Docker"
	@echo "  docker-smoke           Run GPU smoke test in Docker (no build)"
	@echo "  docker-smoke-build     Build and run GPU smoke test in Docker"
	@echo "  test-latest            Run inference against latest trained model"
	@echo "  local-smoke            Run a tiny local smoke test"
	@echo "  docker-train-cpu       Run CPU training in Docker (no build)"
	@echo "  docker-smoke-cpu       Run CPU smoke test in Docker (no build)"
	@echo "  docker-train-cpu-build Build and run CPU training in Docker"
	@echo "  docker-smoke-cpu-build Build and run CPU smoke test in Docker"
	@echo "  rsync-root             Sync repo to root@74.2.96.43:/workspace"

gen-training-data:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/gen_training_data.py

test-latest:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --latest --tokenizer-model mistralai/Mistral-7B-v0.1 --decode --max-new-tokens 400 --temperature 0.9 --repetition-penalty 1.1 --interactive

fizzbot:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --model-dir llm/runs/fizzbot_mistral_7b --tokenizer-model mistralai/Mistral-7B-v0.1 --decode --max-new-tokens 200 --temperature 0.9 --repetition-penalty 1.1 --interactive

fizzbot-cpu:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --runs-dir runs/fizzbot_cpu --latest --tokenizer-model mistralai/Mistral-7B-v0.1 --decode --max-new-tokens 400 --temperature 0.9 --repetition-penalty 1.1 --interactive

fizzbot-cpu-once:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --runs-dir runs/fizzbot_cpu --latest --decode --max-new-tokens 400 --temperature 0.9 --repetition-penalty 1.1 --speaker "$(SPEAKER)" --content "$(CONTENT)"

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

rsync-root:
	rsync -avz \
		--no-owner \
		--no-group \
		--no-perms \
		--exclude ".venv/" \
		--exclude ".uv_cache/" \
		--exclude "__pycache__/" \
		--exclude "target/" \
		--exclude ".git/" \
		--exclude "fizzbot_cpu/" \
		--exclude "*.zip" \
		--exclude "data/*.py" \
		-e "ssh" \
		./ sleen@100.79.76.20:/home/sleen/workspace
