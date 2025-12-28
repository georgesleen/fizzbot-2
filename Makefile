IMAGE_NAME ?= fizzbot-llm
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKERFILE ?= $(ROOT_DIR)/llm/Dockerfile

.PHONY: help docker-train-build docker-publish docker-train-gpu docker-smoke docker-train-cpu docker-smoke-cpu local-smoke test-local-smoke gen-training-data fizzbot fizzbot-cpu rsync-root

help:
	@echo "Targets:"
	@echo "  fizzbot                Run inference using fizzbot_mistral_7b in llm/runs/"
	@echo "  fizzbot-cpu            Run inference against latest fizzbot_cpu run"
	@echo "  gen-training-data      Generate training examples JSONL"
	@echo "  local-smoke            Run a tiny local smoke test"
	@echo "  test-local-smoke       Train + run the smoke model locally"
	@echo "  docker-publish         Build and push georgesleen/fizzbot:latest"
	@echo "  docker-train-build     Build the training Docker image"
	@echo "  docker-train-gpu       Run GPU training in Docker (no build)"
	@echo "  docker-train-cpu       Run CPU training in Docker (no build)"
	@echo "  docker-smoke           Run GPU smoke test in Docker (no build)"
	@echo "  docker-smoke-cpu       Run CPU smoke test in Docker (no build)"
	@echo "  rsync-root             Sync repo to personal server"

gen-training-data:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/gen_training_data.py

fizzbot:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --model-dir llm/runs/fizzbot_mistral_7b --tokenizer-model mistralai/Mistral-7B-v0.1 --decode --max-new-tokens 200 --temperature 0.9 --repetition-penalty 1.1 --interactive

fizzbot-cpu:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --runs-dir runs/fizzbot_cpu --latest --tokenizer-model mistralai/Mistral-7B-v0.1 --decode --max-new-tokens 400 --temperature 0.9 --repetition-penalty 1.1 --interactive

docker-train-build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) $(ROOT_DIR)

docker-publish:
	docker build -t georgesleen/fizzbot:latest .
	docker push georgesleen/fizzbot:latest

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

local-smoke:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/train.py --smoke-test

test-local-smoke:
	UV_CACHE_DIR=$(ROOT_DIR)/.uv_cache uv run llm/run.py --runs-dir runs/fizzbot --latest --tokenizer-model sshleifer/tiny-gpt2 --decode --max-new-tokens 400 --temperature 0.9 --repetition-penalty 1.1 --interactive

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
		--exclude "fizzbot" \
		--exclude "*.zip" \
		--exclude "data/*.py" \
		-e "ssh" \
		./ george-sleen@gs-server:/home/george-sleen/Documents/fizzbot
