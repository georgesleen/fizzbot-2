# fizzbot-2

## Quick Start
1) Install deps:
```
uv sync
```

2) If you have the pretrained model zip (`fizzbot_mistral_7b.zip`):
   - Put it at `llm/runs/fizzbot_mistral_7b.zip`
   - Unzip to `llm/runs/fizzbot_mistral_7b/`

3) Run fizzbot:
```
make fizzbot
```

4) For training data from Discrub, put raw exports in:
```
data/data_cleaned/
```

Multi-speaker chat LLM training and inference pipeline for Discord-style data.

This repo provides:
- A data pipeline to build JSONL training examples.
- A training script with YAML config and optional eval split.
- A CLI for inference and decoding speaker tokens.
- Docker and Makefile helpers for local and cloud runs.

## Requirements
- Python 3.10+ (recommended)
- `uv` for dependency management
- Optional: Docker + NVIDIA GPU for accelerated training

## Quick Start (Local CPU)
1) Install dependencies:
```
uv sync
```

2) Generate training data:
```
make gen-training-data
```

3) Smoke test training (tiny model):
```
make local-smoke
```

4) Run inference on the latest model:
```
make test-latest
```

## Project Layout
- `llm/gen_training_data.py` generates JSONL training examples.
- `llm/train.py` trains a causal LM from JSONL with YAML config.
- `llm/run.py` runs inference and decodes speaker tokens.
- `llm/decode.py` decodes raw model output to `username: content`.
- `llm/train_config.yaml` GPU/QLoRA-oriented config.
- `llm/train_config_cpu.yaml` CPU-friendly config.
- `llm/run_config.yaml` inference defaults.
- `train_data/` training outputs (generated).

## Data Pipeline
Input format: per-channel JSON or JSONL with:
```
{"username": "...", "content": "...", "timestamp": "..."}
```

Raw Discrub exports go in:
```
data/data_cleaned/
```
Place one or more `*.json` files there (Discrub channel exports) before running
`make gen-training-data` or `uv run llm/gen_training_data.py`.

Pipeline stages:
1) Normalize to `{username, content, timestamp}`.
2) Sort by timestamp (per channel).
3) Build `(context -> target)` examples.
4) Replace usernames with speaker tokens `<S0>`, `<S1>`, ...
5) Append `<EOT>` to each message.
6) Write final JSONL:
```
{"context": "...", "target": "..."}
```

Notes:
- Context never mixes channels.
- Targets are single-message by default.
- Empty/URL-only/mention-only messages are filtered.
- `speaker_map.json` maps usernames to speaker tokens.

Generate examples:
```
uv run llm/gen_training_data.py
```

Outputs:
- `train_data/training_examples.jsonl`
- `train_data/speaker_map.json`

## Training
Training is controlled by `llm/train_config.yaml` (GPU) or
`llm/train_config_cpu.yaml` (CPU).

Key fields:
- `model.name_or_path`
- `quantization.mode` (none | 4bit | 8bit)
- `data.train_jsonl`, `data.val_split`
- `training.*` (batch size, lr, eval steps, etc.)

Run training:
```
uv run llm/train.py --config llm/train_config.yaml
```

Run CPU training:
```
uv run llm/train.py --config llm/train_config_cpu.yaml
```

Training outputs go to:
```
llm/runs/<run_name>/<timestamp>/
```

## Inference
Run the latest model:
```
uv run llm/run.py --latest --speaker "<S0>" --content "hello everyone"
```

Run the "fizzbot" preset (expects a model at `llm/runs/fizzbot_mistral_7b`):
```
make fizzbot
```
If you trained a model elsewhere, copy or symlink it to:
```
llm/runs/fizzbot_mistral_7b
```

Run a specific checkpoint:
```
uv run llm/run.py --model-dir llm/runs/fizzbot_cpu/20251223_090951/checkpoint-5000 \
  --speaker "<S0>" --content "hello everyone"
```

Generate N independent responses:
```
uv run llm/run.py --latest --speaker "<S0>" --content "hello everyone" --turns 3
```

Common generation knobs:
- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--repetition-penalty`
- `--no-eos-stop`

## Decoding Speaker Tokens
Decode raw output:
```
uv run llm/decode.py --text "<S0> hello <EOT>\n<S1> hi <EOT>"
```

Pipe from `run.py`:
```
uv run llm/run.py --latest --speaker "<S0>" --content "hi" | uv run llm/decode.py
```

## Makefile Targets
```
make help
```

Common targets:
- `make gen-training-data`
- `make local-smoke`
- `make test-latest`
- `make docker-build`
- `make docker-train-gpu`
- `make docker-smoke`
- `make docker-train-cpu`
- `make docker-smoke-cpu`
- `make fix-uv-cache`
- `make fix-venv-perms`

## Docker
Root Dockerfile (runs the Discord bot):
```
docker build -t fizzbot -f Dockerfile .
docker run --rm -it \
  -e DISCORD_TOKEN="your_token_here" \
  -v "$PWD/llm/runs":/workspace/llm/runs \
  -v "$PWD/llm/train_data":/workspace/llm/train_data \
  fizzbot
```

Docker Compose (recommended):
```
export DISCORD_TOKEN="your_token_here"
docker compose up --build
```

Notes:
- The bot calls `make fizzbot`, which expects `llm/runs` and `llm/train_data`.
- `discord-bot/.env` is ignored in the image; pass `DISCORD_TOKEN` or mount a `.env`.

LLM Dockerfile (GPU training):
Build:
```
docker build -t fizzbot-llm -f llm/Dockerfile .
```

Train (GPU):
```
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace fizzbot-llm \
  uv run llm/train.py
```

Train (CPU):
```
docker run --rm -it -v "$PWD":/workspace -w /workspace fizzbot-llm \
  uv run llm/train.py --config llm/train_config_cpu.yaml
```

## Troubleshooting
- `Permission denied` from `uv` cache:
  - `make fix-uv-cache`
  - or set `UV_CACHE_DIR=.uv_cache` for local runs.
- If `run.py` says a checkpoint is missing tokenizer files:
  - set `tokenizer_model` in `llm/run_config.yaml` or pass `--tokenizer-model`.
- If loss is flat:
  - ensure empty targets are filtered
  - add `val_split` to monitor eval loss
  - train longer and/or increase data size
