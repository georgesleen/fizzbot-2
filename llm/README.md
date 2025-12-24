# LLM
Training and inference pipeline for the fizzbot multi-speaker chat model.

This directory contains:
- Data preprocessing to build JSONL training examples.
- Training script with YAML config and optional validation split.
- Inference CLI that decodes speaker tokens to usernames.
- Docker support for GPU training.

## Design Notes (Why These Choices)
- **Speaker tokens (`<S0>`, `<S1>`, ...)**: the model learns both who speaks next and what they say as a single text generation task (no classifier).
- **Single-message targets**: simpler and stable for next-turn prediction; multi-turn generation happens at inference by running multiple generations.
- **`<EOT>` tokens**: optional explicit turn boundaries; useful once the model learns them, but decoding does not rely on `<EOT>`.
- **Context windows**: randomized length during preprocessing to improve robustness.
- **No channel mixing**: contexts never cross channel boundaries.
- **Train/val split**: added to validate loss trends and reduce overfitting.

## Project Layout
- `llm/gen_training_data.py`: build training JSONL + speaker map.
- `llm/train.py`: train a causal LM with masking on target only.
- `llm/run.py`: inference + decoding.
- `llm/decode.py`: decode raw model output (`<S#>` → `username:`).
- `llm/train_config.yaml`: GPU/QLoRA-focused config.
- `llm/train_config_cpu.yaml`: CPU-focused config.
- `llm/run_config.yaml`: inference defaults.
- `llm/Dockerfile`: GPU-ready training container.

## Data Pipeline
Input format (per channel):
```
{"username": "...", "content": "...", "timestamp": "..."}
```

Stages:
1) Normalize to `{username, content, timestamp}`.
2) Sort by timestamp per channel.
3) Build randomized context → target examples.
4) Map usernames → speaker tokens.
5) Append `<EOT>` to each message.
6) Write JSONL:
```
{"context": "...", "target": "..."}
```

Filters:
- Empty messages are dropped.
- URL-only and mention-only messages are dropped.

Generate data:
```
uv run llm/gen_training_data.py
```

Outputs:
- `train_data/training_examples.jsonl`
- `train_data/speaker_map.json`

## Training
Training is configured via YAML:
- `llm/train_config.yaml` (GPU/QLoRA)
- `llm/train_config_cpu.yaml` (CPU)

Run GPU training:
```
uv run llm/train.py --config llm/train_config.yaml
```

Run CPU training:
```
uv run llm/train.py --config llm/train_config_cpu.yaml
```

Validation:
- Use `data.val_split` (e.g. 0.02) or set `data.val_jsonl`.
- Eval runs every `training.eval_steps`.

Outputs:
```
llm/runs/<name>/<timestamp>/
```

## Inference
Latest run:
```
uv run llm/run.py --latest --speaker "<S0>" --content "hello everyone"
```

Specific checkpoint:
```
uv run llm/run.py --model-dir llm/runs/fizzbot_cpu/<run>/checkpoint-5000 \
  --speaker "<S0>" --content "hello everyone"
```

Multiple responses (runs generation N times and concatenates results):
```
uv run llm/run.py --latest --speaker "<S0>" --content "hello everyone" --turns 3
```

Common knobs:
- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--repetition-penalty`
- `--no-eos-stop`
- `--interactive`

## Decoding
Decode raw output:
```
uv run llm/decode.py --text "<S0> hello <EOT>\n<S1> hi <EOT>"
```

Pipe from inference:
```
uv run llm/run.py --latest --speaker "<S0>" --content "hi" | uv run llm/decode.py
```

## Makefile Targets
These are defined in the repo root `Makefile`:
- `make gen-training-data`
- `make local-smoke`
- `make test-latest`
- `make docker-build`
- `make docker-train-gpu`
- `make docker-smoke`
- `make docker-train-cpu`
- `make docker-smoke-cpu`
- `make docker-train-gpu-build`
- `make docker-smoke-build`
- `make docker-train-cpu-build`
- `make docker-smoke-cpu-build`
- `make fix-uv-cache`
- `make fix-venv-perms`

## Docker
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
- `Permission denied` from uv cache:
  - `make fix-uv-cache`
  - or set `UV_CACHE_DIR=.uv_cache`
- `run.py` can't find tokenizer in checkpoint:
  - set `tokenizer_model` in `llm/run_config.yaml` or pass `--tokenizer-model`
- Loss flat:
  - ensure data filters are on (empty/URL/mention-only removed)
  - use `val_split` to monitor eval loss
  - train longer and/or increase data size
