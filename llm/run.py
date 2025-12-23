from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_model_dir(
    model_dir: Path | None,
    runs_dir: Path,
    latest: bool,
    device: str,
    cpu_model: str | None,
    gpu_model: str | None,
) -> str:
    """Resolve the model path or id, optionally selecting the latest run."""
    if model_dir is not None and latest:
        raise ValueError("Use either --model-dir or --latest, not both.")
    if model_dir is not None:
        return str(model_dir)
    if latest:
        if not runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No run directories found in: {runs_dir}")
        return str(max(candidates, key=lambda p: p.stat().st_mtime))

    if device == "cpu" and cpu_model:
        return cpu_model
    if device == "cuda" and gpu_model:
        return gpu_model

    raise ValueError("Provide --model-dir/--latest or set a device-specific model.")


def main() -> None:
    """CLI entrypoint for running a trained fizzbot model."""
    parser = argparse.ArgumentParser(description="Run a trained fizzbot model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "run_config.yaml",
        help="Path to YAML config for defaults",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Path to a trained model directory",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent run under --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "fizzbot",
        help="Base directory containing timestamped runs.",
    )
    parser.add_argument(
        "--cpu-model",
        type=str,
        default=None,
        help="Model name or path to use when --device=cpu.",
    )
    parser.add_argument(
        "--gpu-model",
        type=str,
        default=None,
        help="Model name or path to use when --device=cuda.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection (auto uses CUDA if available).",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        required=True,
        help="Speaker token for the prompt, e.g. <S0>",
    )
    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="Prompt content that follows the speaker token.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config).get("run", {})

    runs_dir = Path(
        args.runs_dir
        if args.runs_dir is not None
        else cfg.get("runs_dir", "runs/fizzbot")
    )
    if not runs_dir.is_absolute():
        runs_dir = Path(__file__).resolve().parent / runs_dir

    cpu_model = args.cpu_model if args.cpu_model is not None else cfg.get("cpu_model")
    gpu_model = args.gpu_model if args.gpu_model is not None else cfg.get("gpu_model")
    device = args.device if args.device != "auto" else cfg.get("device", "auto")

    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 120))
    temperature = args.temperature or float(cfg.get("temperature", 0.8))
    top_p = args.top_p or float(cfg.get("top_p", 0.9))
    seed = args.seed or int(cfg.get("seed", 0))

    if seed:
        torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = _resolve_model_dir(
        args.model_dir, runs_dir, args.latest, device, cpu_model, gpu_model
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    prompt = f"{args.speaker} {args.content} <EOT>"
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
