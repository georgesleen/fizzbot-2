from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_model_dir(model_dir: Path | None, runs_dir: Path, latest: bool) -> Path:
    """Resolve the model directory, optionally selecting the latest run."""
    if model_dir is not None and latest:
        raise ValueError("Use either --model-dir or --latest, not both.")
    if model_dir is not None:
        return model_dir
    if not latest:
        raise ValueError("Provide --model-dir or use --latest.")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in: {runs_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """CLI entrypoint for running a trained fizzbot model."""
    parser = argparse.ArgumentParser(description="Run a trained fizzbot model")
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

    if args.seed:
        torch.manual_seed(args.seed)

    model_dir = _resolve_model_dir(args.model_dir, args.runs_dir, args.latest)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    prompt = f"{args.speaker} {args.content}"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
