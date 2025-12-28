from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import sys
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SPEAKER_RE = re.compile(r"<S\d+>")
END_OF_CONTENT_MARKER = "EOF"


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load a YAML config file into a dictionary.
    """
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
    """
    Resolve the model path or id, optionally selecting the latest run.
    """
    if model_dir is not None and latest:
        raise ValueError("Use either --model-dir or --latest, not both.")

    if model_dir is not None:
        candidate = model_dir
        if not candidate.exists() and not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent / candidate
        if not candidate.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                f"Tried {candidate} as a fallback."
            )
        return str(candidate.resolve())

    if latest:
        if not runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No run directories found in: {runs_dir}")
        latest_run = max(candidates, key=lambda p: p.stat().st_mtime)
        config_path = latest_run / "config.json"
        if config_path.exists():
            return str(latest_run)
        # Fall back to newest checkpoint with a config.json
        checkpoints = [
            p
            for p in latest_run.iterdir()
            if p.is_dir()
            and (p / "config.json").exists()
            and p.name.startswith("checkpoint-")
        ]
        if checkpoints:
            return str(max(checkpoints, key=lambda p: p.stat().st_mtime))
        return str(latest_run)

    if device == "cpu" and cpu_model:
        return cpu_model
    if device == "cuda" and gpu_model:
        return gpu_model

    raise ValueError("Provide --model-dir/--latest or set a device-specific model.")


def _resolve_tokenizer_dir(model_dir: Path) -> Path:
    """
    Use checkpoint tokenizer if present; otherwise fall back to parent run dir.
    """
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    if any((model_dir / name).exists() for name in candidates):
        return model_dir
    if any((model_dir.parent / name).exists() for name in candidates):
        return model_dir.parent
    return model_dir


def _find_tokenizer_dir(runs_dir: Path) -> Path | None:
    """
    Find the newest run directory that contains tokenizer files.
    """
    if not runs_dir.exists():
        return None
    candidates = [
        p
        for p in runs_dir.iterdir()
        if p.is_dir()
        and any(
            (p / name).exists()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_speaker_map(path: Path) -> dict[str, str]:
    """
    Load the speaker map (username -> token) and invert it.
    """
    with path.open("r", encoding="utf-8") as f:
        speaker_map = json.load(f)
    return {token: username for username, token in speaker_map.items()}


def decode_text(text: str, token_to_user: dict[str, str]) -> str:
    """
    Replace speaker tokens with usernames and format lines as 'user: content'.
    """
    lines = []
    segments = re.split(r"(<S\d+>)", text)
    current_token = None
    buffer = []

    for segment in segments:
        if not segment:
            continue
        if SPEAKER_RE.fullmatch(segment):
            if current_token is not None:
                content = "".join(buffer).replace("<EOT>", "").strip()
                if content:
                    username = token_to_user.get(current_token, current_token)
                    lines.append(f"{username}: {content}".rstrip())
            current_token = segment
            buffer = []
        else:
            buffer.append(segment)

    if current_token is not None:
        content = "".join(buffer).replace("<EOT>", "").strip()
        if content:
            username = token_to_user.get(current_token, current_token)
            lines.append(f"{username}: {content}".rstrip())

    return "\n".join(lines)


def count_speaker_tokens(text: str) -> int:
    """
    Count speaker tokens in text.
    """
    return len(SPEAKER_RE.findall(text))


def truncate_to_turns(base_text: str, full_text: str, turns: int) -> str:
    """
    Truncate generated text to a fixed number of new speaker turns.
    """
    if turns <= 0:
        return full_text
    if not full_text.startswith(base_text):
        return full_text
    gen_text = full_text[len(base_text) :]
    segments = re.split(r"(<S\d+>)", gen_text)
    new_turns = 0
    out = []
    for seg in segments:
        if not seg:
            continue
        out.append(seg)
        if SPEAKER_RE.fullmatch(seg):
            new_turns += 1
            if new_turns >= turns:
                break
    return base_text + "".join(out)


def trim_decoded_turns(
    base_text: str, full_text: str, turns: int, token_to_user: dict[str, str]
) -> str:
    """
    Return only the newly generated turns after the prompt.
    """
    if turns <= 0:
        return decode_text(full_text, token_to_user)
    base_lines = decode_text(base_text, token_to_user).splitlines()
    full_lines = decode_text(full_text, token_to_user).splitlines()
    new_lines = full_lines[len(base_lines) : len(base_lines) + turns]
    if not new_lines:
        return "\n".join(full_lines)
    return "\n".join(new_lines)


def _extract_first_new_line(
    base_text: str, full_text: str, token_to_user: dict[str, str]
) -> str:
    """
    Extract the first new decoded line after the prompt.
    """
    base_lines = decode_text(base_text, token_to_user).splitlines()
    full_lines = decode_text(full_text, token_to_user).splitlines()
    if len(full_lines) > len(base_lines):
        return full_lines[len(base_lines)]
    if full_lines:
        return full_lines[-1]
    return ""


def _extract_first_new_turn(
    base_text: str, full_text: str, token_to_user: dict[str, str]
) -> str:
    """
    Extract the first newly generated speaker turn after the prompt.
    """
    gen_text = full_text[len(base_text) :] if full_text.startswith(base_text) else full_text
    match = SPEAKER_RE.search(gen_text)
    if not match:
        return ""
    token = match.group(0)
    rest = gen_text[match.end() :]
    next_match = SPEAKER_RE.search(rest)
    content = rest[: next_match.start()] if next_match else rest
    content = content.replace("<EOT>", "").strip()
    if not content:
        return ""
    username = token_to_user.get(token, token)
    return f"{content} (most likely to be said by: {username})".rstrip()


def _model_max_length(model) -> int:
    """
    Get model max position length for safe truncation.
    """
    if hasattr(model.config, "max_position_embeddings"):
        return int(model.config.max_position_embeddings)
    if hasattr(model.config, "n_positions"):
        return int(model.config.n_positions)
    return 1024


def _prompt_input(prompt: str) -> str:
    """
    Prompt on stderr to keep stdout clean for bot parsing.
    """
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return input()


def main() -> None:
    """
    CLI entrypoint for running a trained fizzbot model.
    """
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
        "--speaker-map",
        type=Path,
        default=Path("train_data/speaker_map.json"),
        help="Path to speaker_map.json for decoding output.",
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
        "--base-model",
        type=str,
        default=None,
        help="Base model name/path to use when loading LoRA adapters.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection (auto uses CUDA if available).",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode speaker tokens into usernames using speaker_map.json.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=0,
        help="Number of independent generations to run (0 = disabled).",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="Tokenizer model id/path to use when missing from checkpoint.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        required=False,  # CHANGED: Now optional (checked manually later)
        help="Speaker token for the prompt, e.g. <S0>",
    )
    parser.add_argument(
        "--content",
        type=str,
        required=False,  # CHANGED: Now optional
        help="Prompt content that follows the speaker token.",
    )
    parser.add_argument(
        "--interactive",  # NEW FLAG
        action="store_true",
        help="Enter interactive mode to prompt the model repeatedly.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-eos-stop",
        action="store_true",
        help="Do not stop generation on EOS; use max_new_tokens instead.",
    )
    args = parser.parse_args()

    # MANUAL VALIDATION: If not interactive, we still need speaker + content
    if not args.interactive and (not args.speaker or not args.content):
        parser.error(
            "the following arguments are required: --speaker, --content (unless using --interactive)"
        )

    cfg = load_config(args.config).get("run", {})
    config_dir = args.config.resolve().parent

    runs_dir = Path(
        args.runs_dir
        if args.runs_dir is not None
        else cfg.get("runs_dir", "runs/fizzbot")
    )
    if not runs_dir.is_absolute():
        runs_dir = config_dir / runs_dir

    cpu_model = args.cpu_model if args.cpu_model is not None else cfg.get("cpu_model")
    gpu_model = args.gpu_model if args.gpu_model is not None else cfg.get("gpu_model")
    device = args.device if args.device != "auto" else cfg.get("device", "auto")
    tokenizer_model = (
        args.tokenizer_model
        if args.tokenizer_model is not None
        else cfg.get("tokenizer_model")
    )
    base_model_override = (
        args.base_model if args.base_model is not None else cfg.get("base_model")
    )

    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 120))
    temperature = args.temperature or float(cfg.get("temperature", 0.8))
    top_p = args.top_p or float(cfg.get("top_p", 0.9))
    repetition_penalty = args.repetition_penalty or float(
        cfg.get("repetition_penalty", 1.0)
    )
    seed = args.seed or int(cfg.get("seed", 0))
    no_eos_stop = args.no_eos_stop or bool(cfg.get("no_eos_stop", False))
    turns = args.turns or int(cfg.get("turns", 0))
    if turns <= 0 and args.turns:
        raise ValueError("--turns must be > 0")
    decode_output = args.decode or bool(cfg.get("decode", False))
    speaker_map_path = Path(
        args.speaker_map
        if args.speaker_map is not None
        else cfg.get("speaker_map", "train_data/speaker_map.json")
    )
    if not speaker_map_path.is_absolute():
        speaker_map_path = config_dir / speaker_map_path

    if seed:
        torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = _resolve_model_dir(
        args.model_dir, runs_dir, args.latest, device, cpu_model, gpu_model
    )
    model_path = Path(model_id)
    local_only = model_path.exists()
    adapter_base_model = None
    adapter_config_path = model_path / "adapter_config.json"
    if local_only and adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        adapter_base_model = adapter_config.get("base_model_name_or_path")
    tokenizer_dir = _resolve_tokenizer_dir(model_path) if local_only else model_path
    if local_only and tokenizer_dir == model_path:
        if tokenizer_model:
            tokenizer_dir = tokenizer_model
        else:
            inferred = _find_tokenizer_dir(runs_dir)
            if inferred:
                tokenizer_dir = inferred
            else:
                fallback = (
                    base_model_override or adapter_base_model or cpu_model or gpu_model
                )
                if not fallback:
                    raise ValueError(
                        "Tokenizer files not found in checkpoint; provide --tokenizer-model."
                    )
                tokenizer_dir = fallback

    # Check for local tokenizer existence to avoid offline errors with base models
    is_local_tokenizer = Path(str(tokenizer_dir)).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=is_local_tokenizer
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    if local_only and adapter_base_model:
        base_model_name = base_model_override or adapter_base_model
        if not base_model_name:
            raise ValueError(
                "Adapter checkpoint found but no base model specified. "
                "Use --base-model or set run.base_model in run_config.yaml."
            )
        base_model_local = Path(str(base_model_name)).exists()
        print(f"Loading base model from {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            local_files_only=base_model_local,
            quantization_config=bnb_config,
            device_map="auto",
        )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "LoRA adapter detected but peft is not installed."
            ) from exc
        print(f"Loading adapter from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=local_only,
            quantization_config=bnb_config,
            device_map="auto",
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Pre-load map if needed for decoding
    token_to_user = load_speaker_map(speaker_map_path) if decode_output else {}

    # === INTERACTIVE LOOP ===
    print("Ready!")
    while True:
        if args.interactive:
            try:
                print("\n--- New Prompt ---")
                current_speaker = _prompt_input("Speaker (e.g. <S0>): ").strip()
                if current_speaker.lower() in ["exit", "quit"]:
                    break
                if not current_speaker:
                    continue

                content_lines = []
                while True:
                    line = _prompt_input(
                        "Content line (end message by sending EOF): "
                    )
                    if line.strip() == END_OF_CONTENT_MARKER:
                        break
                    content_lines.append(line)

                current_content = "\n".join(content_lines)

                if current_content.lower() in ["exit", "quit"]:
                    break

            except KeyboardInterrupt:
                print("\nExiting...")
                break
        else:
            current_speaker = args.speaker
            current_content = args.content

        prompt = f"{current_speaker} {current_content} <EOT>"
        inputs = tokenizer(prompt, return_tensors="pt")

        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            eos_token_id = None if no_eos_stop else model.config.eos_token_id

            if turns > 0:
                base_text = tokenizer.decode(
                    inputs["input_ids"][0], skip_special_tokens=False
                )

                print(f"\n{current_speaker}: {current_content}")

                for i in range(turns):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    full_text = tokenizer.decode(out[0], skip_special_tokens=False)

                    if decode_output:
                        response = _extract_first_new_line(
                            base_text, full_text, token_to_user
                        )
                    else:
                        truncated = truncate_to_turns(base_text, full_text, 1)
                        if truncated.startswith(base_text):
                            response = truncated[len(base_text) :]
                        else:
                            response = truncated

                    clean_response = response.strip()
                    if clean_response:
                        print(clean_response)

                    # Update history
                    inputs = {
                        "input_ids": out,
                        "attention_mask": torch.ones_like(out),
                    }
                    base_text = full_text

            else:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                output_text = tokenizer.decode(out[0], skip_special_tokens=False)

                if decode_output:
                    # If single turn, clean it up
                    base_text = tokenizer.decode(
                        inputs["input_ids"][0], skip_special_tokens=False
                    )
                    if args.interactive:
                        final_text = _extract_first_new_turn(
                            base_text, output_text, token_to_user
                        )
                    else:
                        final_text = trim_decoded_turns(
                            base_text, output_text, turns, token_to_user
                        )
                    print("\n" + final_text)
                else:
                    print("\n" + output_text)

        # If not interactive, run once and exit
        if not args.interactive:
            break


if __name__ == "__main__":
    main()
