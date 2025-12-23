from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SPEAKER_RE = re.compile(r"<S(\d+)>")


def load_speaker_map(path: Path) -> dict[str, str]:
    """Load the speaker map (username -> token) and invert it."""
    with path.open("r", encoding="utf-8") as f:
        speaker_map = json.load(f)
    return {token: username for username, token in speaker_map.items()}


def decode_text(text: str, token_to_user: dict[str, str]) -> str:
    """Replace speaker tokens with usernames and format lines as 'user: content'."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode speaker tokens to usernames")
    parser.add_argument(
        "--speaker-map",
        type=Path,
        default=Path("train_data/speaker_map.json"),
        help="Path to speaker_map.json",
    )
    parser.add_argument("--text", type=str, help="Raw model output with <S#> tokens")
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to a text file containing model output",
    )
    args = parser.parse_args()

    if args.text and args.file:
        raise ValueError("Use either --text or --file, not both.")
    if not args.text and not args.file:
        if sys.stdin.isatty():
            raise ValueError("Provide --text, --file, or pipe input via stdin.")
        text = sys.stdin.read()
    elif args.file:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = args.text

    token_to_user = load_speaker_map(args.speaker_map)
    print(decode_text(text, token_to_user))


if __name__ == "__main__":
    main()
