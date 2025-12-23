from pathlib import Path
import json
import random

src_root = Path("data/data_cleaned")
dst_root = Path("train_data")
examples_path = dst_root / "training_examples.jsonl"
speaker_map_path = dst_root / "speaker_map.json"

MIN_CTX = 1
MAX_CTX = 8
END_TOKEN = "<EOT>"


def normalize_data(dataset: list) -> list:
    """
    Takes JSON downloads from Discrub and cleans them to contain ONLY
    username, content, and timestamp.
    """
    clean = []

    for msg in dataset:
        clean_msg = {}

        username = msg["author"]["username"]
        content = msg["content"]
        timestamp = msg["timestamp"]

        clean_msg["username"] = username
        clean_msg["content"] = content
        clean_msg["timestamp"] = timestamp

        clean.append(clean_msg)

    clean.sort(key=lambda m: m["timestamp"])
    return clean


def _speaker_token(username: str, speaker_map: dict) -> str:
    if username not in speaker_map:
        speaker_map[username] = f"<S{len(speaker_map)}>"
    return speaker_map[username]


def build_training_examples(
    dataset: list,
    speaker_map: dict,
    min_ctx: int = MIN_CTX,
    max_ctx: int = MAX_CTX,
) -> list:
    """
    Builds randomized context -> target examples from a sorted message list.
    """
    examples = []

    for idx in range(1, len(dataset)):
        ctx_len = random.randint(min_ctx, max_ctx)
        start = max(0, idx - ctx_len)

        context_msgs = dataset[start:idx]
        target_msg = dataset[idx]

        context = "\n".join(
            f"{_speaker_token(m['username'], speaker_map)} {m['content']} {END_TOKEN}"
            for m in context_msgs
        )
        target = (
            f"{_speaker_token(target_msg['username'], speaker_map)} "
            f"{target_msg['content']} {END_TOKEN}"
        )

        examples.append({"context": context, "target": target})

    return examples


def json_to_jsonl(dataset: list) -> str:
    """
    Converts a list of JSON objects into JSONL text.
    """
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in dataset)


def main() -> None:
    all_examples = []
    speaker_map = {}

    for src_path in src_root.rglob("*.json"):
        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            clean_data = normalize_data(data)

        all_examples.extend(build_training_examples(clean_data, speaker_map))

    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with examples_path.open("w", encoding="utf-8") as f:
        f.write(json_to_jsonl(all_examples))
    with speaker_map_path.open("w", encoding="utf-8") as f:
        json.dump(speaker_map, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
