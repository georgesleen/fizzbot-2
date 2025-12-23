from pathlib import Path
import json

src_root = Path("data/data_raw")
dst_root = Path("train_data")


def make_minimal_dataset(dataset: list) -> list:
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

    return clean


def json_to_jsonl(dataset: list) -> str:
    """
    Converts a list of JSON objects into JSONL text.
    """
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in dataset)


def main() -> None:
    for src_path in src_root.rglob("*.json"):
        rel_path = src_path.relative_to(src_root)
        dst_path = (dst_root / rel_path).with_suffix(".jsonl")
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            clean_data = make_minimal_dataset(data)

        with dst_path.open("w", encoding="utf-8") as f:
            f.write(json_to_jsonl(clean_data))


if __name__ == "__main__":
    main()
