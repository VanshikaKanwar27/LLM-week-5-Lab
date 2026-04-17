from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_FILES = (
    "user_subset.json",
    "item_subset.json",
    "review_subset.json",
    "test_review_subset.json",
)


def convert_jsonl_to_array(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"Skipping {file_path.name}: file not found.")
        return False

    content = file_path.read_text(encoding="utf-8").lstrip()
    if not content:
        print(f"Skipping {file_path.name}: file is empty.")
        return False

    if content.startswith("["):
        print(f"{file_path.name} is already a JSON array.")
        return False

    rows = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))

    file_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Converted {file_path.name} to a JSON array with {len(rows)} objects.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AgentReview JSONL dataset files into JSON arrays.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional file paths to convert. Defaults to the files in the local data directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.paths:
        files = [Path(path).resolve() for path in args.paths]
    else:
        data_dir = Path(__file__).resolve().parent / "data"
        files = [data_dir / name for name in DEFAULT_FILES]

    converted = 0
    for file_path in files:
        converted += int(convert_jsonl_to_array(file_path))

    print(f"Completed data normalization. Files converted: {converted}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
