from __future__ import annotations

from pathlib import Path

from fix_data import convert_jsonl_to_array


def main() -> int:
    data_dir = Path(__file__).resolve().parent / "data"
    converted = 0

    for file_path in sorted(data_dir.glob("*_subset.json")):
        converted += int(convert_jsonl_to_array(file_path))

    print(f"Done fixing JSON files. Files converted: {converted}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
