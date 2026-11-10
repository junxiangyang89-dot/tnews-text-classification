"""Prepare TNEWS data from the current Hugging Face repository."""

import json

from datasets import load_dataset
from src.download_data import DATA_DIR, build_label_map, save_split


def main():
    dataset = load_dataset("clue/clue", "tnews")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    label_map = build_label_map(dataset["train"].features)
    label_path = DATA_DIR / "label_map.json"
    label_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    for split, filename in (("train", "train"), ("validation", "val"), ("test", "test")):
        save_split(dataset[split], DATA_DIR / f"{filename}.json", filename)


if __name__ == "__main__":
    main()
