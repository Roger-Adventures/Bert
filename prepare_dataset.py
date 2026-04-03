from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


SEED = 42
TARGETS = {
    "normal": 800,
    "abuse": 800,
    "sexual": 800,
    "ad": 800,
}

DATA_DIR = Path("data")
CHINESE_HARM_DATASET = "zjunlp/ChineseHarm-bench"
CHINESE_HARM_LABEL_MAP = {
    "不违规": "normal",
    "谩骂引战": "abuse",
    "低俗色情": "sexual",
    "黑产广告": "ad",
}
COLD_BASE_URL = "https://raw.githubusercontent.com/thu-coai/COLDataset/main/COLDataset"

TEXT_MIN_LEN = 4
TEXT_MAX_LEN = 160


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\u3000", " ").replace("\n", " ").split()).strip()


def deduplicate_texts(texts: list[str]) -> list[str]:
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in texts:
        clean_text = normalize_text(text)
        if not clean_text or clean_text in seen:
            continue
        seen.add(clean_text)
        unique_texts.append(clean_text)
    return unique_texts


def sample_texts(texts: list[str], target: int, seed_offset: int) -> list[str]:
    candidates = deduplicate_texts(texts)
    if len(candidates) < target:
        raise ValueError(f"Not enough texts to sample {target} items, only found {len(candidates)}")

    random.Random(SEED + seed_offset).shuffle(candidates)
    return candidates[:target]


def load_chinese_harm_texts() -> tuple[dict[str, list[str]], dict[str, int]]:
    dataset = load_dataset(CHINESE_HARM_DATASET, split="train")
    grouped: dict[str, list[str]] = {label: [] for label in TARGETS}
    raw_counts: dict[str, int] = {label: 0 for label in TARGETS}

    for item in dataset:
        raw_label = item["标签"]
        mapped_label = CHINESE_HARM_LABEL_MAP.get(raw_label)
        if mapped_label is None:
            continue

        text = normalize_text(item["文本"])
        if not (TEXT_MIN_LEN <= len(text) <= TEXT_MAX_LEN):
            continue

        grouped[mapped_label].append(text)
        raw_counts[mapped_label] += 1

    return {label: deduplicate_texts(texts) for label, texts in grouped.items()}, raw_counts


def load_cold_abuse_texts() -> list[str]:
    train_frame = pd.read_csv(f"{COLD_BASE_URL}/train.csv")
    dev_frame = pd.read_csv(f"{COLD_BASE_URL}/dev.csv")
    frame = pd.concat([train_frame, dev_frame], ignore_index=True)
    offensive = frame[frame["label"] == 1]["TEXT"].astype(str).tolist()

    cleaned = []
    for text in offensive:
        clean_text = normalize_text(text)
        if TEXT_MIN_LEN <= len(clean_text) <= TEXT_MAX_LEN:
            cleaned.append(clean_text)

    return deduplicate_texts(cleaned)


def build_dataset() -> tuple[pd.DataFrame, dict[str, object]]:
    chinese_harm_texts, chinese_harm_raw_counts = load_chinese_harm_texts()
    selected_by_label: dict[str, list[str]] = {}
    source_notes: dict[str, str] = {}

    for index, label in enumerate(TARGETS):
        candidates = list(chinese_harm_texts[label])
        source_notes[label] = f"ChineseHarm-bench / {label}"

        if label == "abuse" and len(candidates) < TARGETS[label]:
            candidates.extend(load_cold_abuse_texts())
            source_notes[label] = "ChineseHarm-bench / abuse + COLDataset offensive fallback"

        selected_by_label[label] = sample_texts(candidates, TARGETS[label], seed_offset=index * 17)

    rows = []
    for label, texts in selected_by_label.items():
        rows.extend({"text": text, "label": label} for text in texts)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    frame = frame.sample(frac=1, random_state=SEED).reset_index(drop=True)

    summary = {
        "total": int(len(frame)),
        "label_distribution": frame["label"].value_counts().sort_index().to_dict(),
        "targets": TARGETS,
        "sources": {
            "primary_dataset": CHINESE_HARM_DATASET,
            "label_mapping": {
                "不违规": "normal",
                "谩骂引战": "abuse",
                "低俗色情": "sexual",
                "黑产广告": "ad",
            },
            "ignored_categories": ["博彩", "欺诈"],
            "fallback_dataset": {
                "abuse": "COLDataset offensive samples",
            },
            "selected_sources": source_notes,
        },
        "primary_raw_counts": chinese_harm_raw_counts,
    }
    return frame, summary


def split_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.30,
        random_state=SEED,
        stratify=frame["label"],
    )
    dev_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_frame["label"],
    )
    return train_frame.reset_index(drop=True), dev_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def write_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame, summary = build_dataset()
    train_frame, dev_frame, test_frame = split_dataset(frame)

    train_frame.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8")
    dev_frame.to_csv(DATA_DIR / "dev.csv", index=False, encoding="utf-8")
    test_frame.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8")

    summary.update(
        {
            "train": int(len(train_frame)),
            "dev": int(len(dev_frame)),
            "test": int(len(test_frame)),
        }
    )

    with (DATA_DIR / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_dataset()
