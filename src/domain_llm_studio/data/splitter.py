"""Stratified train / dev / test splitter with fixed random seed."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_split(
    samples: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_key: str = "task",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test with stratification by task type."""
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s.get(stratify_key, "unknown")].append(s)

    train, dev, test = [], [], []

    for key in sorted(groups.keys()):
        group = groups[key]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)

        train.extend(group[:n_train])
        dev.extend(group[n_train : n_train + n_dev])
        test.extend(group[n_train + n_dev :])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    return train, dev, test


def save_splits(
    train: list[dict],
    dev: list[dict],
    test: list[dict],
    output_dir: Path,
) -> dict[str, int]:
    """Save splits as JSONL files. Returns counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        counts[name] = len(data)

    return counts
