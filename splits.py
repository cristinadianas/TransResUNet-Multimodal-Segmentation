from __future__ import annotations

import json
import random
from pathlib import Path


DATASET_ROOT = Path("datasets/bmshare")
OUT_PATH = DATASET_ROOT / "splits.json"

SEED = 42
TRAIN_RATIO = 0.70
EVAL_RATIO = 0.15
TEST_RATIO = 0.15


def list_cases(dataset_root: Path) -> list[str]:
    cases = [
        p.name for p in dataset_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]

    cases = [c for c in cases if c not in {"train", "eval", "test"}]

    cases.sort()
    return cases


def make_splits(cases: list[str]) -> dict:
    if not cases:
        raise RuntimeError(f"No case folders found in: {DATASET_ROOT}")

    if abs((TRAIN_RATIO + EVAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise RuntimeError("TRAIN_RATIO + EVAL_RATIO + TEST_RATIO must equal 1.0")

    rng = random.Random(SEED)
    shuffled = cases[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    n_eval = int(round(n * EVAL_RATIO))

    n_train = min(n_train, n)
    n_eval = min(n_eval, n - n_train)
    n_test = n - n_train - n_eval

    train = shuffled[:n_train]
    eval_ = shuffled[n_train:n_train + n_eval]
    test = shuffled[n_train + n_eval:]

    assert len(train) + len(eval_) + len(test) == n
    assert len(test) == n_test

    return {
        "dataset_root": str(DATASET_ROOT).replace("\\", "/"),
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "eval": EVAL_RATIO, "test": TEST_RATIO},
        "counts": {"train": len(train), "eval": len(eval_), "test": len(test)},
        "train": train,
        "eval": eval_,
        "test": test,
    }


def main() -> None:
    cases = list_cases(DATASET_ROOT)
    splits = make_splits(cases)

    OUT_PATH.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(f"[OK] Wrote: {OUT_PATH}")
    print(f"[INFO] Counts: {splits['counts']}")


if __name__ == "__main__":
    main()
