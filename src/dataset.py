from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_HEIGHT = 256
TARGET_WIDTH = 256


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_grayscale_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _list_slice_names(case_dir: Path) -> List[str]:
    seg_dir = case_dir / "seg"
    names = sorted(p.name for p in seg_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    return names


def _validate_slice_exists(case_dir: Path, slice_name: str, modalities: Sequence[int]) -> bool:
    if not (case_dir / "seg" / slice_name).exists():
        return False

    for m in modalities:
        if not (case_dir / str(m) / slice_name).exists():
            return False

    return True


def _pad_to_target(array: np.ndarray) -> np.ndarray:
    height, width = array.shape

    if height > TARGET_HEIGHT or width > TARGET_WIDTH:
        raise ValueError(
            f"Input shape {array.shape} is larger than "
            f"target shape {(TARGET_HEIGHT, TARGET_WIDTH)}."
        )

    pad_height = TARGET_HEIGHT - height
    pad_width = TARGET_WIDTH - width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return np.pad(
        array,
        pad_width=(
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ),
        mode="constant",
        constant_values=0,
    )


class UnifiedBrainSeg2DDataset(Dataset):
    """
    dataset_root/
      splits.json
      <CASE_ID>/
        0/*.png 1/*.png 2/*.png 3/*.png seg/*.png
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,  # "train" | "eval" | "test"
        modalities: List[int],
        *,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        strict: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.strict = strict

        skipped = 0

        if len(modalities) < 1:
            raise ValueError("Modalities must be a non-empty list like [0] or [0,1,2,3].")
        if any(m not in (0, 1, 2, 3) for m in modalities):
            raise ValueError(f"Modalities must be in [0,1,2,3]. Got: {modalities}")
        self.modalities = list(modalities)

        splits_path = self.dataset_root / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"Missing splits.json at: {splits_path}")

        splits = _read_json(splits_path)
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {splits_path}. Keys: {list(splits.keys())}")

        case_ids: List[str] = splits[split]
        self.samples: List[tuple[str, str]] = []  # (case_id, slice_name)

        for case_id in case_ids:
            case_dir = self.dataset_root / case_id
            if not case_dir.exists():
                if strict:
                    raise FileNotFoundError(f"Case folder not found: {case_dir}")
                continue

            slice_names = _list_slice_names(case_dir)
            if not slice_names:
                if strict:
                    raise RuntimeError(f"No seg slices found in: {case_dir / 'seg'}")
                continue

            for sname in slice_names:
                if _validate_slice_exists(case_dir, sname, self.modalities):
                    self.samples.append((case_id, sname))
                else:
                    skipped += 1
                    if strict:
                        raise RuntimeError(
                            f"Missing seg or requested modalities {self.modalities} for {case_id}/{sname}"
                        )

        print(f"[INFO] Indexed {len(self.samples)} samples (skipped {skipped}) for split={split}, modalities={self.modalities}")

        if not self.samples:
            raise RuntimeError(f"No samples indexed for {dataset_root} split={split} (modalities={self.modalities})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_id, slice_name = self.samples[idx]
        case_dir = self.dataset_root / case_id

        chans = []
        for m in self.modalities:
            img = _load_grayscale_png(case_dir / str(m) / slice_name)
            img = _pad_to_target(img)
            chans.append(img)

        x = np.stack(chans, axis=0).astype(np.float32) / 255.0

        mask = _load_grayscale_png(case_dir / "seg" / slice_name)
        mask = _pad_to_target(mask)

        y = (mask > 0).astype(np.float32)
        y = np.expand_dims(y, axis=0)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if self.transform is not None:
            x_t, y_t = self.transform(x_t, y_t)

        return x_t, y_t
