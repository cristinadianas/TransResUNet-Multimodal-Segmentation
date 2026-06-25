from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from early_fusion.model import TResUnet
from dataset import UnifiedBrainSeg2DDataset
from utils import seeding, calculate_metrics


DATASET_ROOT = Path("datasets/rembrandt2") 
MODALITIES = [0, 3]
SPLIT = "test" 

BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True


def build_checkpoint_path() -> Path:
    dataset_name = DATASET_ROOT.name.lower()
    modality_tag = "".join(str(m) for m in MODALITIES)
    checkpoint_dir = Path("checkpoints")
    return (checkpoint_dir / dataset_name / f"early_fusion_m{modality_tag}.pth")


def build_results_csv_path(checkpoint_path: Path) -> Path:
    dataset_name = checkpoint_path.parent.name
    results_dir = Path("results") / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{checkpoint_path.stem}.csv"


@torch.no_grad()
def evaluate_to_csv(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    csv_path: Path,
    checkpoint_path: Path,
) -> None:
    metric_names = ["jaccard", "f1", "recall", "precision", "acc", "fbeta"]

    sums = np.zeros(len(metric_names), dtype=np.float64)
    n = 0
    batch_times = []

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["checkpoint", str(checkpoint_path)])
        writer.writerow(["dataset_root", str(DATASET_ROOT)])
        writer.writerow(["split", SPLIT])
        writer.writerow(["modalities", ",".join(map(str, MODALITIES))])
        writer.writerow([])

        writer.writerow(["index"] + metric_names)

        global_idx = 0
        for x, y in tqdm(loader, desc=f"Eval ({SPLIT})"):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            t0 = time.time()
            logits = model(x)
            y_pred = torch.sigmoid(logits)
            batch_times.append(time.time() - t0)

            for yt, yp in zip(y, y_pred):
                scores = calculate_metrics(yt, yp)
                scores = np.asarray(scores, dtype=np.float64)

                sums += scores
                n += 1

                writer.writerow(
                    [global_idx] + [f"{v:.3f}" for v in scores]
                )

                global_idx += 1

        writer.writerow([])
        avgs = sums / max(n, 1)
        writer.writerow(
            ["AVERAGE"] + [f"{v:.3f}" for v in avgs]
        )

    avgs = sums / max(n, 1)
    print("\nAverages:")
    for name, val in zip(metric_names, avgs):
        print(f"  {name:>9}: {val:.6f}")

    mean_bt = float(np.mean(batch_times)) if batch_times else float("nan")
    if mean_bt > 0:
        print(f"\nMean batch time: {mean_bt:.4f}s")
        print(f"Mean batch FPS: {1.0 / mean_bt:.2f}")
    else:
        print("\nMean batch time: n/a")
        print("Mean batch FPS: n/a")

    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    seeding(42)

    in_channels = len(MODALITIES)

    checkpoint_path = build_checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}\n")

    csv_path = build_results_csv_path(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = UnifiedBrainSeg2DDataset(
        DATASET_ROOT,
        split=SPLIT,
        modalities=MODALITIES,
        strict=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and (device.type == "cuda"),
    )

    model = TResUnet(in_channels=in_channels).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Dataset: {DATASET_ROOT} | split={SPLIT} | modalities={MODALITIES}")
    print(f"[INFO] Writing results to: {csv_path}")

    evaluate_to_csv(model, test_loader, device, csv_path, checkpoint_path)