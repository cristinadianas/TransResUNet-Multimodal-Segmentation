import time
import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from early_fusion.model import TResUnet
from utils import seeding, print_and_save, epoch_time, calculate_metrics
from metrics import DiceBCELoss
from dataset import UnifiedBrainSeg2DDataset


DATASET_ROOT = Path("datasets/rembrandt2")
MODALITIES = [0, 3]

scaler = GradScaler("cuda")


def train(model, loader, optimizer, loss_fn, device, scaler: GradScaler):
    model.train()

    running_loss = 0.0

    tqdm_bar = tqdm(loader, desc="Train", leave=False)
    for i, (x, y) in enumerate(tqdm_bar):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Eval", leave=False)):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_prob = torch.sigmoid(y_pred)
            y_hat = (y_prob > 0.5).float()

            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, y_hat):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss      = epoch_loss      / len(loader)
        epoch_jac       = epoch_jac       / len(loader)
        epoch_f1        = epoch_f1        / len(loader)
        epoch_recall    = epoch_recall    / len(loader)
        epoch_precision = epoch_precision / len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


if __name__ == "__main__":
    seeding(42)

    in_channels  = len(MODALITIES)
    dataset_name = DATASET_ROOT.name.lower()
    modality_tag = "".join(str(m) for m in MODALITIES)

    log_dir = Path("train_log")
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = (log_dir / f"{dataset_name}_early_fusion_m{modality_tag}.txt")
    if train_log_path.exists():
        print("Log file exists")
    else:
        train_log_path.write_text("\n", encoding="utf-8")

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    batch_size = 16
    num_epochs = 20
    lr = 1e-4

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = (checkpoint_dir / dataset_name / f"early_fusion_m{modality_tag}.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    data_str  = f"Batch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    print_and_save(train_log_path, data_str)

    train_dataset = UnifiedBrainSeg2DDataset(
        DATASET_ROOT, "train", 
        modalities=MODALITIES, 
        strict=False
    )
    valid_dataset = UnifiedBrainSeg2DDataset(
        DATASET_ROOT, "eval",  
        modalities=MODALITIES, 
        strict=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    device = torch.device('cuda')
    model = TResUnet(in_channels=in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str  = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    best_valid_metrics  = 0.0

    for epoch in trange(num_epochs, desc="Epochs"):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device, scaler)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print_and_save(train_log_path, f"Current LR: {current_lr:.6e}")

        if valid_metrics[1] > best_valid_metrics:
            data_str = (
                f"Valid F1 improved from {best_valid_metrics:2.4f} "
                f"to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            )
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)

            tqdm.write(f"[OK] Saved checkpoint: {checkpoint_path}")

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str  = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)