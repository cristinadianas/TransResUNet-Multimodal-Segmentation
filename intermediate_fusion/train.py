import time
import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from intermediate_fusion.model import TResUnet, EncoderBlock
from utils import seeding, print_and_save, epoch_time, calculate_metrics
from metrics import DiceBCELoss
from dataset import UnifiedBrainSeg2DDataset


DATASET_ROOT = Path("datasets/rembrandt2")
MODALITIES = [0, 3]

scaler = GradScaler("cuda")


class _MiniDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.d3 = block(1024 + 512, 512)
        self.d2 = block(512  + 256, 256)
        self.d1 = block(256  +  64,  64)
        self.d0 = block(64,           32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, s1, s2, s3, s4):
        x = self.d3(torch.cat([self.up(s4), s3], dim=1))
        x = self.d2(torch.cat([self.up(x),  s2], dim=1))
        x = self.d1(torch.cat([self.up(x),  s1], dim=1))
        return self.head(self.d0(self.up(x)))


class EncoderWithHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderBlock(pretrained=False)
        self.decoder = _MiniDecoder()

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        return self.decoder(s1, s2, s3, s4)


def freeze_encoders(model: TResUnet) -> None:
    for enc in model.encoders:
        for p in enc.parameters():
            p.requires_grad = False


def load_encoder_weights(model: TResUnet, ckpt_paths: list, device) -> None:
    assert len(ckpt_paths) == model.num_modalities, (
        f"Expected {model.num_modalities} encoder checkpoints, got {len(ckpt_paths)}"
    )
    for i, path in enumerate(ckpt_paths):
        model.encoders[i].load_state_dict(torch.load(path, map_location=device))


def train(model, loader, optimizer, loss_fn, device, scaler: GradScaler, max_batches: int | None = None):
    model.train()

    running_loss = 0.0

    tqdm_bar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)
    for i, (x, y) in enumerate(tqdm_bar):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.float32, non_blocking=True)

        if y.ndim == 3:
            y = y.unsqueeze(1)
        y = y.float()

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")

        if max_batches is not None and (i + 1) >= max_batches:
            print(f"Smoke test passed ({max_batches} batches).")
            break

    num_batches = i + 1
    return running_loss / num_batches


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Eval", leave=False, dynamic_ncols=True)):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            if y.ndim == 3:
                y = y.unsqueeze(1)
            y = y.float()

            with autocast("cuda"):
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
    train_log_path = (log_dir / f"{dataset_name}_intermediate_fusion_m{modality_tag}.txt")
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

    phase1_epochs = 20
    phase1_lr     = 1e-4

    checkpoint_dir = Path("checkpoints")
    checkpoint_path = (checkpoint_dir / dataset_name / f"intermediate_fusion_m{modality_tag}.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    encoder_ckpt_dir = checkpoint_dir / dataset_name / "encoders"
    encoder_ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_str  = f"Batch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Phase-1 Epochs: {phase1_epochs}\nPhase-1 LR: {phase1_lr}\n"
    print_and_save(train_log_path, data_str)

    device = torch.device('cuda')
    loss_fn = DiceBCELoss()

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

    # PHASE 1: per-modality encoder pretraining

    print_and_save(train_log_path, "\n========== PHASE 1: Encoder Pretraining ==========")

    encoder_ckpt_paths = []

    for modality in MODALITIES:
        enc_ckpt = encoder_ckpt_dir / f"enc_m{modality}.pth"
        encoder_ckpt_paths.append(str(enc_ckpt))

        if enc_ckpt.exists():
            print_and_save(train_log_path,
                f"[Modality {modality}] Checkpoint found, skipping pretraining.")
            continue

        print_and_save(train_log_path,
            f"\n--- Phase 1 | Modality index {modality} ---")

        p1_train_ds = UnifiedBrainSeg2DDataset(
            DATASET_ROOT,
            "train",
            modalities=[modality],
            strict=False
        )

        p1_train_loader = DataLoader(
            p1_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )

        enc_model = EncoderWithHead().to(device)
        enc_optimizer = torch.optim.Adam(enc_model.parameters(), lr=phase1_lr)
        enc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            enc_optimizer, 'min', patience=5
        )

        best_p1_loss = float("inf")

        for epoch in range(phase1_epochs):
            p1_loss = train(enc_model, p1_train_loader, enc_optimizer, loss_fn, device, scaler)

            enc_scheduler.step(p1_loss)

            if p1_loss < best_p1_loss:
                best_p1_loss = p1_loss
                torch.save(enc_model.encoder.state_dict(), enc_ckpt)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print_and_save(
                    train_log_path,
                    f"  [Mod {modality}]"
                    f"  Epoch {epoch + 1:02}/{phase1_epochs}"
                    f"  loss={p1_loss:.4f} "
                    f"  best={best_p1_loss:.4f} "
                )

        print_and_save(
            train_log_path,
            f"[Mod {modality}] Done. "
            f"Best loss={best_p1_loss:.4f}. "
            f"Saved -> {enc_ckpt}"
        )

    # PHASE 2: full model with frozen encoders

    print_and_save(train_log_path, "\n========== PHASE 2: Full Model Training ==========")

    model = TResUnet(in_channels=in_channels, pretrained=False).to(device)

    load_encoder_weights(model, encoder_ckpt_paths, device)
    freeze_encoders(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print_and_save(train_log_path,
        f"Trainable params : {trainable:,}\n"
        f"Frozen params    : {frozen:,}  (encoders)\n"
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    loss_name = "BCE Dice Loss"
    data_str  = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    best_valid_metrics  = 0.0

    for epoch in range(num_epochs):
        print_and_save(train_log_path,
            f"\n------------------------------ EPOCH {epoch+1:02}/{num_epochs} ------------------------------")

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

            print_and_save(train_log_path, f"[OK] Saved checkpoint: {checkpoint_path}")

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str  = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)