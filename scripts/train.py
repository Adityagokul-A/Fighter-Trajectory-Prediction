import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.dataset import TrajectoryDataset
from src.models.predictor import TrajectoryPredictor
from src.training.losses import CombinedLoss
from src.training.trainer import Trainer


# -----------------------------
# Utility: load YAML
# -----------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Main
# -----------------------------
def main():
    # -------------------------
    # 1. Load configs
    # -------------------------
    model_cfg = load_yaml("config/model_config.yaml")["model"]
    train_cfg = load_yaml("config/train_config.yaml")["training"]

    # -------------------------
    # 2. Device Setup
    # -------------------------
    device = torch.device(
        "cuda" if train_cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # 3. Dataset & Dataloaders
    # -------------------------
    dataset = TrajectoryDataset(
        processed_csv_path="dataset/processed_data.csv",
        input_window=model_cfg.get("input_window", 30),
        pred_window=model_cfg.get("pred_window", 10),
        normalize=True,
    )

    # NOTE: Simple random_split on sliding windows causes some data leakage 
    # (adjacent windows share data). For production, split by Flight ID.
    val_fraction = train_cfg.get("val_fraction", 0.1)
    num_samples = len(dataset)
    num_val = int(num_samples * val_fraction)
    num_train = num_samples - num_val

    train_dataset, val_dataset = random_split(
        dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # 4. Model
    # -------------------------
    model = TrajectoryPredictor(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_regimes=model_cfg["num_regimes"],
        regime_embed_dim=model_cfg["regime_embed_dim"],
    )

    # -------------------------
    # 5. Loss & Optimizer
    # -------------------------
    criterion = CombinedLoss(
        weight_pos=train_cfg["loss_weights"]["position"],
        weight_smooth=train_cfg["loss_weights"]["smoothness"],
        weight_entropy=train_cfg["loss_weights"]["entropy"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # -------------------------
    # 6. Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=train_cfg["checkpoint_dir"],
        grad_clip=train_cfg.get("grad_clip", 1.0),
    )

    # -------------------------
    # 7. Training Loop
    # -------------------------
    num_epochs = train_cfg["num_epochs"]
    pred_steps = model_cfg.get("pred_window", 10)
    tf_cfg = train_cfg["teacher_forcing"]

    best_val_loss = float("inf")

    # TQDM Wrapper for Epochs
    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Overall Training", unit="epoch")

    for epoch in epoch_pbar:
        # Run Training
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch=epoch,
            pred_steps=pred_steps,
            tf_cfg=tf_cfg,
        )

        # Run Validation
        val_metrics = trainer.val_epoch(
            val_loader,
            pred_steps=pred_steps,
        )

        # Logging to console (compact)
        epoch_pbar.set_postfix({
            "Train L": f"{train_metrics['loss_total']:.4f}",
            "Val L": f"{val_metrics['loss_total']:.4f}",
            "Ent": f"{train_metrics['loss_ent']:.3f}"
        })

        # Detailed print every few epochs or always if preferred
        # print(f"\nEpoch {epoch}: Train Loss {train_metrics['loss_total']:.4f} | Val Loss {val_metrics['loss_total']:.4f}")

        # Checkpointing
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            trainer.save_checkpoint(epoch, best_val_loss)
    
    print(f"\n[INFO] Training complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()