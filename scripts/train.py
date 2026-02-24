import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Assuming these are your file paths, update them if your folder structure differs!
from src.data.dataset import HybridTrajectoryDataset
from src.models.predictor import BaselineTrajectoryModel
from src.models.cvae_predictor import TrajectoryCVAE
 
from src.training.losses import BaselineLossWrapper
from src.training.cvae_losses import CVAELossWrapper

from src.training.trainer import UniversalTrainer
from src.training.training_monitor import UnifiedTrainingMonitor

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # -------------------------
    # 1. Configuration & A/B Toggle
    # -------------------------
    model_cfg = load_yaml("config/model_config.yaml")["model"]
    train_cfg = load_yaml("config/train_config.yaml")["training"]
    
    # --- A/B TEST TOGGLE ---
    # Set this to "baseline" or "cvae"
    MODEL_TYPE = "cvae" 
    
    device = torch.device("cuda" if train_cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Starting {MODEL_TYPE.upper()} Pipeline on {device}")

    # -------------------------
    # 2. Dataset & Zero-Leakage Split
    # -------------------------
    dataset = HybridTrajectoryDataset(
        parquet_path="dataset/processed/smoothed_kinematic_trajectories.parquet",
        input_window=model_cfg.get("input_window", 30),
        pred_window=model_cfg.get("pred_window", 10),
        normalize=True,
    )

    # CRITICAL FIX: Split by Unique Flight ID to prevent sliding window leakage
    print("\n[INFO] Performing Zero-Leakage Data Split by Flight ID...")
    num_flights = len(dataset.flights)
    val_fraction = train_cfg.get("val_fraction", 0.1)
    
    # Shuffle flight indices
    all_flight_indices = np.random.permutation(num_flights)
    split_idx = int(num_flights * (1 - val_fraction))
    
    train_flight_ids = set(all_flight_indices[:split_idx])
    val_flight_ids = set(all_flight_indices[split_idx:])
    
    # Map back to the sliding windows (index_map stores: (flight_idx, start, l_in, l_tgt))
    train_indices = [i for i, meta in enumerate(dataset.index_map) if meta[0] in train_flight_ids]
    val_indices = [i for i, meta in enumerate(dataset.index_map) if meta[0] in val_flight_ids]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True, drop_last=False
    )

    # -------------------------
    # 3. Model & Loss Initialization
    # -------------------------
    if MODEL_TYPE == "baseline":
        model = BaselineTrajectoryModel()
        criterion = BaselineLossWrapper(
            weight_pos=train_cfg["loss_weights"].get("position", 1.0),
            weight_turn=train_cfg["loss_weights"].get("turn", 10.0),
            weight_entropy=train_cfg["loss_weights"].get("entropy", 0.1)
        )
    else:
        model = TrajectoryCVAE()
        criterion = CVAELossWrapper(
            weight_pos=train_cfg["loss_weights"].get("position", 1.0),
            weight_kl=train_cfg["loss_weights"].get("kl", 0.1),
            weight_turn=train_cfg["loss_weights"].get("turn", 10.0)
        )

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_cfg["learning_rate"], 
        weight_decay=train_cfg.get("weight_decay", 1e-5)
    )

    # -------------------------
    # 4. Trainer & Monitor Setup
    # -------------------------
    trainer = UniversalTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=train_cfg["checkpoint_dir"],
        model_type=MODEL_TYPE,
        grad_clip=train_cfg.get("grad_clip", 1.0),
    )

    monitor = UnifiedTrainingMonitor(
        checkpoint_dir=train_cfg["checkpoint_dir"], 
        model_name=MODEL_TYPE
    )

    # -------------------------
    # 5. Training Loop
    # -------------------------
    num_epochs = train_cfg["num_epochs"]
    pred_steps = model_cfg.get("pred_window", 10)
    tf_cfg = train_cfg.get("teacher_forcing", {"start": 1.0, "end": 0.0, "decay_epochs": int(num_epochs/2)})

    best_val_loss = float("inf")
    print(f"\n[INFO] Commencing Training: {num_epochs} Epochs")

    for epoch in range(1, num_epochs + 1):
        
        # Train & Val
        train_metrics = trainer.train_epoch(train_loader, epoch, pred_steps, tf_cfg)
        val_metrics = trainer.val_epoch(val_loader, pred_steps)

        # Logging
        monitor.log_step(epoch, train_metrics, val_metrics, train_metrics.get("teacher_forcing_ratio", 0.0))
        monitor.plot_curves()

        # Console Output
        print(f"Epoch {epoch:03d} | "
              f"Train Pos: {train_metrics.get('loss_pos', 0):.4f} | "
              f"Val Pos: {val_metrics.get('loss_pos', 0):.4f} | "
              f"Val Total: {val_metrics.get('loss_total', 0):.4f}")

        # Checkpointing
        if val_metrics.get("loss_total", float('inf')) < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            trainer.save_checkpoint(epoch, best_val_loss, filename=f"{MODEL_TYPE}_best_model.pt")
            print("   -> [Saved Best Model]")
    
    print(f"\n[INFO] {MODEL_TYPE.upper()} Training Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()