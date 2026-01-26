import os
from typing import Dict, Optional

import torch
import torch.nn as nn # Added for clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    """
    Training engine. Handles:
    - train / val epochs
    - teacher forcing schedule
    - optimizer steps
    - gradient clipping (CRITICAL FOR GRU/RNN)
    - checkpointing
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        device: torch.device,
        checkpoint_dir: str,
        grad_clip: float = 1.0, # Added config for clip value
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip  # Store clip value

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model.to(self.device)

    # -----------------------------
    # Teacher forcing schedule
    # -----------------------------
    @staticmethod
    def compute_teacher_forcing_ratio(
        epoch: int,
        start: float,
        end: float,
        decay_epochs: int,
    ) -> float:
        if epoch >= decay_epochs:
            return end
        alpha = epoch / decay_epochs
        return start * (1 - alpha) + end * alpha

    # -----------------------------
    # One training epoch
    # -----------------------------
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        pred_steps: int,
        tf_cfg: Dict,
    ) -> Dict[str, float]:

        self.model.train()

        tf_ratio = self.compute_teacher_forcing_ratio(
            epoch,
            tf_cfg["start"],
            tf_cfg["end"],
            tf_cfg["decay_epochs"],
        )

        metrics_accum = {
            "loss_total": 0.0,
            "loss_pos": 0.0,
            "loss_smooth": 0.0,
            "loss_ent": 0.0,
        }

        num_batches = 0

        for x_past, y_target in tqdm(
            dataloader, desc=f"Train Epoch {epoch}", leave=False
        ):
            x_past = x_past.to(self.device)
            y_target = y_target.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            pred, regime_probs = self.model(
                x_past,
                pred_steps=pred_steps,
                targets=y_target,
                teacher_forcing_ratio=tf_ratio,
            )

            # Loss
            loss, batch_metrics = self.criterion(
                pred, y_target, regime_probs
            )

            loss.backward()
            
            # --- CRITICAL ADDITION: Gradient Clipping ---
            # Prevents exploding gradients in RNNs
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()

            for k in metrics_accum:
                # Use .get() to be safe against missing keys in edge cases
                metrics_accum[k] += batch_metrics.get(k, 0.0)

            num_batches += 1

        # Average metrics
        for k in metrics_accum:
            metrics_accum[k] /= max(1, num_batches)

        metrics_accum["teacher_forcing_ratio"] = tf_ratio
        return metrics_accum

    # -----------------------------
    # One validation epoch
    # -----------------------------
    @torch.no_grad()
    def val_epoch(
        self,
        dataloader: DataLoader,
        pred_steps: int,
    ) -> Dict[str, float]:

        self.model.eval()

        metrics_accum = {
            "loss_total": 0.0,
            "loss_pos": 0.0,
            "loss_smooth": 0.0,
            "loss_ent": 0.0,
        }

        num_batches = 0

        for x_past, y_target in tqdm(
            dataloader, desc="Validation", leave=False
        ):
            x_past = x_past.to(self.device)
            y_target = y_target.to(self.device)

            # Validation: Force autoregressive (TF=0.0)
            pred, regime_probs = self.model(
                x_past,
                pred_steps=pred_steps,
                targets=None,                
                teacher_forcing_ratio=0.0,
            )

            loss, batch_metrics = self.criterion(
                pred, y_target, regime_probs
            )

            for k in metrics_accum:
                metrics_accum[k] += batch_metrics.get(k, 0.0)

            num_batches += 1

        for k in metrics_accum:
            metrics_accum[k] /= max(1, num_batches)

        return metrics_accum

    # -----------------------------
    # Checkpointing
    # -----------------------------
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        filename: Optional[str] = None,
    ):
        if filename is None:
            filename = f"epoch_{epoch:03d}_val_{val_loss:.4f}.pt"

        path = os.path.join(self.checkpoint_dir, filename)

        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"], checkpoint["val_loss"]