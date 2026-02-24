import os
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class UniversalTrainer:
    """
    Unified training engine for both Baseline and CVAE architectures.
    Handles variable-length padding, gradient clipping, and dynamic metric logging.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        checkpoint_dir: str,
        model_type: str = "baseline", # "baseline" or "cvae"
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_type = model_type.lower()
        self.grad_clip = grad_clip

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model.to(self.device)

    @staticmethod
    def compute_teacher_forcing_ratio(epoch: int, start: float, end: float, decay_epochs: int) -> float:
        """ Linear decay for teacher forcing. (Used only by the Baseline model). """
        if epoch >= decay_epochs:
            return end
        alpha = epoch / decay_epochs
        return start * (1 - alpha) + end * alpha

    def train_epoch(self, dataloader: DataLoader, epoch: int, pred_steps: int, tf_cfg: Dict) -> Dict[str, float]:
        self.model.train()
        
        # CVAE doesn't use teacher forcing, lock it to 0.0 if applicable
        tf_ratio = 0.0
        if self.model_type == "baseline":
            tf_ratio = self.compute_teacher_forcing_ratio(
                epoch, tf_cfg["start"], tf_cfg["end"], tf_cfg["decay_epochs"]
            )

        metrics_accum = defaultdict(float)
        num_batches = 0

        for x_padded, y_padded, input_lengths, target_mask in tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False):
            # Move data to device. input_lengths remains on CPU for pack_padded_sequence.
            x_padded = x_padded.to(self.device)
            y_padded = y_padded.to(self.device)
            target_mask = target_mask.to(self.device)

            self.optimizer.zero_grad()

            # --- FORWARD PASS & LOSS COMPUTATION ---
            if self.model_type == "baseline":
                pred, regime_probs = self.model(
                    x_padded, input_lengths, steps=pred_steps, 
                    targets=y_padded, tf_ratio=tf_ratio
                )
                loss, batch_metrics = self.criterion(pred, y_padded, target_mask, regime_probs)

            elif self.model_type == "cvae":
                pred, mu_post, logvar_post, mu_prior, logvar_prior = self.model(
                    x_padded, input_lengths, y_future=y_padded, steps=pred_steps
                )
                loss, batch_metrics = self.criterion(
                    pred, y_padded, target_mask, 
                    mu_post, logvar_post, mu_prior, logvar_prior
                )

            # --- BACKWARD & OPTIMIZE ---
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # Accumulate metrics dynamically
            for k, v in batch_metrics.items():
                metrics_accum[k] += v

            num_batches += 1

        # Average all metrics for the epoch
        avg_metrics = {k: v / max(1, num_batches) for k, v in metrics_accum.items()}
        avg_metrics["teacher_forcing_ratio"] = tf_ratio
        
        return avg_metrics

    @torch.no_grad()
    def val_epoch(self, dataloader: DataLoader, pred_steps: int) -> Dict[str, float]:
        self.model.eval()
        metrics_accum = defaultdict(float)
        num_batches = 0

        for x_padded, y_padded, input_lengths, target_mask in tqdm(dataloader, desc="Validation", leave=False):
            x_padded = x_padded.to(self.device)
            y_padded = y_padded.to(self.device)
            target_mask = target_mask.to(self.device)

            if self.model_type == "baseline":
                # Strict autoregressive evaluation (no teacher forcing)
                pred, regime_probs = self.model(
                    x_padded, input_lengths, steps=pred_steps, 
                    targets=None, tf_ratio=0.0
                )
                _, batch_metrics = self.criterion(pred, y_padded, target_mask, regime_probs)

            elif self.model_type == "cvae":
                pred, mu_post, logvar_post, mu_prior, logvar_prior = self.model(
                    x_padded, input_lengths, y_future=y_padded, steps=pred_steps
                )
                _, batch_metrics = self.criterion(
                    pred, y_padded, target_mask, 
                    mu_post, logvar_post, mu_prior, logvar_prior
                )

            for k, v in batch_metrics.items():
                metrics_accum[k] += v
            num_batches += 1

        return {k: v / max(1, num_batches) for k, v in metrics_accum.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, filename: Optional[str] = None):
        if filename is None:
            filename = f"epoch_{epoch:03d}_val_{val_loss:.4f}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"], checkpoint["val_loss"]