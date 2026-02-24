import os
import glob
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import HybridTrajectoryDataset
from src.models.predictor import BaselineTrajectoryModel
from src.models.cvae_predictor import TrajectoryCVAE


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    # -------------------------
    # 1. Setup & A/B Toggle
    # -------------------------
    model_cfg = load_yaml("config/model_config.yaml")["model"]
    train_cfg = load_yaml("config/train_config.yaml")["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A/B TEST TOGGLE ---
    MODEL_TYPE = "cvae" # "baseline" or "cvae"
    NUM_CVAE_SAMPLES = 5 # Used for Best-of-N metrics if MODEL_TYPE == "cvae"
    
    print(f"\n[INFO] Starting Final Evaluation for {MODEL_TYPE.upper()} on {device}")

    # -------------------------
    # 2. Dataset & Zero-Leakage Split
    # -------------------------
    dataset = HybridTrajectoryDataset(
        parquet_path="dataset/processed/smoothed_kinematic_trajectories.parquet",
        input_window=model_cfg.get("input_window", 30),
        pred_window=model_cfg.get("pred_window", 10),
        normalize=True,
    )
    
    num_flights = len(dataset.flights)
    val_fraction = train_cfg.get("val_fraction", 0.1)
    np.random.seed(42) # MUST match train.py to ensure zero leakage!
    
    all_flight_indices = np.random.permutation(num_flights)
    split_idx = int(num_flights * (1 - val_fraction))
    val_flight_ids = set(all_flight_indices[split_idx:])
    
    val_indices = [i for i, meta in enumerate(dataset.index_map) if meta[0] in val_flight_ids]
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    # -------------------------
    # 3. Model Loading
    # -------------------------
    if MODEL_TYPE == "baseline":
        model = BaselineTrajectoryModel()
    else:
        model = TrajectoryCVAE()
        
    search_pattern = os.path.join(train_cfg["checkpoint_dir"], f"{MODEL_TYPE}_best_model*.pt")
    ckpts = glob.glob(search_pattern)
    if not ckpts:
        ckpts = glob.glob(os.path.join(train_cfg["checkpoint_dir"], "*.pt"))
        
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"[INFO] Loaded Weights: {latest_ckpt}")
    
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # -------------------------
    # 4. Evaluation Loop
    # -------------------------
    spatial_std = torch.from_numpy(dataset.std[dataset.pos_idx].astype(np.float32)).to(device)
    
    all_metrics = {"ADE": [], "FDE": [], "ATE": [], "CTE": []}

    with torch.no_grad():
        for x_padded, y_padded, input_lengths, target_mask in tqdm(val_loader, desc="Evaluating"):
            x_padded = x_padded.to(device)
            y_padded = y_padded.to(device)
            target_mask = target_mask.to(device) # [B, k]
            
            # --- PREDICTION ---
            if MODEL_TYPE == "baseline":
                pred, _ = model(x_padded, input_lengths, steps=y_padded.size(1), targets=None, tf_ratio=0.0)
                # Add a dummy dimension so it matches CVAE shape [B, num_samples, k, 3]
                pred = pred.unsqueeze(1) 
            elif MODEL_TYPE == "cvae":
                pred = model.inference(x_padded, input_lengths, steps=y_padded.size(1), num_samples=NUM_CVAE_SAMPLES)
            
            # --- DENORMALIZATION ---
            spatial_std_t = spatial_std.view(1, 1, 1, 3)
            pred_m = pred * spatial_std_t             # [B, S, k, 3]
            target_m = y_padded.unsqueeze(1) * spatial_std_t # [B, 1, k, 3]
            
            error_vec = pred_m - target_m # [B, S, k, 3]
            
            # --- 1. Distances (ADE / FDE) ---
            dist = torch.linalg.norm(error_vec, dim=-1) # [B, S, k]
            
            # Mask out invalid padding zeros
            expanded_mask = target_mask.unsqueeze(1).expand_as(dist) # [B, S, k]
            masked_dist = dist * expanded_mask
            
            # Average Displacement Error (ADE) per sample sequence
            seq_ade = masked_dist.sum(dim=-1) / (expanded_mask.sum(dim=-1) + 1e-9) # [B, S]
            
            # Final Displacement Error (FDE) (Extract distance at the last VALID timestep)
            # length of sequence = mask.sum(dim=-1). Index is length - 1.
            valid_lengths = target_mask.sum(dim=-1).long() # [B]
            fde_indices = (valid_lengths - 1).view(-1, 1, 1).expand(-1, pred.size(1), 1) # [B, S, 1]
            seq_fde = torch.gather(dist, 2, fde_indices).squeeze(2) # [B, S]
            
            # Best-of-N: Take the minimum error across all S samples
            min_ade, _ = seq_ade.min(dim=1) # [B]
            min_fde, _ = seq_fde.min(dim=1) # [B]
            
            all_metrics["ADE"].append(min_ade.cpu())
            all_metrics["FDE"].append(min_fde.cpu())
            
            # --- 2. Aviation Metrics (ATE / CTE) ---
            # Compute Track Direction from Ground Truth
            # target_m shape: [B, 1, k, 3]. Because start is relative (0,0,0), first vector is target[0] - 0
            zeros = torch.zeros_like(target_m[:, :, :1, :])
            prev_pos = torch.cat([zeros, target_m[:, :, :-1, :]], dim=2)
            track_vec = target_m - prev_pos # [B, 1, k, 3]
            
            # We only care about 2D lateral tracking for standard ATE/CTE
            track_vec_2d = track_vec[:, :, :, 0:2]
            error_vec_2d = error_vec[:, :, :, 0:2]
            
            track_mag = torch.linalg.norm(track_vec_2d, dim=-1, keepdim=True) + 1e-6
            track_unit = track_vec_2d / track_mag # [B, 1, k, 2]
            
            # Project error onto track vector (Along-Track)
            ate = torch.sum(error_vec_2d * track_unit, dim=-1) # [B, S, k]
            
            # Rotate track unit 90 degrees for cross-track: (x, y) -> (y, -x)
            track_perp = torch.stack([track_unit[..., 1], -track_unit[..., 0]], dim=-1)
            cte = torch.sum(error_vec_2d * track_perp, dim=-1) # [B, S, k]
            
            # Mask, average over time, and take Best-of-N absolute error
            masked_ate = ate.abs() * expanded_mask
            seq_ate = masked_ate.sum(dim=-1) / (expanded_mask.sum(dim=-1) + 1e-9)
            min_ate, _ = seq_ate.min(dim=1)
            
            masked_cte = cte.abs() * expanded_mask
            seq_cte = masked_cte.sum(dim=-1) / (expanded_mask.sum(dim=-1) + 1e-9)
            min_cte, _ = seq_cte.min(dim=1)

            all_metrics["ATE"].append(min_ate.cpu())
            all_metrics["CTE"].append(min_cte.cpu())

    # -------------------------
    # 5. Result Aggregation
    # -------------------------
    for k in all_metrics: 
        all_metrics[k] = torch.cat(all_metrics[k]).numpy()
    
    print(f"\n=======================================================")
    print(f" FINAL RESULTS: {MODEL_TYPE.upper()} ")
    if MODEL_TYPE == "cvae":
        print(f" Evaluated using Best-of-N (N={NUM_CVAE_SAMPLES} samples)")
    print(f"=======================================================")
    
    for k, v in all_metrics.items():
        print(f"{k:>3}: Mean = {np.mean(v):6.2f} m | Median = {np.median(v):6.2f} m | 95th Pct = {np.percentile(v, 95):6.2f} m")
    print(f"=======================================================\n")

if __name__ == "__main__":
    evaluate()