import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import glob
import os
from torch.utils.data import DataLoader, random_split

from src.data.dataset import TrajectoryDataset
from src.models.predictor import TrajectoryPredictor

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_inference():
    # 1. Setup
    model_cfg = load_yaml("config/model_config.yaml")["model"]
    train_cfg = load_yaml("config/train_config.yaml")["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Dataset
    dataset = TrajectoryDataset(
        processed_csv_path="dataset/processed_data.csv",
        input_window=model_cfg["input_window"],
        pred_window=model_cfg["pred_window"],
        normalize=True,
    )
    
    # Validation Split
    val_fraction = train_cfg.get("val_fraction", 0.15)
    num_val = int(len(dataset) * val_fraction)
    _, val_dataset = random_split(
        dataset, 
        [len(dataset) - num_val, num_val], 
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # 3. Model
    model = TrajectoryPredictor(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_regimes=model_cfg["num_regimes"],
        regime_embed_dim=model_cfg["regime_embed_dim"],
    )
    
    ckpts = glob.glob(os.path.join(train_cfg["checkpoint_dir"], "*.pt"))
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"Loading: {latest_ckpt}")
    
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # 4. Visualization Loop
    print("Generating plots...")
    
    # We need the spatial std to denormalize targets/predictions
    # dataset.std is [11], we need first 3 (x,y,z)
    spatial_std = dataset.std[0:3].astype(np.float32)

    with torch.no_grad():
        for i, (x_past, y_future_gt) in enumerate(val_loader):
            if i >= 5: break
            
            x_past = x_past.to(device)
            
            # Predict (No Teacher Forcing!)
            pred_deltas, _ = model(x_past, pred_steps=model_cfg["pred_window"], teacher_forcing_ratio=0.0)
            
            # --- DENORMALIZATION ---
            
            # 1. Past History (Standard Z-score denorm)
            # x_past was normalized with (val - mean) / std
            # We want to get back to Meters (relative to end point)
            x_past_np = x_past.cpu().numpy().squeeze()
            x_past_meters = (x_past_np * dataset.std) + dataset.mean
            path_past = x_past_meters[:, 0:3] # Only XYZ
            
            # 2. Future Ground Truth (Scaled Deltas)
            # Targets were just divided by spatial_std. Multiply back.
            y_gt_deltas = y_future_gt.numpy().squeeze() * spatial_std
            
            # 3. Prediction (Scaled Deltas)
            y_pred_deltas = pred_deltas.cpu().numpy().squeeze() * spatial_std
            
            # --- INTEGRATION (THE FIX) ---
            # We must CUMSUM the deltas to get a path
            path_gt = np.cumsum(y_gt_deltas, axis=0)
            path_pred = np.cumsum(y_pred_deltas, axis=0)
            
            # --- PLOTTING ---
            plt.figure(figsize=(10, 8))
            
            # Anchor everything to (0,0) which is the aircraft's current position
            # Past path ends at (0,0), so we shift it
            path_past = path_past - path_past[-1] 
            
            plt.plot(path_past[:, 0], path_past[:, 1], 'b.-', label='Past (Input)', alpha=0.6)
            
            # GT and Pred start at 0,0 (relative to current)
            # Insert [0,0,0] at the start for valid plotting connection
            path_gt = np.vstack([[0,0,0], path_gt])
            path_pred = np.vstack([[0,0,0], path_pred])
            
            plt.plot(path_gt[:, 0], path_gt[:, 1], 'g.-', label='True Future', linewidth=2)
            plt.plot(path_pred[:, 0], path_pred[:, 1], 'r.--', label='Predicted', linewidth=2)
            
            # Add markers for start/end
            plt.plot(0, 0, 'ko', markersize=8, label='Current Pos')
            
            plt.title(f"Sample {i}: Trajectory Prediction")
            plt.xlabel("East (m)")
            plt.ylabel("North (m)")
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            plt.savefig(f"prediction_{i}.png")
            print(f"Saved prediction_{i}.png")
            plt.close()

if __name__ == "__main__":
    run_inference()