import os
import glob
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Ensure these match your actual import paths
from src.data.dataset import HybridTrajectoryDataset
from src.models.predictor import BaselineTrajectoryModel
from src.models.cvae_predictor import TrajectoryCVAE


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_inference():
    # -------------------------
    # 1. Setup & A/B Toggle
    # -------------------------
    model_cfg = load_yaml("config/model_config.yaml")["model"]
    train_cfg = load_yaml("config/train_config.yaml")["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A/B TEST TOGGLE ---
    MODEL_TYPE = "cvae" # "baseline" or "cvae"
    
    print(f"\n[INFO] Starting Inference for {MODEL_TYPE.upper()} on {device}")

    # -------------------------
    # 2. Dataset & Zero-Leakage Validation Split
    # -------------------------
    dataset = HybridTrajectoryDataset(
        parquet_path="dataset/processed/smoothed_kinematic_trajectories.parquet",
        input_window=model_cfg.get("input_window", 30),
        pred_window=model_cfg.get("pred_window", 10),
        normalize=True,
    )
    
    # Replicate the zero-leakage split from train.py to ensure we only test unseen flights
    num_flights = len(dataset.flights)
    val_fraction = train_cfg.get("val_fraction", 0.1)
    
    # Use a fixed seed so the validation set is identical to what was used in training
    np.random.seed(42)
    all_flight_indices = np.random.permutation(num_flights)
    split_idx = int(num_flights * (1 - val_fraction))
    val_flight_ids = set(all_flight_indices[split_idx:])
    
    val_indices = [i for i, meta in enumerate(dataset.index_map) if meta[0] in val_flight_ids]
    val_dataset = Subset(dataset, val_indices)
    
    # Batch size 1 for easy visualization
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # -------------------------
    # 3. Model Loading
    # -------------------------
    if MODEL_TYPE == "baseline":
        model = BaselineTrajectoryModel()
    else:
        model = TrajectoryCVAE()
        
    # Find the latest checkpoint for this specific model type
    search_pattern = os.path.join(train_cfg["checkpoint_dir"], f"{MODEL_TYPE}_best_model*.pt")
    ckpts = glob.glob(search_pattern)
    
    if not ckpts:
        # Fallback to general checkpoints if specifically named ones aren't found
        ckpts = glob.glob(os.path.join(train_cfg["checkpoint_dir"], "*.pt"))
        
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"[INFO] Loading Checkpoint: {latest_ckpt}")
    
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # -------------------------
    # 4. Denormalization Constants
    # -------------------------
    # Extract the standard deviation used for the spatial coordinates (x, y, z)
    spatial_std = dataset.std[dataset.pos_idx].astype(np.float32)

    # -------------------------
    # 5. Inference & Plotting Loop
    # -------------------------
    print("\n[INFO] Generating trajectory plots...")
    os.makedirs("inference_plots", exist_ok=True)

    with torch.no_grad():
        for i, (x_padded, y_padded, input_lengths, target_mask) in enumerate(val_loader):
            if i >= 10: # Generate 10 sample plots
                break
                
            x_padded = x_padded.to(device)
            l_in = input_lengths.item()
            l_tgt = target_mask.sum().int().item()
            
            plt.figure(figsize=(10, 8))
            
            # --- 1. Extract and Denormalize Past History ---
            # Grab only the valid history points (ignore padding)
            x_past_norm = x_padded[0, :l_in, dataset.pos_idx].cpu().numpy()
            
            # Since dataset.mean[pos_idx] is exactly 0.0, we just multiply by std
            path_past = x_past_norm * spatial_std
            plt.plot(path_past[:, 0], path_past[:, 1], 'b.-', label='Past (Input)', alpha=0.6)
            
            # --- 2. Extract and Denormalize Ground Truth Future ---
            y_gt_norm = y_padded[0, :l_tgt, :].cpu().numpy()
            path_gt = y_gt_norm * spatial_std
            
            # Insert origin [0,0,0] so the plot line connects perfectly to the current position
            path_gt = np.vstack([[0, 0, 0], path_gt])
            plt.plot(path_gt[:, 0], path_gt[:, 1], 'g.-', label='True Future', linewidth=2)

            # --- 3. Model Prediction ---
            if MODEL_TYPE == "baseline":
                # Deterministic Autoregressive Prediction
                pred_norm, _ = model(x_padded, input_lengths, steps=l_tgt, targets=None, tf_ratio=0.0)
                path_pred = pred_norm[0, :l_tgt, :].cpu().numpy() * spatial_std
                path_pred = np.vstack([[0, 0, 0], path_pred])
                
                plt.plot(path_pred[:, 0], path_pred[:, 1], 'r.--', label='Baseline Prediction', linewidth=2)

            elif MODEL_TYPE == "cvae":
                # Multi-modal Generative Inference (Generate 5 distinct futures)
                num_samples = 5
                preds_norm = model.inference(x_padded, input_lengths, steps=l_tgt, num_samples=num_samples)
                
                # preds_norm shape: [B, num_samples, steps, 3]
                for s in range(num_samples):
                    path_pred = preds_norm[0, s, :l_tgt, :].cpu().numpy() * spatial_std
                    path_pred = np.vstack([[0, 0, 0], path_pred])
                    
                    label = 'CVAE Predictions' if s == 0 else "" # Only label once for the legend
                    plt.plot(path_pred[:, 0], path_pred[:, 1], 'r.--', label=label, linewidth=1.5, alpha=0.6)

            # --- 4. Plot Aesthetics ---
            # Mark the exact current position of the aircraft
            plt.plot(0, 0, 'ko', markersize=8, label='Current Pos (Anchor)')
            
            plt.title(f"Sample {i+1}: {MODEL_TYPE.upper()} Trajectory Forecast")
            plt.xlabel("East (Meters)")
            plt.ylabel("North (Meters)")
            plt.legend()
            plt.grid(True)
            plt.axis('equal') # CRITICAL for trajectories: ensures 1m East looks the same as 1m North
            
            save_path = f"inference_plots/{MODEL_TYPE}_prediction_{i:02d}.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {save_path}")

    print("\n[INFO] Inference complete. Check the 'inference_plots' directory.")

if __name__ == "__main__":
    run_inference()