import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Added for 3D plotting
import yaml
import glob
import os
from torch.utils.data import DataLoader, random_split

from src.data.dataset import TrajectoryDataset
from src.models.predictor import TrajectoryPredictor

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_long_inference(extended_horizon=40):
    """
    Runs inference with a longer prediction window than trained for.
    extended_horizon: Number of future steps to predict (e.g. 40)
    Generates 3D interactive plots using Plotly.
    """
    # 1. Setup
    model_cfg = load_yaml("config/model_config.yaml")["model"]
    train_cfg = load_yaml("config/train_config.yaml")["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] Running Long-Horizon Inference for {extended_horizon} steps...")
    print(f"[Info] Model was trained on {model_cfg['pred_window']} steps.")

    # 2. Dataset - CRITICAL CHANGE
    # We must initialize the dataset with the NEW horizon so we get 
    # enough Ground Truth data to compare against.
    dataset = TrajectoryDataset(
        processed_csv_path="dataset/processed_data.csv",
        input_window=model_cfg["input_window"],
        pred_window=extended_horizon, # Override here!
        normalize=True,
    )
    
    # Validation Split
    val_fraction = train_cfg.get("val_fraction", 0.15)
    num_val = int(len(dataset) * val_fraction)
    # Use same seed to try and get similar set, though sizes changed
    _, val_dataset = random_split(
        dataset, 
        [len(dataset) - num_val, num_val], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Filter for interesting samples (high turns) if possible, 
    # but for now random shuffle is fine.
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # 3. Model
    model = TrajectoryPredictor(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_regimes=model_cfg["num_regimes"],
        regime_embed_dim=model_cfg["regime_embed_dim"],
    )
    
    # Load Best Checkpoint
    ckpts = glob.glob(os.path.join(train_cfg["checkpoint_dir"], "*.pt"))
    if not ckpts:
        print("No checkpoints found!")
        return
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"Loading weights: {latest_ckpt}")
    
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # 4. Loop
    spatial_std = dataset.std[0:3].astype(np.float32)

    save_dir = "long_term_plots_3d"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (x_past, y_future_gt) in enumerate(val_loader):
            if i >= 10: break # Generate 10 examples
            
            x_past = x_past.to(device)
            
            # --- PREDICTION ---
            # We ask the model to predict 'extended_horizon' steps.
            # The Decoder will simply keep looping using its own output.
            pred_deltas, _ = model(
                x_past, 
                pred_steps=extended_horizon, 
                teacher_forcing_ratio=0.0
            )
            
            # --- RECONSTRUCTION (Same as inference.py) ---
            # 1. Past
            x_past_np = x_past.cpu().numpy().squeeze()
            x_past_meters = (x_past_np * dataset.std) + dataset.mean
            path_past = x_past_meters[:, 0:3]
            
            # 2. GT Future
            y_gt_deltas = y_future_gt.numpy().squeeze() * spatial_std
            path_gt = np.cumsum(y_gt_deltas, axis=0)
            
            # 3. Pred Future
            y_pred_deltas = pred_deltas.cpu().numpy().squeeze() * spatial_std
            path_pred = np.cumsum(y_pred_deltas, axis=0)
            
            # --- PLOTTING 3D (Plotly) ---
            
            # Center plots on current position (0,0,0)
            path_past = path_past - path_past[-1] 
            
            # Connect past to future (insert 0,0,0 at start)
            path_gt = np.vstack([[0,0,0], path_gt])
            path_pred = np.vstack([[0,0,0], path_pred])

            fig = go.Figure()

            # 1. Plot History (Cyan)
            fig.add_trace(go.Scatter3d(
                x=path_past[:, 0], y=path_past[:, 1], z=path_past[:, 2],
                mode='lines+markers',
                name='History (30s)',
                line=dict(color='#00ffff', width=4), # Cyan
                marker=dict(size=3, color='#00ffff', opacity=0.8)
            ))

            # 2. Plot True Future (Lime Green)
            fig.add_trace(go.Scatter3d(
                x=path_gt[:, 0], y=path_gt[:, 1], z=path_gt[:, 2],
                mode='lines',
                name=f'True Future ({extended_horizon*10}s)',
                line=dict(color='#39ff14', width=6) # Neon Green
            ))

            # 3. Plot Predicted Future (Magenta/Red)
            fig.add_trace(go.Scatter3d(
                x=path_pred[:, 0], y=path_pred[:, 1], z=path_pred[:, 2],
                mode='lines',
                name=f'Predicted ({extended_horizon*10}s)',
                line=dict(color='#ff00ff', width=6, dash='solid') # Magenta
            ))

            # 4. Mark "Current Position" (White Orb)
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                name='Current Position',
                marker=dict(size=10, color='white', symbol='circle')
            ))

            # 5. Mark "Training Horizon Limit" (Where model stops being trained)
            train_horizon = model_cfg['pred_window']
            if train_horizon < extended_horizon:
                fig.add_trace(go.Scatter3d(
                    x=[path_pred[train_horizon, 0]], 
                    y=[path_pred[train_horizon, 1]], 
                    z=[path_pred[train_horizon, 2]],
                    mode='markers',
                    name='Training Horizon Limit',
                    marker=dict(size=8, color='yellow', symbol='x')
                ))

            # Layout Styling
            fig.update_layout(
                title=f"3D Trajectory Prediction (Sample {i})",
                template="plotly_dark",
                scene=dict(
                    xaxis_title='East (m)',
                    yaxis_title='North (m)',
                    zaxis_title='Up (m)',
                    aspectmode='data'  # Crucial for realistic physics scaling
                ),
                margin=dict(r=0, l=0, b=0, t=40),
                legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05)
            )
            
            filename = os.path.join(save_dir, f"long_pred_3d_{i}.html")
            fig.write_html(filename)
            print(f"Saved interactive 3D plot: {filename}")

if __name__ == "__main__":
    # You can change this number to 40, 60, etc.
    run_long_inference(extended_horizon=40)