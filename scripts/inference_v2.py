import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.data.dataset import HybridTrajectoryDataset
from src.models.cvae_predictor import TrajectoryCVAE


def run_cvae_3x2_visualization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_window = 30
    pred_window = 10
    num_samples = 3

    # ---- COLOR PALETTE (high contrast, clean) ----
    COLOR_PAST = "#1f77b4"      # strong blue
    COLOR_TRUE = "#2ca02c"      # strong green
    SAMPLE_COLORS = ["#d62728", "#ff7f0e", "#9467bd"]  # red, orange, purple

    # ---- LOAD DATASET ----
    dataset = HybridTrajectoryDataset(
        parquet_path="dataset/processed/smoothed_kinematic_trajectories.parquet",
        input_window=input_window,
        pred_window=pred_window,
        normalize=True,
    )

    spatial_std = dataset.std[dataset.pos_idx].astype(np.float32)

    # ---- LOAD MODEL ----
    model = TrajectoryCVAE()
    ckpt = torch.load("checkpoints/cvae_best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ---- LOAD RAW DATA FOR CTN IDS ----
    df = pd.read_parquet("dataset/processed/smoothed_kinematic_trajectories.parquet")

    selected = []
    for ctn, flight_df in df.groupby("CTN_New"):
        flight_df = flight_df.sort_values("TIME")
        X = flight_df[dataset.feature_cols].values.astype(np.float32)

        if len(X) >= input_window + pred_window:
            selected.append((ctn, X))

        if len(selected) == 6:
            break

    if len(selected) < 6:
        raise ValueError("Not enough valid flights for 3x2 plot.")

    # ---- FIGURE SETUP ----
    plt.style.use("default")

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes = axes.flatten()

    fig.suptitle(
        "CVAE Trajectory Forecast (3 Samples per Flight)",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    seq_len = torch.tensor([input_window], device=device)

    # ---- INFERENCE ----
    with torch.no_grad():
        for i, (ctn_id, X_full) in enumerate(selected):
            ax = axes[i]

            # ---- PAST ----
            x_raw = X_full[:input_window].copy()
            anchor = x_raw[-1, dataset.pos_idx].copy()
            past = x_raw[:, dataset.pos_idx]

            # ---- NORMALIZE ----
            x_local = x_raw.copy()
            x_local[:, dataset.pos_idx] -= anchor
            x_norm = (x_local - dataset.mean) / dataset.std
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)

            # ---- GROUND TRUTH ----
            true_future = X_full[
                input_window - 1 : input_window + pred_window,
                dataset.pos_idx
            ]

            # ---- MULTI-SAMPLE PREDICTION ----
            preds = model.inference(
                x_tensor,
                seq_len,
                steps=pred_window,
                num_samples=num_samples
            )[0].cpu().numpy()  # [3, k, 3]

            # ---- PLOT PAST ----
            ax.plot(
                past[:, 0], past[:, 1],
                color=COLOR_PAST,
                linewidth=1.8,
                marker='o',
                markersize=3,
                label="Past"
            )

            # ---- PLOT TRUE ----
            ax.plot(
                true_future[:, 0], true_future[:, 1],
                color=COLOR_TRUE,
                linewidth=3,
                label="Actual"
            )

            # ---- PLOT ALL 3 SAMPLES ----
            for s in range(num_samples):
                pred_local = preds[s]
                pred_global = pred_local * spatial_std + anchor
                pred_path = np.vstack([anchor, pred_global])

                ax.plot(
                    pred_path[:, 0], pred_path[:, 1],
                    linestyle='--',
                    linewidth=2.2,
                    color=SAMPLE_COLORS[s],
                    label=f"Sample {s+1}"
                )

            # ---- ANCHOR POINT ----
            ax.scatter(
                anchor[0], anchor[1],
                c='black',
                s=50,
                zorder=5,
                label="Anchor"
            )

            # ---- STYLING ----
            ax.text(
                -0.12, 0.5,                 # left of axis
                f"CTN {ctn_id}",
                transform=ax.transAxes,     # use axis coordinates
                fontsize=11,
                fontweight="bold",
                va='center',                # vertical center
                ha='right',                 # align text to right edge
                rotation=90                 # vertical text (clean look)
            )
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")

            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axis('equal')

            # ---- LEGEND PER SUBPLOT ----
            ax.legend(
                fontsize=8,
                loc="best",
                frameon=True,
                facecolor="white",
                edgecolor="black"
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("inference_plots/cvae_3x2_forecast.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_cvae_3x2_visualization()

