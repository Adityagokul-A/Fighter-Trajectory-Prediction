import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.models.cvae_predictor import TrajectoryCVAE


# =========================
# 1. EXTRACT LATENTS
# =========================
def extract_latent_intents():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Extracting Latent Profiles on {device}")

    data_path = "dataset/processed/smoothed_kinematic_trajectories.parquet"
    df = pd.read_parquet(data_path)

    feature_cols = [
        'x_s', 'y_s', 'z_s',
        'vx', 'vy', 'vz',
        'ax', 'ay', 'az',
        'ENU_Speed', 'Acceleration',
        'sin_course', 'cos_course', 'Turn_Rate'
    ]

    # Global normalization (same as before)
    X_all = df[feature_cols].values
    global_mean = X_all.mean(axis=0).astype(np.float32)
    global_std = (X_all.std(axis=0) + 1e-6).astype(np.float32)

    pos_idx = [0, 1, 2]
    global_mean[pos_idx] = 0.0

    # Load model
    model = TrajectoryCVAE().to(device)
    ckpt_path = max(glob.glob("checkpoints/cvae_best_model*.pt"), key=os.path.getctime)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()

    latent_vectors = []
    ctn_labels = []

    print("[INFO] Processing Whole-Flight Sequences...")

    with torch.no_grad():
        for ctn, group in tqdm(df.groupby('CTN_New')):
            seq = group[feature_cols].values.astype(np.float32)

            # Anchor position
            seq[:, pos_idx] -= seq[-1, pos_idx]

            # Normalize
            seq = (seq - global_mean) / global_std

            x_tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
            seq_len = torch.tensor([len(seq)], dtype=torch.long)

            # Encode
            context = model.encoder(x_tensor, seq_len)
            mu, logvar = model.prior(context)

            # 👉 Improved: use both mean + logvar
            z = torch.cat([mu, logvar], dim=-1)

            latent_vectors.append(z.squeeze(0).cpu().numpy())
            ctn_labels.append(ctn)

    return np.array(latent_vectors), ctn_labels


# =========================
# 2. FIT GMM
# =========================
def fit_gmm(latent_vectors, max_clusters=13):
    print("\n[INFO] Scaling latent vectors...")
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)

    print("[INFO] Fitting Gaussian Mixture Models...")

    bics = []
    models = []
    k_range = range(2, max_clusters + 1)

    for k in tqdm(k_range):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='diag',
            n_init=10,
            reg_covar=1e-5,
            random_state=42
        )
        gmm.fit(latent_scaled)

        bics.append(gmm.bic(latent_scaled))
        models.append(gmm)

    # Plot
    os.makedirs("inference_plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, bics, marker='o')
    plt.title("BIC for Flight Profiles")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("BIC (Lower is Better)")
    plt.grid(True)
    plt.savefig("inference_plots/gmm_bic_8.png")
    plt.close()

    best_idx = np.argmin(bics)
    best_k = k_range[best_idx]
    best_model = models[best_idx]

    print(f"[RESULT] Best K: {best_k}")

    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(best_model, "checkpoints/gmm_model.pkl")
    joblib.dump(scaler, "checkpoints/gmm_scaler.pkl")

    print("[SUCCESS] Saved GMM + scaler")

    return best_model, scaler


# =========================
# 3. ASSIGN PROFILES
# =========================
def assign_profiles(gmm, scaler, latent_vectors, ctn_labels):
    latent_scaled = scaler.transform(latent_vectors)

    probs = gmm.predict_proba(latent_scaled)
    clusters = gmm.predict(latent_scaled)

    results = []
    for i in range(len(ctn_labels)):
        confidence = np.max(probs[i]) * 100

        results.append({
            "CTN_New": ctn_labels[i],
            "Assigned_Profile": int(clusters[i]),
            "Confidence_%": round(confidence, 2),
            "Softmax": np.round(probs[i], 3).tolist()
        })

    df_out = pd.DataFrame(results)

    os.makedirs("dataset/processed", exist_ok=True)
    out_path = "dataset/processed/flight_profiles.csv"
    df_out.to_csv(out_path, index=False)

    print(f"[SUCCESS] Saved results → {out_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    latent_vecs, ctns = extract_latent_intents()
    gmm, scaler = fit_gmm(latent_vecs)
    assign_profiles(gmm, scaler, latent_vecs, ctns)