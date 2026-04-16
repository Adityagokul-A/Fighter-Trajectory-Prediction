import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# =========================
# CONFIG
# =========================
MODEL_PATH = "checkpoints/discrete_intent_profiler_best.pt"
DATA_PATH = "dataset/processed/smoothed_kinematic_trajectories.parquet"
ASSIGNMENTS_PATH = "dataset/processed/flight_profile_assignments.csv"

OUTPUT_DIR = "inference_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# IMPORT YOUR PIPELINE
# =========================
from scripts.discrete_profiler import (
    build_sequences_from_assignments,
    SoftLabelProfiler,
)

# =========================
# LOAD MODEL
# =========================
def load_model(num_profiles):
    model = SoftLabelProfiler(num_profiles=num_profiles).to(DEVICE)
    payload = torch.load(MODEL_PATH, map_location=DEVICE)

    model.encoder.load_state_dict(payload["encoder_state"])
    model.classifier.load_state_dict(payload["classifier_state"])

    model.eval()
    return model


# =========================
# ENCODE + PREDICT
# =========================
@torch.no_grad()
def encode_and_predict(model, sequences):
    embeddings = []
    preds = []

    for seq in tqdm(sequences, desc="Encoding + Predicting"):
        x = seq.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([len(seq)], device=DEVICE)

        z = model.encoder(x, lengths)
        logits = model.classifier(z)

        embeddings.append(z.squeeze(0).cpu().numpy())
        preds.append(torch.argmax(logits, dim=-1).item())

    return np.array(embeddings), np.array(preds)


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(y_true, y_pred, num_profiles):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_profiles)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    print("[SAVED] confusion_matrix.png")


# =========================
# KMEANS + METRICS
# =========================
def run_clustering(embeddings, y_true, num_clusters):
    print("[INFO] Running KMeans clustering...")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(y_true, cluster_ids)
    nmi = normalized_mutual_info_score(y_true, cluster_ids)
    sil = silhouette_score(embeddings, cluster_ids)

    print("\n=== CLUSTERING METRICS ===")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"Silhouette: {sil:.4f}")

    return cluster_ids


# =========================
# t-SNE VISUALIZATION
# =========================
def run_tsne(embeddings):
    print("[INFO] Normalizing embeddings...")
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    print("[INFO] Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate=200,
        max_iter=1000,
        init="pca",
        random_state=42,
    )

    return tsne.fit_transform(embeddings)


def plot_tsne(reduced, labels, title, filename):
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        alpha=0.7,
    )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

    print(f"[SAVED] {filename}")


# =========================
# MAIN
# =========================
def main():
    print("[INFO] Loading data...")
    df = pd.read_parquet(DATA_PATH)

    sequences, hard_labels, _, _, num_profiles = (
        build_sequences_from_assignments(df, ASSIGNMENTS_PATH)
    )

    y_true = np.array(hard_labels)

    print("[INFO] Loading model...")
    model = load_model(num_profiles)

    print("[INFO] Encoding sequences...")
    embeddings, preds = encode_and_predict(model, sequences)

    # =========================
    # CONFUSION MATRIX
    # =========================
    plot_confusion_matrix(y_true, preds, num_profiles)

    # =========================
    # CLUSTERING
    # =========================
    cluster_ids = run_clustering(embeddings, y_true, num_profiles)

    # =========================
    # t-SNE
    # =========================
    reduced = run_tsne(embeddings)

    # Ground truth
    plot_tsne(reduced, y_true, "t-SNE (Ground Truth)", "tsne_ground_truth.png")

    # Model predictions
    plot_tsne(reduced, preds, "t-SNE (Model Predictions)", "tsne_predictions.png")

    # KMeans clusters
    plot_tsne(reduced, cluster_ids, "t-SNE (KMeans Clusters)", "tsne_clusters.png")

    print("\n[DONE] Full evaluation complete.")


if __name__ == "__main__":
    main()