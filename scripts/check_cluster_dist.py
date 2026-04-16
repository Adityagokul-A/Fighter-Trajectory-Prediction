import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "dataset/processed/flight_profiles.csv"
MIN_CLUSTER_SIZE = 20   # threshold for "too small"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

print(f"[INFO] Loaded {len(df)} flights")

# =========================
# CLUSTER COUNTS
# =========================
counts = df["Assigned_Profile"].value_counts().sort_index()
percentages = (counts / len(df) * 100).round(2)

summary = pd.DataFrame({
    "Count": counts,
    "Percentage (%)": percentages
})

print("\n=== Cluster Distribution ===")
print(summary)

# =========================
# FLAG BAD CLUSTERS
# =========================
small_clusters = summary[summary["Count"] < MIN_CLUSTER_SIZE]

if len(small_clusters) > 0:
    print("\n[WARNING] Small / unstable clusters detected:")
    print(small_clusters)
else:
    print("\n[SUCCESS] All clusters have reasonable size")

# =========================
# OPTIONAL: PLOT
# =========================
plt.figure()
plt.bar(summary.index.astype(str), summary["Count"])
plt.xlabel("Cluster ID")
plt.ylabel("Number of Flights")
plt.title("Flight Profile Distribution (K=13)")
plt.grid()

plt.savefig("inference_plots/cluster_distribution.png")
print("\n[SUCCESS] Saved plot → inference_plots/cluster_distribution.png")

plt.show()