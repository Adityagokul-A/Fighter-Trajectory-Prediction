import os
import json
import ast
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "dataset/processed/smoothed_kinematic_trajectories.parquet"
ASSIGNMENTS_PATH = "dataset/processed/flight_profile_assignments.csv"

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "discrete_intent_profiler_best.pt")
BANK_PATH = os.path.join(CHECKPOINT_DIR, "behavior_prototypes.npz")
LOSS_PLOT_PATH = os.path.join("inference_plots/training_loss_curve.png")
HISTORY_PATH = os.path.join("inference_plots/training_history.json")

ACCEL_DEADBAND = 2.0
TURN_DEADBAND = 0.05
CLIMB_DEADBAND = 5.0

PAD_TOKEN = 27
VOCAB_SIZE = 28  # 27 discrete states + PAD

TOKEN_EMBED_DIM = 32
ENCODER_HIDDEN_DIM = 96
EMB_DIM = 64
NUM_LAYERS = 2

BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42
TEMPERATURE = 15.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "x_s", "y_s", "z_s",
    "vx", "vy", "vz",
    "ax", "ay", "az",
    "ENU_Speed", "Acceleration",
    "sin_course", "cos_course", "Turn_Rate",
]


# =========================================================
# HELPERS
# =========================================================
def ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("dataset/processed", exist_ok=True)
    os.makedirs("inference_plots", exist_ok=True)


def parse_softmax_array(value):
    """
    Robustly parse the stored probability array from CSV.
    """
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)

    if pd.isna(value):
        raise ValueError("Encountered empty probability field in CSV.")

    if isinstance(value, str):
        try:
            return np.asarray(json.loads(value), dtype=np.float32)
        except Exception:
            return np.asarray(ast.literal_eval(value), dtype=np.float32)

    return np.asarray(value, dtype=np.float32)


def iter_flights(df):
    for ctn, group in df.groupby("CTN_New", sort=True):
        yield ctn, group.sort_values("TIME")


def discretize_kinematics(val, deadband):
    if val > deadband:
        return 1
    if val < -deadband:
        return -1
    return 0


def map_to_state_vocab(a, t, v):
    """
    Maps [-1, 0, 1] x [-1, 0, 1] x [-1, 0, 1] to [0..26].
    """
    a_idx = a + 1
    t_idx = t + 1
    v_idx = v + 1
    return (a_idx * 9) + (t_idx * 3) + v_idx


def build_sequence_from_group(group):
    """
    Build one tokenized trajectory sequence from a single flight group.
    """
    accel_raw = group["Acceleration"].values.astype(np.float32)
    turn_raw = group["Turn_Rate"].values.astype(np.float32)
    climb_raw = group["vz"].values.astype(np.float32)

    token_sequence = []
    for i in range(len(accel_raw)):
        a = discretize_kinematics(accel_raw[i], ACCEL_DEADBAND)
        t = discretize_kinematics(turn_raw[i], TURN_DEADBAND)
        v = discretize_kinematics(climb_raw[i], CLIMB_DEADBAND)
        token_sequence.append(map_to_state_vocab(a, t, v))

    return torch.tensor(token_sequence, dtype=torch.long)


def _safe_stratified_split(indices, labels, test_size=0.15, random_state=RANDOM_STATE):
    """
    Stratified split when possible; falls back to a plain split if a class
    has too few samples to stratify safely.
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    can_stratify = len(unique) > 1 and np.all(counts >= 2)

    if can_stratify:
        try:
            return train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=labels,
            )
        except ValueError:
            pass

    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )


# =========================================================
# BUILD TRAINING SEQUENCES FROM GMM OUTPUT
# =========================================================
def build_sequences_from_assignments(df, assignments_csv=ASSIGNMENTS_PATH):
    """
    Returns:
        sequences: list[Tensor]
        hard_labels: list[int]
        soft_labels: list[Tensor]
        ctns: list[str]
        num_profiles: int
    """
    print("[INFO] Loading flight assignments...")
    assign_df = pd.read_csv(assignments_csv)

    ctn_to_row = {}
    for _, row in assign_df.iterrows():
        ctn_to_row[str(row["CTN_New"])] = row

    sequences = []
    hard_labels = []
    soft_labels = []
    ctns = []

    print("[INFO] Discretizing flights into token sequences...")
    for ctn, group in tqdm(list(iter_flights(df)), desc="Building sequences"):
        if ctn not in ctn_to_row:
            continue

        row = ctn_to_row[ctn]
        probs = parse_softmax_array(row["Full_Softmax_Array"])
        hard = int(row["Assigned_Profile"])

        seq = build_sequence_from_group(group)

        sequences.append(seq)
        hard_labels.append(hard)
        soft_labels.append(torch.tensor(probs, dtype=torch.float32))
        ctns.append(ctn)

    if len(soft_labels) == 0:
        raise ValueError(
            "No matching CTNs found between the parquet file and flight_profile_assignments.csv"
        )

    num_profiles = int(soft_labels[0].shape[0])
    print(f"[INFO] Loaded {len(sequences)} labeled flights across {num_profiles} profiles")
    return sequences, hard_labels, soft_labels, ctns, num_profiles


# =========================================================
# DATASET + COLLATE
# =========================================================
class TrajectoryDataset(Dataset):
    def __init__(self, sequences, hard_labels, soft_labels, ctns):
        self.sequences = sequences
        self.hard_labels = hard_labels
        self.soft_labels = soft_labels
        self.ctns = ctns

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.hard_labels[idx],
            self.soft_labels[idx],
            self.ctns[idx],
        )


def collate_fn(batch):
    sequences, hard_labels, soft_labels, ctns = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_TOKEN)
    hard_labels = torch.tensor(hard_labels, dtype=torch.long)
    soft_labels = torch.stack(soft_labels, dim=0)
    return padded, hard_labels, soft_labels, lengths, ctns


# =========================================================
# MODEL
# =========================================================
class EncoderGRU(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        pad_token=PAD_TOKEN,
        token_embed_dim=TOKEN_EMBED_DIM,
        hidden_dim=ENCODER_HIDDEN_DIM,
        emb_dim=EMB_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_embed_dim,
            padding_idx=pad_token,
        )
        self.gru = nn.GRU(
            input_size=token_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x_padded, lengths):
        x = self.embedding(x_padded)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)

        if self.gru.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]

        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z


class SoftLabelProfiler(nn.Module):
    def __init__(self, num_profiles):
        super().__init__()
        self.encoder = EncoderGRU()
        self.classifier = nn.Linear(EMB_DIM, num_profiles)

    def forward(self, x_padded, lengths):
        z = self.encoder(x_padded, lengths)
        logits = self.classifier(z)
        return logits

    def encode(self, x_padded, lengths):
        return self.encoder(x_padded, lengths)


# =========================================================
# PROTOTYPE BANK
# =========================================================
class BehaviorPrototypeBank:
    def __init__(self, prototypes, names=None, temperature=TEMPERATURE, metadata=None):
        """
        prototypes: np.ndarray [K, D]
        """
        proto = torch.tensor(prototypes, dtype=torch.float32)
        self.prototypes = F.normalize(proto, dim=1).cpu()
        self.temperature = float(temperature)
        self.names = names if names is not None else [f"profile_{i}" for i in range(len(prototypes))]
        self.metadata = metadata if metadata is not None else [{} for _ in range(len(prototypes))]

    def __len__(self):
        return self.prototypes.shape[0]

    def to(self, device):
        self.prototypes = self.prototypes.to(device)
        return self

    def cpu(self):
        self.prototypes = self.prototypes.cpu()
        return self

    def logits(self, embeddings):
        emb = F.normalize(embeddings, dim=-1)
        prototypes = self.prototypes.to(embeddings.device)
        return self.temperature * (emb @ prototypes.T)

    def probs(self, embeddings):
        return torch.softmax(self.logits(embeddings), dim=-1)

    def predict(self, embeddings):
        return torch.argmax(self.logits(embeddings), dim=-1)

    def add_behavior(self, prototype_vec, name, metadata=None, replace_existing=False):
        """
        Add a new behavior prototype.
        If replace_existing=True and name already exists, the prototype is updated in place.
        """
        p = torch.tensor(prototype_vec, dtype=torch.float32).unsqueeze(0)
        p = F.normalize(p, dim=1)
        p = p.to(self.prototypes.device)

        metadata = metadata or {}

        if replace_existing and name in self.names:
            idx = self.names.index(name)
            self.prototypes[idx] = p.squeeze(0)
            self.metadata[idx] = metadata
            return idx

        self.prototypes = torch.cat([self.prototypes, p], dim=0)
        self.names.append(name)
        self.metadata.append(metadata)
        return len(self.names) - 1

    def save(self, path=BANK_PATH):
        np.savez(
            path,
            prototypes=self.prototypes.cpu().numpy(),
            names=np.array(self.names, dtype=object),
            metadata=np.array(self.metadata, dtype=object),
            temperature=np.array([self.temperature], dtype=np.float32),
        )

    @classmethod
    def load(cls, path=BANK_PATH):
        data = np.load(path, allow_pickle=True)
        prototypes = data["prototypes"]
        names = list(data["names"])
        metadata = list(data["metadata"]) if "metadata" in data.files else [{} for _ in range(len(names))]
        temperature = float(data["temperature"][0]) if "temperature" in data.files else TEMPERATURE
        return cls(
            prototypes=prototypes,
            names=names,
            temperature=temperature,
            metadata=metadata,
        )


# =========================================================
# PLOTTING
# =========================================================
def plot_training_curves(history, out_path=LOSS_PLOT_PATH):
    """
    Save train/val KL loss curves to disk.
    """
    if not history or not history.get("train_loss") or not history.get("val_loss"):
        print("[WARN] No history available to plot.")
        return

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train KL Loss")
    plt.plot(epochs, history["val_loss"], label="Val KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[SUCCESS] Saved loss curve plot to {out_path}")


def save_training_history(history, out_path=HISTORY_PATH):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[SUCCESS] Saved training history to {out_path}")


# =========================================================
# TRAINING
# =========================================================
def train_discrete_profiler(sequences, hard_labels, soft_labels, ctns, num_profiles):
    print("[INFO] Training discrete profiler...")

    idx = np.arange(len(sequences))
    train_idx, val_idx = _safe_stratified_split(
        idx,
        hard_labels,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    X_train = [sequences[i] for i in train_idx]
    yhard_train = [hard_labels[i] for i in train_idx]
    ysoft_train = [soft_labels[i] for i in train_idx]
    ctn_train = [ctns[i] for i in train_idx]

    X_val = [sequences[i] for i in val_idx]
    yhard_val = [hard_labels[i] for i in val_idx]
    ysoft_val = [soft_labels[i] for i in val_idx]
    ctn_val = [ctns[i] for i in val_idx]

    train_loader = DataLoader(
        TrajectoryDataset(X_train, yhard_train, ysoft_train, ctn_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TrajectoryDataset(X_val, yhard_val, ysoft_val, ctn_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SoftLabelProfiler(num_profiles=num_profiles).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_val_kl = float("inf")
    best_state = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for seqs, _, soft_lbls, lengths, _ in train_loader:
            seqs = seqs.to(DEVICE)
            soft_lbls = soft_lbls.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seqs, lengths)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, soft_lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for seqs, hard_lbls, soft_lbls, lengths, _ in val_loader:
                seqs = seqs.to(DEVICE)
                hard_lbls = hard_lbls.to(DEVICE)
                soft_lbls = soft_lbls.to(DEVICE)
                lengths = lengths.to(DEVICE)

                logits = model(seqs, lengths)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, soft_lbls)
                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == hard_lbls).sum().item()
                total += hard_lbls.size(0)

        avg_train = train_loss / max(1, train_batches)
        avg_val = val_loss / max(1, val_batches)
        val_acc = correct / max(1, total)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Train KL: {avg_train:.4f} | "
            f"Val KL: {avg_val:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        if avg_val < best_val_kl:
            best_val_kl = avg_val
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "classifier_state": model.classifier.state_dict(),
            "num_profiles": num_profiles,
            "emb_dim": EMB_DIM,
            "token_embed_dim": TOKEN_EMBED_DIM,
            "hidden_dim": ENCODER_HIDDEN_DIM,
        },
        MODEL_PATH,
    )
    print(f"[SUCCESS] Saved best profiler to {MODEL_PATH}")

    save_training_history(history, HISTORY_PATH)
    plot_training_curves(history, LOSS_PLOT_PATH)

    return model, history


# =========================================================
# EMBEDDINGS + PROTOTYPES
# =========================================================
@torch.no_grad()
def encode_sequences(encoder, sequences):
    encoder.eval()
    embeddings = []

    for seq in tqdm(sequences, desc="Encoding sequences"):
        x = seq.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([len(seq)], dtype=torch.long, device=DEVICE)
        z = encoder(x, lengths).squeeze(0).detach().cpu()
        embeddings.append(z)

    return torch.stack(embeddings, dim=0)


@torch.no_grad()
def build_prototype_bank(encoder, sequences, hard_labels, num_profiles):
    print("[INFO] Building prototype bank...")

    E = encode_sequences(encoder, sequences)
    Y = torch.tensor(hard_labels, dtype=torch.long)

    prototypes = []
    names = []
    metadata = []

    for k in range(num_profiles):
        mask = (Y == k)
        if mask.any():
            proto = E[mask].mean(dim=0)
            count = int(mask.sum().item())
        else:
            proto = E.mean(dim=0)
            count = 0

        proto = F.normalize(proto, dim=0)
        prototypes.append(proto.numpy())
        names.append(f"profile_{k}")
        metadata.append({"count": count, "source": "initial_gmm_clusters"})

    prototypes = np.vstack(prototypes)
    bank = BehaviorPrototypeBank(
        prototypes=prototypes,
        names=names,
        temperature=TEMPERATURE,
        metadata=metadata,
    )
    bank.save(BANK_PATH)
    print(f"[SUCCESS] Saved prototype bank to {BANK_PATH}")
    return bank


# =========================================================
# INFERENCE / ASSIGNMENT
# =========================================================
@torch.no_grad()
def assign_with_prototypes(
    encoder,
    bank,
    sequences,
    ctns,
    out_csv="dataset/processed/prototype_profile_assignments.csv",
):
    print("[INFO] Assigning flights using prototype similarity...")

    encoder.eval()
    results = []

    for seq, ctn in tqdm(list(zip(sequences, ctns)), desc="Prototype inference"):
        x = seq.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([len(seq)], dtype=torch.long, device=DEVICE)

        z = encoder(x, lengths)
        probs = bank.probs(z).squeeze(0).cpu().numpy()
        pred = int(np.argmax(probs))

        results.append({
            "CTN_New": ctn,
            "Assigned_Profile": pred,
            "Profile_Name": bank.names[pred],
            "Confidence_%": round(float(np.max(probs) * 100.0), 2),
            "Full_Softmax_Array": json.dumps(np.round(probs, 4).tolist()),
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved assignments to {out_csv}")
    return df_out


# =========================================================
# ADD NEW BEHAVIOR
# =========================================================
@torch.no_grad()
def add_new_behavior_from_dataframe(
    new_df,
    encoder,
    bank,
    behavior_name,
    replace_existing=False,
    feature_cols=FEATURE_COLS,
    output_bank_path=BANK_PATH,
):
    """
    Add a new behavior prototype from a dataframe of new flights.

    Expected input:
        - new_df contains one or more flights
        - columns: CTN_New, TIME, and the same motion columns used by the profiler

    This function:
        1) tokenizes each new flight
        2) encodes each flight with the GRU encoder
        3) averages the embeddings into one prototype
        4) appends or replaces the prototype in the bank
        5) saves the updated bank
    """
    print(f"[INFO] Adding new behavior: {behavior_name}")

    sequences = []
    for _, group in iter_flights(new_df):
        seq = build_sequence_from_group(group)
        if len(seq) > 0:
            sequences.append(seq)

    if len(sequences) == 0:
        raise ValueError("No valid flights found in new_df for behavior addition.")

    embeddings = encode_sequences(encoder, sequences)
    prototype = F.normalize(embeddings.mean(dim=0), dim=0).cpu().numpy()

    idx = bank.add_behavior(
        prototype_vec=prototype,
        name=behavior_name,
        metadata={
            "count": len(sequences),
            "source": "manual_new_behavior",
        },
        replace_existing=replace_existing,
    )

    bank.save(output_bank_path)
    print(f"[SUCCESS] Behavior '{behavior_name}' saved at index {idx}")
    print(f"[SUCCESS] Updated prototype bank saved to {output_bank_path}")
    return bank


@torch.no_grad()
def add_new_behavior_from_sequences(
    sequences,
    encoder,
    bank,
    behavior_name,
    replace_existing=False,
    output_bank_path=BANK_PATH,
):
    """
    Same as add_new_behavior_from_dataframe, but takes pre-tokenized sequences directly.
    """
    print(f"[INFO] Adding new behavior from token sequences: {behavior_name}")

    if len(sequences) == 0:
        raise ValueError("No sequences were provided.")

    embeddings = encode_sequences(encoder, sequences)
    prototype = F.normalize(embeddings.mean(dim=0), dim=0).cpu().numpy()

    idx = bank.add_behavior(
        prototype_vec=prototype,
        name=behavior_name,
        metadata={
            "count": len(sequences),
            "source": "manual_new_behavior_sequences",
        },
        replace_existing=replace_existing,
    )

    bank.save(output_bank_path)
    print(f"[SUCCESS] Behavior '{behavior_name}' saved at index {idx}")
    print(f"[SUCCESS] Updated prototype bank saved to {output_bank_path}")
    return bank


# =========================================================
# LOAD TRAINED MODEL
# =========================================================
def load_trained_encoder(num_profiles):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    model = SoftLabelProfiler(num_profiles=num_profiles).to(DEVICE)
    payload = torch.load(MODEL_PATH, map_location=DEVICE)
    model.encoder.load_state_dict(payload["encoder_state"])
    model.classifier.load_state_dict(payload["classifier_state"])
    model.eval()
    print(f"[INFO] Loaded trained profiler from {MODEL_PATH}")
    return model


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    print(f"[INFO] Loading parquet data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    sequences, hard_labels, soft_labels, ctns, num_profiles = build_sequences_from_assignments(
        df,
        assignments_csv=ASSIGNMENTS_PATH,
    )

    model, history = train_discrete_profiler(
        sequences=sequences,
        hard_labels=hard_labels,
        soft_labels=soft_labels,
        ctns=ctns,
        num_profiles=num_profiles,
    )

    bank = build_prototype_bank(
        encoder=model.encoder,
        sequences=sequences,
        hard_labels=hard_labels,
        num_profiles=num_profiles,
    )

    assign_with_prototypes(
        encoder=model.encoder,
        bank=bank,
        sequences=sequences,
        ctns=ctns,
        out_csv="dataset/processed/prototype_profile_assignments.csv",
    )

    print("[DONE] Initial profiling pipeline complete.")
    print("[DONE] You can now add new behaviors with add_new_behavior_from_dataframe(...) or add_new_behavior_from_sequences(...).")


if __name__ == "__main__":
    main()