# Fighter Aircraft Trajectory Prediction

A deep learning framework for predicting future trajectories of fighter aircraft using kinematic history and **latent maneuver recognition**.  
The aircraft is modeled as a **hybrid dynamical system**, separating continuous motion from discrete latent maneuvers.

---

## 1. Design Summary

### 1.1 Input

#### Raw Input Data (per aircraft, per timestep)
Derived from ADS-B–like logs:

- **CTN**: Unique aircraft ID (handling reuse and long gaps)
- **TIME_IST**: Timestamp
- **Latitude**, **Longitude**
- **Geo Altitude**
- **Speed**
- **Course**

#### Preprocessing

- **Temporal Resampling**: Fixed interval (10 s)
- **Savitzky–Golay Filtering**:
  - Smooths raw position signals
  - Derives clean kinematic quantities (Acceleration, Turn Rate)
- **Coordinate Transform**:
  - WGS84 (Lat/Lon) → Local **ENU** (East–North–Up)
- **Translation Invariance**:
  - Spatial origin reset to aircraft position at current time \( t \)

#### Final Input Features (\(N \times 11\))

\[
[x, y, z, v_x, v_y, v_z, \text{speed}, \text{acceleration}, \sin(\text{course}), \cos(\text{course}), \text{turn\_rate}]
\]

---

### 1.2 Problem Statement

Given the last \( N \) observed kinematic states of a fighter aircraft, predict its future trajectory over the next \( k \) timesteps.

**Constraints**
- No flight plans, waypoints, or cooperative intent signals

**Key Challenge**
- Future motion depends on **latent maneuvers** (e.g., high-G turn vs. cruise) that are not directly observed.

---

### 1.3 Solution Architecture

**Latent-Regime–Conditioned Seq2Seq Model**

- **Encoder (GRU)**  
  Consumes past kinematic history (\( N \) steps)

- **Regime Head**  
  Infers a soft probability distribution over \( K \) latent maneuver regimes

- **Decoder (GRU)**  
  Autoregressively predicts future motion deltas, conditioned on the regime embedding

---

### 1.4 Training Objective

The model is trained end-to-end using a composite loss:

- **Position Loss (MSE)**  
  Ensures accurate future trajectory prediction

- **Smoothness Loss**  
  Penalizes physically implausible jerk / acceleration oscillations

- **Entropy Regularization**  
  Maximizes **batch-level regime diversity** (mutual information) to prevent regime collapse

---

## 2. Project Structure
```text

fighter_traj_pred/
├── configs/               # YAML configuration files
│   ├── model_config.yaml  # Model hyperparameters (layers, hidden_dim)
│   └── train_config.yaml  # Training parameters (lr, epochs, loss weights)
├── data/                  # Data storage
│   ├── raw/               # Original CSV logs
│   └── processed/         # Smoothed ENU tensors
├── src/
│   ├── data/              # Preprocessing, Smoothing, and Dataset class
│   ├── models/            # Encoder, Decoder, and RegimeHead modules
│   └── training/          # Loss functions and Training loop
├── scripts/
│   ├── train.py           # Main training entry point
│   └── inference.py       # Visualization and testing script
└── requirements.txt

```
---

## 3. Usage

This section describes how to set up the environment, preprocess data, train the model, and visualize predictions. All experiments are **configuration-driven** via YAML files—no code changes are required to modify hyperparameters.


### 3.1 Installation

Install all required Python dependencies:

```bash
pip install -r requirements.txt


