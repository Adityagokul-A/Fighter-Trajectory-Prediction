Fighter Aircraft Trajectory Prediction

A Deep Learning framework for predicting future trajectories of fighter aircraft using kinematic history and latent maneuver recognition. This project models the aircraft as a hybrid dynamical system, separating continuous motion from discrete latent maneuvers.

1. Design Summary

1.1 Input

Raw Input Data (per aircraft, per timestep):
Derived from ADS-B–like logs:

CTN: Unique aircraft ID (handling reuse/gaps).

TIME_IST: Timestamp.

Latitude, Longitude, Geo Altitude, Speed, Course.

Preprocessing:

Temporal resampling to a fixed interval (10s).

Savitzky-Golay Filtering: Used to smooth raw position data and derive clean derivatives (Acceleration, Turn Rate).

Coordinate Transform: WGS84 (Lat/Lon) $\rightarrow$ Local ENU (East-North-Up).

Translation Invariance: Origin reset to aircraft position at current time $t$.

Final Input Features ($N \times 11$):


$$[x, y, z, v_x, v_y, v_z, \text{speed}, \text{acceleration}, \sin(\text{course}), \cos(\text{course}), \text{turn\_rate}]$$

1.2 Problem Statement

Given the last $N$ observed kinematic states, predict the future trajectory over the next $k$ timesteps.

Constraint: No flight plans or cooperative intent signals.

Challenge: Future motion depends on latent maneuvers (e.g., High-G turn vs. Cruise) which are unobserved.

1.3 Solution Architecture

Latent-Regime Conditioned Seq2Seq:

Encoder (GRU): Consumes past kinematic history ($N$ steps).

Regime Head: Infers a soft probability distribution over $K$ latent regimes.

Decoder (GRU): Autoregressively predicts future motion deltas, conditioned on the regime embedding.

1.4 Training Objective

The model is trained end-to-end using a composite loss function:

Position Loss (MSE): Accuracy of future trajectory.

Smoothness Loss: Penalizes physically impossible jerk/acceleration changes.

Entropy Regularization: Maximizes batch diversity (Mutual Information) to prevent regime collapse.

2. Project Structure

fighter_traj_pred/
├── config/               # YAML configuration files
│   ├── model_config.yaml  # Model hyperparameters (layers, hidden_dim)
│   └── train_config.yaml  # Training parameters (lr, epochs, loss weights)
|
├── dataset/                  # Data storage
|
├── src/
│   ├── data/              # Preprocessing, Smoothing, and Dataset class
│   ├── models/            # Encoder, Decoder, and RegimeHead modules
│   └── training/          # Loss functions and Training loop
├── scripts/
│   ├── train.py           # Main training entry point
│   └── inference.py       # Visualization and testing script
└── requirements.txt


3. Usage

Installation

pip install -r requirements.txt


Data Preparation

Place your raw logs in data/raw/ and run the smoothing pipeline:

# Runs the Savitzky-Golay filter and ENU transformation
python -m src.data.smoothing


Training

To train the model using parameters defined in configs/train_config.yaml:

python -m scripts.train


Visualization

To generate trajectory plots from the validation set:

python -m scripts.inference


4. Acknowledgments & Attribution

This project had the assistance of Google Gemini and ChatGPT for editing scripts.