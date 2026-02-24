import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HybridTrajectoryDataset(Dataset):
    """
    Trajectory Dataset supporting variable lengths.
    Uses sliding windows for long flights and a 70/30 split for short flights.
    """
    def __init__(self, parquet_path, input_window=30, pred_window=10, normalize=True):
        self.N = input_window
        self.k = pred_window
        self.normalize = normalize
        
        # 14 Input Features
        self.feature_cols = [
            'x_s', 'y_s', 'z_s', 
            'vx', 'vy', 'vz', 
            'ax', 'ay', 'az',
            'ENU_Speed', 'Acceleration', 
            'sin_course', 'cos_course', 'Turn_Rate'
        ]
        
        # Physics-informed Autoregressive Target (Positions only)
        self.target_cols = ['x_s', 'y_s', 'z_s']

        # Dynamic Indices
        self.pos_idx = [self.feature_cols.index(c) for c in ['x_s', 'y_s', 'z_s']]
        self.angle_idx = [self.feature_cols.index(c) for c in ['sin_course', 'cos_course']]
        self.kin_idx = [i for i in range(len(self.feature_cols)) if i not in self.pos_idx + self.angle_idx]

        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        self.flights = []
        # Stores: (flight_idx, start_idx, input_length, target_length)
        self.index_map = [] 

        for _, flight_df in df.groupby('CTN_New'):
            flight_df = flight_df.sort_values('TIME')
            T = len(flight_df)
            
            # Minimum points threshold check (File 1 ensures T >= 10)
            if T < 10: 
                continue
                
            X = flight_df[self.feature_cols].values.astype(np.float32)
            Y = flight_df[self.target_cols].values.astype(np.float32)
            
            flight_idx = len(self.flights)
            self.flights.append((X, Y))
            
            # LONG FLIGHTS: Sliding Windows
            if T >= self.N + self.k:
                for start in range(T - self.N - self.k + 1):
                    self.index_map.append((flight_idx, start, self.N, self.k))
            
            # SHORT FLIGHTS: Single 70/30 Split
            else:
                l_in = int(T * 0.7)
                l_in = min(l_in, self.N)
                l_tgt = T - l_in
                l_tgt = min(l_tgt, self.k)
                self.index_map.append((flight_idx, 0, l_in, l_tgt))
                
        if self.normalize:
            self._compute_normalization()
            
    def _compute_normalization(self):
        num_features = len(self.feature_cols)
        self.mean = np.zeros(num_features, dtype=np.float32)
        self.std = np.ones(num_features, dtype=np.float32)
        
        # All valid data across all flights
        all_X = np.concatenate([f[0] for f in self.flights], axis=0)
        global_mean = all_X.mean(axis=0)
        global_std = all_X.std(axis=0) + 1e-6
        
        # A. Kinematics (Global Normalization)
        for i in self.kin_idx:
            self.mean[i] = global_mean[i]
            self.std[i] = global_std[i]
            
        # B. Positions (Local Displacement Normalization)
        print("Sampling windows to compute local position stats...")
        sample_size = min(5000, len(self.index_map))
        sample_indices = np.random.choice(len(self.index_map), size=sample_size, replace=False)
        
        local_displacements = []
        for idx in sample_indices:
            f_idx, start, l_in, _ = self.index_map[idx]
            X_full, _ = self.flights[f_idx]
            
            # Extract only the valid historical window
            window_pos = X_full[start : start + l_in, self.pos_idx]
            center_pos = window_pos[-1] # Anchor to last known point
            local_displacements.append(window_pos - center_pos)
            
        local_displacements = np.concatenate(local_displacements, axis=0)
        self.mean[self.pos_idx] = 0.0 
        self.std[self.pos_idx] = local_displacements.std(axis=0) + 1e-6
        
        # C. Angles (Passthrough: Mean 0, Std 1)
        self.mean[self.angle_idx] = 0.0
        self.std[self.angle_idx] = 1.0
        
        print(f"[Dataset] Normalization Computed. Active Samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        flight_idx, start, l_in, l_tgt = self.index_map[idx]
        X_full, Y_full = self.flights[flight_idx]
        
        # Extract Valid Data
        x_raw = X_full[start : start + l_in].copy()
        y_raw = Y_full[start + l_in : start + l_in + l_tgt].copy()
        
        # Translation Invariance (Anchor to x_raw's last point)
        current_pos = x_raw[-1, self.pos_idx]
        x_raw[:, self.pos_idx] -= current_pos
        y_raw -= current_pos 

        if self.normalize:
            x_raw = (x_raw - self.mean) / self.std
            y_raw = y_raw / self.std[self.pos_idx]

        # Initialize Padded Tensors
        x_padded = np.zeros((self.N, len(self.feature_cols)), dtype=np.float32)
        y_padded = np.zeros((self.k, 3), dtype=np.float32)
        target_mask = np.zeros(self.k, dtype=np.float32)
        
        # Apply Post-Padding
        x_padded[:l_in, :] = x_raw
        y_padded[:l_tgt, :] = y_raw
        target_mask[:l_tgt] = 1.0 # 1 for valid targets, 0 for padded

        return (
            torch.from_numpy(x_padded), 
            torch.from_numpy(y_padded), 
            torch.tensor(l_in, dtype=torch.long), 
            torch.from_numpy(target_mask)
        )