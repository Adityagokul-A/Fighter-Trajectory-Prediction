import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Sliding-window dataset for single-aircraft trajectory prediction.
    
    KEY FEATURES:
    1. Translation Invariance: Re-centers all spatial inputs relative to the 
       aircraft's position at the current timestep (t).
    2. Safe Normalization: Z-scores continuous variables but preserves 
       geometric properties of sin/cos.
    """

    def __init__(
        self, 
        processed_csv_path, 
        input_window=30, 
        pred_window=10, 
        normalize=False
    ):
        self.N = input_window
        self.k = pred_window
        self.normalize = normalize
        
        # Feature columns (ORDER MATTERS)
        # Indices: 
        # 0,1,2 -> x, y, z
        # 3,4,5 -> vx, vy, vz
        # 6 -> Speed
        # 7 -> Acceleration
        # 8,9 -> sin_course, cos_course (Do not normalize)
        # 10 -> Turn_Rate
        self.feature_cols = [
            'x', 'y', 'z', 
            'vx', 'vy', 'vz', 
            'Speed', 
            'Acceleration', 
            'sin_course', 'cos_course', 
            'Turn_Rate'
        ]
        
        # We predict future position deltas
        self.target_cols = ['x', 'y', 'z']

        print(f"Loading {processed_csv_path}...")
        df = pd.read_csv(processed_csv_path)
        
        self.flights = []
        self.index_map = [] 

        # Build per-flight arrays
        # grouping by Unique_Flight_ID ensures we don't slide a window across two different jets
        for _, flight_df in df.groupby('Unique_Flight_ID'):
            flight_df = flight_df.sort_values('Linear_Time')
            
            # Skip short flights
            if len(flight_df) < self.N + self.k:
                continue
                
            X = flight_df[self.feature_cols].values.astype(np.float32)
            Y = flight_df[self.target_cols].values.astype(np.float32)
            
            flight_idx = len(self.flights)
            self.flights.append((X, Y))
            
            # Create valid start indices for sliding window
            # T is total timesteps in this flight
            T = len(flight_df)
            # We need [start : start + N + k] to be valid
            for start in range(T - self.N - self.k + 1):
                self.index_map.append((flight_idx, start))
                
        # Compute normalization stats (if enabled)
        if self.normalize:
            self._compute_normalization()
            
        print(f"[Dataset] Flights used: {len(self.flights)}")
        print(f"[Dataset] Total samples: {len(self.index_map)}")

    def _compute_normalization(self):
        """
        Computes mean/std for Z-score normalization.
        CRITICAL: Manually sets mean=0, std=1 for sin/cos columns (indices 8,9)
        to prevent destroying their unit-circle properties.
        """
        # Concatenate all flight data to compute global stats
        all_X = np.concatenate([f[0] for f in self.flights], axis=0)
        
        self.mean = all_X.mean(axis=0)
        self.std = all_X.std(axis=0) + 1e-6 # Avoid div by zero
        
        # Indices 8 (sin) and 9 (cos) should NOT be scaled
        # (x - 0) / 1 = x
        self.mean[8:10] = 0.0
        self.std[8:10] = 1.0

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        flight_idx, start = self.index_map[idx]
        X_full, Y_full = self.flights[flight_idx]
        
        x_past = X_full[start : start + self.N].copy()
        y_future = Y_full[start + self.N : start + self.N + self.k].copy()
        
        # 1. Translation Invariance (Zero-Centering)
        current_pos = x_past[-1, 0:3]
        x_past[:, 0:3] -= current_pos
        y_future -= current_pos

        # 2. Normalization (Scaling)
        if self.normalize:
            # Scale Inputs: (x - mean) / std
            x_past = (x_past - self.mean) / self.std
            
            # --- NEW FIX: Scale Targets ---
            # We must scale the targets by the same spatial Std Dev 
            # so the model predicts "normalized units" instead of "meters".
            # indices 0,1,2 are x,y,z
            spatial_std = self.std[0:3] 
            y_future = y_future / spatial_std

        return (
            torch.from_numpy(x_past), 
            torch.from_numpy(y_future)
        )