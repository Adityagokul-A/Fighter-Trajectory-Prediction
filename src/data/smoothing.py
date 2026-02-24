import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# --- CONFIGURATION ---
INPUT_PATH = "dataset/processed/resampled_trajectories.parquet"
OUTPUT_PATH = "dataset/processed/smoothed_kinematic_trajectories.parquet"

COL_LAT = 'Latitude'
COL_LON = 'Longitude'
COL_ALT = 'Final_Altitude'
COL_COURSE = 'Course'
COL_SPEED = 'Speed' # Sensor speed 

# SG Filter settings
WINDOW_LENGTH = 9
POLY_ORDER = 3
EARTH_RADIUS = 6378137.0  # meters

def llh_to_enu_vectorized(lat, lon, alt, lat0, lon0, alt0):
    """
    Vectorized conversion of Lat/Lon/Alt to local ENU coordinates (meters).
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    x = EARTH_RADIUS * dlon * np.cos(lat0_rad)
    y = EARTH_RADIUS * dlat
    z = alt - alt0

    return x, y, z

def engineer_kinematic_features(df):
    print(f"\n--- Loading data from {INPUT_PATH} ---")
    
    # 1. Establish the Local Tangent Plane Origin (First point of each flight)
    # Using transform('first') broadcasts the origin values to every row of that specific flight
    print("Converting to ENU Coordinates...")
    df['lat0'] = df.groupby('CTN_New')[COL_LAT].transform('first')
    df['lon0'] = df.groupby('CTN_New')[COL_LON].transform('first')
    df['alt0'] = df.groupby('CTN_New')[COL_ALT].transform('first')
    
    # Vectorized ENU Math (Blazing fast, no loops)
    df['x'], df['y'], df['z'] = llh_to_enu_vectorized(
        df[COL_LAT], df[COL_LON], df[COL_ALT],
        df['lat0'], df['lon0'], df['alt0']
    )
    
    # Drop the temporary origin columns
    df = df.drop(columns=['lat0', 'lon0', 'alt0'])

    # 2. Apply Savitzky-Golay Smoothing & Derivatives
    print("Applying Savitzky-Golay Smoothing and extracting velocity...")
    
    # Define a helper function to keep the groupby apply clean
    def apply_sg(group):
        g = group.copy()
        
        # Smooth absolute positions
        g['x_s'] = savgol_filter(g['x'], WINDOW_LENGTH, POLY_ORDER)
        g['y_s'] = savgol_filter(g['y'], WINDOW_LENGTH, POLY_ORDER)
        g['z_s'] = savgol_filter(g['z'], WINDOW_LENGTH, POLY_ORDER)
        
        # Extract Velocity directly from the polynomial derivative (dt=1.0)
        g['vx'] = savgol_filter(g['x'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['vy'] = savgol_filter(g['y'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['vz'] = savgol_filter(g['z'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)

        # Extract Directional Acceleration (2nd derivative of position / 1st of velocity)
        g['ax'] = savgol_filter(g['vx'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['ay'] = savgol_filter(g['vy'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['az'] = savgol_filter(g['vz'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        
        # Internally consistent ENU Speed and Acceleration
        g['ENU_Speed'] = np.sqrt(g['vx']**2 + g['vy']**2 + g['vz']**2)
        g['Acceleration'] = savgol_filter(g['ENU_Speed'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        
        # Unwrapped Course and Turn Rate
        # Unwrapping prevents massive artificial turn rates when crossing 360 -> 0 degrees
        course_rad_unwrapped = np.unwrap(np.deg2rad(g[COL_COURSE]))
        g['Turn_Rate'] = savgol_filter(course_rad_unwrapped, WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        
        # Cyclical Course Features (NNs love this)
        g['sin_course'] = np.sin(course_rad_unwrapped)
        g['cos_course'] = np.cos(course_rad_unwrapped)
        
        return g

    # Apply the math per flight trajectory
    df_out = df.groupby('CTN_New', group_keys=False).apply(apply_sg)

    return df_out

if __name__ == '__main__':
    # Load Parquet
    df = pd.read_parquet(INPUT_PATH)
    
    df_processed = engineer_kinematic_features(df)
    
    print(f"\n--- Saving processed kinematics to {OUTPUT_PATH} ---")
    df_processed.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
    
    print("\n--- Summary of New Features ---")
    new_cols = ['x_s', 'y_s', 'z_s', 'vx', 'vy', 'vz','ax', 'ay', 'az' ,'ENU_Speed', 'Acceleration', 'Turn_Rate', 'sin_course', 'cos_course']
    print(df_processed[['CTN_New', 'TIME'] + new_cols].head())
