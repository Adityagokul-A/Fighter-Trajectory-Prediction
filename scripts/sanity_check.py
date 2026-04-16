import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
FILE_PATH = "dataset/raw/Combined Data.csv"
OUTPUT_PATH = "dataset/processed/smoothed_kinematic_trajectories.parquet"

GAP_SECONDS = 15
MIN_POINTS_THRESHOLD = 10 

# SG Filter settings
WINDOW_LENGTH = 9
POLY_ORDER = 3
EARTH_RADIUS = 6378137.0  

# Physics Thresholds
MAX_SPEED = 1000.0  # m/s (~Mach 3)
MAX_ACCEL = 150.0   # m/s^2 (~15 Gs)

# ==========================================
# LOGGING HELPER
# ==========================================
def log_stage(name, df):
    if df is None:
        print(f"{name:<35} | DataFrame is None")
        return
    rows = len(df)
    segments = df['CTN_New'].nunique() if 'CTN_New' in df.columns else "N/A"
    print(f"{name:<35} | Rows: {rows:>10} | Segments: {segments}")

# ==========================================
# PHASE 1: DATA CLEANING & SEGMENTATION
# ==========================================
def process_time(df):
    hms = df['TIME_IST'].str.replace(' ', '', regex=False)\
                        .str.split(':', expand=True).astype(int)
    h = hms[0].where(hms[0] != 1, 25)
    df['TIME'] = (h - 1) * 3600 + (hms[1] - 1) * 60 + hms[2]
    return df.drop(columns='TIME_IST')

def resolve_zero_dt(df):
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['CTN', 'TIME'], keep='first').copy()
    return df

def fuse_altitudes(df):
    geo_clean = df['Geo Altitude'].replace(0, pd.NA)
    df['Final_Altitude'] = geo_clean.fillna(df['Barometric Altitude'])
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * EARTH_RADIUS

def segment_trajectories(df, gap_seconds, max_speed_threshold):
    df_sorted = df.sort_values(by=["CTN", "TIME"]).copy()

    time_diff = df_sorted.groupby("CTN")["TIME"].diff()
    df_sorted["dt"] = time_diff

    lat_shifted = df_sorted.groupby("CTN")['Latitude'].shift(1)
    lon_shifted = df_sorted.groupby("CTN")['Longitude'].shift(1)

    distance_meters = haversine_distance(
        lat_shifted, lon_shifted,
        df_sorted['Latitude'], df_sorted['Longitude']
    )

    raw_speed = distance_meters / time_diff

    is_time_gap = (time_diff > gap_seconds) | time_diff.isna()
    is_spatial_gap = (raw_speed > max_speed_threshold)
    is_new_flight = is_time_gap | is_spatial_gap

    segment_number = is_new_flight.groupby(df_sorted["CTN"]).cumsum()
    df_sorted["CTN_New"] = df_sorted["CTN"].astype(str) + "_" + segment_number.astype(str)

    df_sorted.loc[is_new_flight, 'dt'] = pd.NA

    print(f"   -> Sliced {is_spatial_gap.sum()} trajectories due to sensor teleportation.")
    return df_sorted

def filter_short_trajectories(df, min_points):
    counts = df['CTN_New'].value_counts()
    valid_segments = counts[counts >= min_points].index
    return df[df['CTN_New'].isin(valid_segments)].copy()

# ==========================================
# PHASE 2: RESAMPLING & INTERPOLATION
# ==========================================
def resample_trajectories(df):
    if 'dt' in df.columns:
        df = df.drop(columns=['dt'])

    df = df.copy()
    df['timedelta'] = pd.to_timedelta(df['TIME'], unit='s')
    df.set_index('timedelta', inplace=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    resampled = df.groupby('CTN_New')[numeric_cols].resample('1s').mean()
    resampled = resampled.reset_index(level='CTN_New')

    interpolated_cols = resampled.groupby('CTN_New')[numeric_cols].transform(
        lambda x: x.interpolate(method='pchip')
    )
    resampled[numeric_cols] = interpolated_cols

    interpolated = resampled.reset_index()
    interpolated['TIME'] = interpolated['timedelta'].dt.total_seconds().astype(int)
    interpolated['CTN'] = interpolated['CTN_New'].str.rsplit('_', n=1).str[0]
    interpolated['dt'] = 1.0

    return interpolated.drop(columns=['timedelta'])

# ==========================================
# PHASE 3: KINEMATICS & DERIVATIVES
# ==========================================
def llh_to_enu_vectorized(lat, lon, alt, lat0, lon0, alt0):
    lat_rad, lon_rad = np.deg2rad(lat), np.deg2rad(lon)
    lat0_rad, lon0_rad = np.deg2rad(lat0), np.deg2rad(lon0)

    x = EARTH_RADIUS * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS * (lat_rad - lat0_rad)
    z = alt - alt0

    return x, y, z

def engineer_kinematic_features(df):
    df['lat0'] = df.groupby('CTN_New')['Latitude'].transform('first')
    df['lon0'] = df.groupby('CTN_New')['Longitude'].transform('first')
    df['alt0'] = df.groupby('CTN_New')['Final_Altitude'].transform('first')

    df['x'], df['y'], df['z'] = llh_to_enu_vectorized(
        df['Latitude'], df['Longitude'], df['Final_Altitude'],
        df['lat0'], df['lon0'], df['alt0']
    )

    df = df.drop(columns=['lat0', 'lon0', 'alt0'])

    def apply_sg(g):
        g = g.copy()

        if len(g) < WINDOW_LENGTH:
            return g

        # Position smoothing
        g['x_s'] = savgol_filter(g['x'], WINDOW_LENGTH, POLY_ORDER)
        g['y_s'] = savgol_filter(g['y'], WINDOW_LENGTH, POLY_ORDER)
        g['z_s'] = savgol_filter(g['z'], WINDOW_LENGTH, POLY_ORDER)

        # Velocity
        g['vx'] = savgol_filter(g['x'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['vy'] = savgol_filter(g['y'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['vz'] = savgol_filter(g['z'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)

        # Acceleration
        g['ax'] = savgol_filter(g['vx'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['ay'] = savgol_filter(g['vy'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)
        g['az'] = savgol_filter(g['vz'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0)

        # Scalar values
        g['ENU_Speed'] = np.sqrt(g['vx']**2 + g['vy']**2 + g['vz']**2)
        g['Acceleration'] = savgol_filter(
            g['ENU_Speed'], WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0
        )

        # Course features
        if 'Course' in g.columns:
            course_rad_unwrapped = np.unwrap(
                np.deg2rad(g['Course'].fillna(method='ffill').fillna(0))
            )
            g['Turn_Rate'] = savgol_filter(
                course_rad_unwrapped, WINDOW_LENGTH, POLY_ORDER, deriv=1, delta=1.0
            )
            g['sin_course'] = np.sin(course_rad_unwrapped)
            g['cos_course'] = np.cos(course_rad_unwrapped)
        else:
            g['Turn_Rate'] = 0.0
            g['sin_course'] = 0.0
            g['cos_course'] = 0.0

        return g

    return df.groupby('CTN_New', group_keys=False).apply(apply_sg)

# ==========================================
# MAIN EXECUTION
# ==========================================
def process_and_save():
    print("\n========== PIPELINE START ==========\n")

    # STEP 1: LOAD + CLEAN
    print("1. Loading and cleaning data...")
    df = pd.read_csv(FILE_PATH)
    log_stage("After load", df)

    if "CTN" in df.columns:
        df = df[~df["CTN"].str.contains("aor_crossing_flag", na=False)]

    df = process_time(df)
    df = fuse_altitudes(df)
    df = resolve_zero_dt(df)
    log_stage("After cleaning", df)

    # STEP 2: SEGMENTATION
    print("\n2. Segmenting trajectories...")
    df_segmented = segment_trajectories(df, GAP_SECONDS, MAX_SPEED)
    log_stage("After segmentation", df_segmented)

    df_filtered = filter_short_trajectories(df_segmented, MIN_POINTS_THRESHOLD)
    log_stage("After filtering short traj", df_filtered)

    # STEP 3: RESAMPLING
    print("\n3. Resampling to 1s Grid...")
    df_resampled = resample_trajectories(df_filtered)
    log_stage("After resampling", df_resampled)

    # STEP 4: KINEMATICS
    print("\n4. Deriving Kinematics...")
    df_processed = engineer_kinematic_features(df_resampled)
    df_processed = df_processed.dropna(subset=['vx', 'vy', 'vz'])
    log_stage("After kinematics", df_processed)

    # STEP 5: PHYSICS FILTER
    print("\n5. Physics-based filtering...")
    invalid_speed = df_processed[df_processed['ENU_Speed'] > MAX_SPEED]['CTN_New'].unique()
    invalid_accel = df_processed[df_processed['Acceleration'].abs() > MAX_ACCEL]['CTN_New'].unique()

    corrupted_ids = set(invalid_speed).union(set(invalid_accel))
    df_clean = df_processed[~df_processed['CTN_New'].isin(corrupted_ids)]

    log_stage("Final clean data", df_clean)

    print(f"\n[SUMMARY] Saved {df_clean['CTN_New'].nunique()} pristine flight segments.")

    df_clean.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
    print(f"[SUCCESS] Data written to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_and_save()