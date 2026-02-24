import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "dataset/raw/Sample Data.csv"
GAP_SECONDS = 15
MIN_POINTS_THRESHOLD = 10 # Updated to 10 to ensure stable cubic interpolation

def process_time(df):
    """
    Converts 'HH:MM:SS' string into total seconds from 00:00:00.
    Handles irregular 24->01 hour rollover by mapping 01 to 25.
    """  
    hms = (
        df['TIME_IST']
        .str.replace(' ', '', regex=False)
        .str.split(':', expand=True)
        .astype(int)
    )

    h = hms[0].where(hms[0] != 1, 25) 
    df['TIME'] = (h - 1) * 3600 + (hms[1] - 1) * 60 + hms[2]

    return df.drop(columns='TIME_IST')

def resolve_zero_dt(df):
    """
    Cleans up duplicate timestamps caused by exact system duplicates 
    and sub-second sampling truncation.
    """
    initial_len = len(df)
    
    df = df.drop_duplicates()
    exact_dropped = initial_len - len(df)
    
    df = df.drop_duplicates(subset=['CTN', 'TIME'], keep='first').copy()
    subsecond_dropped = (initial_len - exact_dropped) - len(df)
    
    print("\n--- Cleaning dt=0 Records ---")
    print(f"Exact duplicate rows removed: {exact_dropped}")
    print(f"Sub-second truncated rows removed: {subsecond_dropped}")
    print(f"Total rows remaining: {len(df)}")
    print("-----------------------------\n")
    
    return df
    
def segment_trajectories(df, gap_seconds=15):
    """
    Sorts by aircraft and time, calculates time difference (dt), 
    and assigns a new segment ID if the gap exceeds the threshold.
    """
    df_sorted = df.sort_values(by=["CTN", "TIME"]).copy()
    time_diff = df_sorted.groupby("CTN")["TIME"].diff()
    
    df_sorted["dt"] = time_diff
    
    is_new_flight = (time_diff > gap_seconds) | time_diff.isna()
    segment_number = is_new_flight.groupby(df_sorted["CTN"]).cumsum()
    df_sorted["CTN_New"] = df_sorted["CTN"].astype(str) + "_" + segment_number.astype(str)
    
    df_sorted.loc[is_new_flight, 'dt'] = pd.NA 
    
    return df_sorted

def diagnose_zero_dt(df):
    print("\n--- Running dt = 0.0 Diagnostics ---")
    same_time_mask = df.duplicated(subset=['CTN', 'TIME'], keep=False)
    zero_dt_subset = df[same_time_mask].sort_values(by=['CTN', 'TIME'])
    print(f"Total rows sharing the exact same CTN and TIME (Should be 0): {len(zero_dt_subset)}")
    print("------------------------------------\n")

def print_trajectory_length_bins(df):
    counts = df['CTN_New'].value_counts()
    max_points = counts.max()
    
    print("\n--- Zoomed In: 0 to 100 Points ---")
    print(f"{'Range (Points)':<20} | {'Number of Trajectories'}")
    print("-" * 45)
    bins_small = list(range(0, 101, 10))
    binned_small = pd.cut(counts, bins=bins_small, right=False).value_counts().sort_index()
    
    for interval, count in binned_small.items():
        if count > 0:
            print(f"{str(interval):<20} | {count}")
            
    print("\n--- The Long Tail: 100+ Points ---")
    bins_large = list(range(100, int(max_points) + 100, 100))
    
    if len(bins_large) > 1:
        binned_large = pd.cut(counts, bins=bins_large, right=False).value_counts().sort_index()
        for interval, count in binned_large.items():
            if count > 0:
                print(f"{str(interval):<20} | {count}")
    print("-" * 45 + "\n")

def analyze_sampling_frequency(df):
    dt_counts = df.groupby(['CTN_New', 'dt']).size().reset_index(name='count')
    dominant_dt = dt_counts.sort_values('count', ascending=False).drop_duplicates('CTN_New')
    summary = dominant_dt['dt'].value_counts().sort_index()
    
    print("\n--- Dominant Sampling Frequency per Trajectory ---")
    print(f"{'Dominant dt (seconds)':<25} | {'Number of Trajectories'}")
    print("-" * 50)
    
    for dt_val, count in summary.items():
        print(f"{str(dt_val) + 's':<25} | {count}")
    print("-" * 50 + "\n")
    
    return dominant_dt

def plot_histograms(df, min_points):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    df['dt'].dropna().clip(upper=20).hist(bins=20, ax=ax1, color='#4C72B0', edgecolor='black')
    ax1.set_title("Sampling Intervals (dt)")
    ax1.set_xlabel("Seconds between points")
    ax1.set_ylabel("Frequency")

    counts = df['CTN_New'].value_counts()
    counts.clip(upper=500).hist(bins=50, ax=ax2, color='#C44E52', edgecolor='black')
    ax2.axvline(min_points, color='black', linestyle='--', linewidth=2, label=f'Threshold ({min_points})')
    ax2.set_title("Points per Segment")
    ax2.set_xlabel("Number of Points")
    ax2.set_ylabel("Number of Trajectories")
    ax2.legend()

    plt.tight_layout()
    plt.show()

def filter_short_trajectories(df, min_points):
    counts = df['CTN_New'].value_counts()
    valid_segments = counts[counts >= min_points].index
    
    df_filtered = df[df['CTN_New'].isin(valid_segments)].copy()
    
    print(f"\n--- Filtering Results ---")
    print(f"Original Segments: {len(counts)}")
    print(f"Segments Kept: {len(valid_segments)}")
    print(f"Segments Dropped: {len(counts) - len(valid_segments)}")
    
    return df_filtered

def resample_trajectories(df):
    """
    Resamples all valid trajectories to a strict 1-second grid.
    Uses fast, vectorized cubic interpolation safely bounded within each flight.
    """
    print("\n--- Resampling Trajectories to 1s Intervals ---")
    
    if 'dt' in df.columns:
        df = df.drop(columns=['dt'])
        
    df = df.copy()
    df['timedelta'] = pd.to_timedelta(df['TIME'], unit='s')
    df.set_index('timedelta', inplace=True)
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    # 1. Resample (This creates a MultiIndex of CTN_New and timedelta)
    resampled = df.groupby('CTN_New')[numeric_cols].resample('1s').mean()
    
    # 2. Flatten ONLY the 'CTN_New' level of the index.
    # This leaves 'timedelta' as the true index, which SciPy NEEDS for cubic math.
    resampled = resampled.reset_index(level='CTN_New')
    
    # 3. Vectorized Transform
    # Applies the cubic spline column-by-column in C, rather than dataframe-by-dataframe in Python.
    interpolated_cols = resampled.groupby('CTN_New')[numeric_cols].transform(
        lambda x: x.interpolate(method='cubic')
    )
    
    # Replace the NaN columns with our newly interpolated data
    resampled[numeric_cols] = interpolated_cols
    
    # 4. Clean up the dataframe structure
    interpolated = resampled.reset_index()
    interpolated['TIME'] = interpolated['timedelta'].dt.total_seconds().astype(int)
    interpolated['CTN'] = interpolated['CTN_New'].str.rsplit('_', n=1).str[0]
    interpolated['dt'] = 1.0
    
    print(f"Original row count (before resampling): {len(df)}")
    print(f"New row count (1s grid interpolated): {len(interpolated)}")
    print("-----------------------------------------------\n")
    
    return interpolated.drop(columns=['timedelta'])


if __name__ == '__main__':
    print("Loading data...")
    df = pd.read_csv(FILE_PATH) 
    
    df = df[~df["CTN"].str.contains("aor_crossing_flag", na=False)]
    df = process_time(df)
    df = resolve_zero_dt(df)

    print("Segmenting trajectories...")
    df_segmented = segment_trajectories(df, gap_seconds=GAP_SECONDS)
    
    diagnose_zero_dt(df_segmented)
    print_trajectory_length_bins(df_segmented)
    analyze_sampling_frequency(df_segmented)
    
    print("Generating plots (close window to continue script)...")
    plot_histograms(df_segmented, min_points=MIN_POINTS_THRESHOLD)

    # Filter out the short segments (now requiring 10 points)
    df_filtered = filter_short_trajectories(df_segmented, min_points=MIN_POINTS_THRESHOLD)
    
    # Resample everything to a 1-second grid using cubic interpolation
    df_resampled = resample_trajectories(df_filtered)
    
    # Display final sanity check
    print(df_resampled[['CTN_New', 'TIME', 'dt']].head(10))