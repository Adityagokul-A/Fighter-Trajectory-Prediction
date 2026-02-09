import pandas as pd
import numpy as np

SECONDS_PER_DAY = 86400


def parse_seconds(time_str):
    """
    Parses non-standard time formats like '23 : 60 : 59' or '24:10:00'.
    Returns total seconds from 00:00:00 WITHOUT modulo normalization.
    """
    try:
        parts = [int(x) for x in time_str.replace(' ', '').split(':')]
        h, m, s = parts
        return h * 3600 + m * 60 + s
    except Exception:
        return np.nan


def process_altitude(row):
    """
    Use Geo Altitude if valid (>100), otherwise fall back to Barometric.
    """
    geo = row['Geo Altitude']
    baro = row['Barometric Altitude']

    if geo > 100:
        return geo
    return baro


def segment_flights(df, gap_threshold_seconds=1800):
    """
    Segments reused CTNs into distinct flights based on large time gaps.
    """
    df = df.sort_values(by='Linear_Time').copy()

    df['dt'] = df['Linear_Time'].diff()
    df['gap_flag'] = (df['dt'] > gap_threshold_seconds).astype(int)

    df['segment_id'] = df.groupby('CTN')['gap_flag'].cumsum()
    df['Unique_Flight_ID'] = df['CTN'] + "_" + df['segment_id'].astype(str)

    return df.drop(columns=['dt', 'gap_flag', 'segment_id'])


def clean_dataset(csv_path, output_path=None):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse TIME_IST into raw seconds
    print("Parsing timestamps...")
    df['Raw_Seconds'] = df['TIME_IST'].apply(parse_seconds)

    processed_dfs = []

    # Handle rollover PER AIRCRAFT, preserving row order
    for ctn, group in df.groupby('CTN', sort=False):
        group = group.copy()

        # Compute difference in original order
        prev_time = group['Raw_Seconds'].shift(1)

        # Rollover occurs when time goes backwards
        rollover = group['Raw_Seconds'] < prev_time

        # Cumulative count of rollovers
        rollover_count = rollover.cumsum()

        # Linear time construction
        group['Linear_Time'] = group['Raw_Seconds'] + rollover_count * SECONDS_PER_DAY

        processed_dfs.append(group)

    df_clean = pd.concat(processed_dfs, ignore_index=True)

    # Fix altitude dropouts
    print("Fixing altitude dropouts...")
    df_clean['Final_Altitude'] = df_clean.apply(process_altitude, axis=1)

    # Segment reused CTNs into flights
    print("Segmenting flights...")
    df_clean = segment_flights(df_clean)

    # Final schema
    final_cols = [
        'Unique_Flight_ID',
        'Linear_Time',
        'Latitude',
        'Longitude',
        'Final_Altitude',
        'Speed',
        'Course'
    ]

    df_final = df_clean[final_cols].copy()

    if output_path:
        df_final.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")

    return df_final


if __name__ == '__main__':
    df = clean_dataset(
        'dataset/Sample Data.csv',
        'dataset/cleaned_data.csv'
    )
