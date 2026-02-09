import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

# WGS84 constant
EARTH_RADIUS = 6378137.0  # meters


def llh_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convert Lat/Lon/Alt to local ENU coordinates (meters)
    using a tangent-plane approximation.
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)

    dlat = lat - lat0
    dlon = lon - lon0

    x = EARTH_RADIUS * dlon * np.cos(lat0)
    y = EARTH_RADIUS * dlat
    z = alt - alt0

    return x, y, z


def smooth_flight_data(df, window_length=9, polyorder=3):
    """
    Converts a single flight to ENU, smooths (if possible),
    and derives kinematic quantities.

    ENU conversion ALWAYS happens.
    Smoothing is applied only if the flight is long enough.
    """

    df = df.sort_values('Linear_Time').reset_index(drop=True)
    n = len(df)

    # --- ENU origin (first point) ---
    lat0 = df.loc[0, 'Latitude']
    lon0 = df.loc[0, 'Longitude']
    alt0 = df.loc[0, 'Final_Altitude']

    x, y, z = llh_to_enu(
        df['Latitude'].values,
        df['Longitude'].values,
        df['Final_Altitude'].values,
        lat0, lon0, alt0
    )

    # --- Estimate dt robustly ---
    t = df['Linear_Time'].values
    if n >= 2:
        dt = np.median(np.diff(t))
    else:
        dt = 1.0  # fallback, should not matter

    # --- Decide if SG smoothing is possible ---
    apply_sg = (
        n >= window_length and
        window_length % 2 == 1 and
        polyorder < window_length
    )

    if apply_sg:
        x_s = savgol_filter(x, window_length, polyorder)
        y_s = savgol_filter(y, window_length, polyorder)
        z_s = savgol_filter(z, window_length, polyorder)

        vx = savgol_filter(x, window_length, polyorder, deriv=1, delta=dt)
        vy = savgol_filter(y, window_length, polyorder, deriv=1, delta=dt)
        vz = savgol_filter(z, window_length, polyorder, deriv=1, delta=dt)
    else:
        # Fallback: finite differences
        x_s, y_s, z_s = x, y, z
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        vz = np.gradient(z, dt)

    # --- Speed derived from ENU velocity ---
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # --- Acceleration (from ENU speed) ---
    accel = np.gradient(speed, dt)

    # --- Turn rate ---
    course_rad = np.unwrap(np.deg2rad(df['Course'].values))
    turn_rate = np.gradient(course_rad, dt)

    # --- Assemble output ---
    df_out = df.copy()

    df_out['x'] = x_s
    df_out['y'] = y_s
    df_out['z'] = z_s

    df_out['vx'] = vx
    df_out['vy'] = vy
    df_out['vz'] = vz

    df_out['Speed'] = speed
    df_out['Acceleration'] = accel
    df_out['Turn_Rate'] = turn_rate

    df_out['sin_course'] = np.sin(course_rad)
    df_out['cos_course'] = np.cos(course_rad)

    return df_out


def process_all_flights(cleaned_csv_path, output_path):
    """
    Applies ENU conversion + smoothing to all flights
    WITHOUT resampling.
    """

    df = pd.read_csv(cleaned_csv_path)
    processed_flights = []

    for flight_id, flight_data in tqdm(
        df.groupby('Unique_Flight_ID'),
        total=df['Unique_Flight_ID'].nunique(),
        desc="Processing flights"
    ):
        # Fuse multi-radar measurements at same timestamp
        flight_data = (
            flight_data
            .sort_values('Linear_Time')
            .groupby('Linear_Time', as_index=False)
            .mean(numeric_only=True)
        )

        if flight_data.shape[0] < 2:
            continue


        smoothed = smooth_flight_data(flight_data)
        smoothed['Unique_Flight_ID'] = flight_id

        processed_flights.append(smoothed)

    final_df = pd.concat(processed_flights, ignore_index=True)
    final_df.to_csv(output_path, index=False)

    print(f"Processed ENU data saved to {output_path}")


if __name__ == '__main__':
    process_all_flights(
        'dataset/cleaned_data.csv',
        'dataset/processed_data.csv'
    )
