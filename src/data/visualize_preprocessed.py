import pandas as pd
import matplotlib.pyplot as plt

def visualize_flight(processed_csv_path, flight_id=None):
    df = pd.read_csv(processed_csv_path)

    # Pick a flight if not specified

    print(df['Unique_Flight_ID'].unique()[:600])

    if flight_id is None:
        flight_id = df['Unique_Flight_ID'].iloc[70000]

    flight = (
        df[df['Unique_Flight_ID'] == flight_id]
        .sort_values('Linear_Time')
        .reset_index(drop=True)
    )

    t = flight['Linear_Time'] - flight['Linear_Time'].iloc[0]

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Sanity Check – Flight {flight_id}", fontsize=16)

    # 1. ENU Trajectory
    axs[0, 0].plot(flight['x'], flight['y'])
    axs[0, 0].set_title("ENU Trajectory (x vs y)")
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].axis('equal')

    # 2. Altitude
    axs[0, 1].plot(t, flight['z'])
    axs[0, 1].set_title("Altitude vs Time")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("z [m]")

    # 3. Speed
    axs[1, 0].plot(t, flight['Speed'])
    axs[1, 0].set_title("Speed vs Time")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Speed")

    # 4. Acceleration
    axs[1, 1].plot(t, flight['Acceleration'])
    axs[1, 1].set_title("Acceleration vs Time")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Acceleration")

    # 5. Turn Rate
    axs[2, 0].plot(t, flight['Turn_Rate'])
    axs[2, 0].set_title("Turn Rate vs Time")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Turn Rate [rad/s]")

    # 6. Velocity Components
    axs[2, 1].plot(t, flight['vx'], label='vx')
    axs[2, 1].plot(t, flight['vy'], label='vy')
    axs[2, 1].set_title("Velocity Components")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Velocity [m/s]")
    axs[2, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    visualize_flight("dataset/processed_data.csv", "LA713_0")
