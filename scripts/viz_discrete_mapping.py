import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==========================================
# FIGHTER JET CONFIGURATION
# ==========================================
DATA_PATH = "dataset/processed/smoothed_kinematic_trajectories.parquet"
OUTPUT_PATH = "figures/real_discrete_transitions.html"

# Fighter-specific Deadbands 
ACCEL_DEADBAND = 5.0      
TURN_DEADBAND = 0.15      
CLIMB_DEADBAND = 15.0     

NUM_SAMPLES = 3           

# ==========================================
# DISCRETIZER LOGIC
# ==========================================
def discretize_kinematics(val, deadband):
    if pd.isna(val): return 0
    if val > deadband: return 1
    if val < -deadband: return -1
    return 0

def compress_states(states):
    if not states: return [], []
    
    compressed, counts = [], []
    prev, count = states[0], 1
    
    for s in states[1:]:
        if s == prev:
            count += 1
        else:
            compressed.append(prev)
            counts.append(count)
            prev = s
            count = 1
            
    compressed.append(prev)
    counts.append(count)
    return compressed, counts

# ==========================================
# PUBLICATION-READY 3D VISUALIZATION
# ==========================================
def plot_trajectory_comparison(trajectories):
    num_plots = len(trajectories)
    
    fig = make_subplots(
        rows=1, cols=num_plots,
        specs=[[{'type': 'scene'} for _ in range(num_plots)]],
        subplot_titles=[f"<b>{t[2]}</b>" for t in trajectories],
        horizontal_spacing=0.05
    )

    edges = [
        (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), (-1,-1,-1),
        (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1), (-1,-1,1)
    ]
    x_edge, y_edge, z_edge = zip(*edges)

    for i, (states_c, counts, title) in enumerate(trajectories):
        col = i + 1
        
        # 1. Bounding Box
        fig.add_trace(go.Scatter3d(
            x=x_edge, y=y_edge, z=z_edge, mode='lines',
            line=dict(width=1.5, color='rgba(0,0,0,0.15)'),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=col)
        
        for (x0, y0) in [(-1,-1), (1,-1), (1,1), (-1,1)]:
            fig.add_trace(go.Scatter3d(
                x=[x0,x0], y=[y0,y0], z=[-1,1], mode='lines', 
                line=dict(width=1.5, color='rgba(0,0,0,0.15)'),
                showlegend=False, hoverinfo='skip'
            ), row=1, col=col)

        if not states_c: continue
        
        # 2. Nodes & Path
        x_nodes, y_nodes, z_nodes = zip(*states_c)
        node_sizes = [6 + (c * 0.4) for c in counts] 
        node_colors = list(range(len(states_c))) 

        # Connecting Path 
        fig.add_trace(go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes, mode='lines',
            line=dict(width=3, color='rgba(50, 100, 200, 0.6)'),
            showlegend=False
        ), row=1, col=col)

        # State Spheres
        fig.add_trace(go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes, mode='markers',
            marker=dict(
                size=node_sizes, color=node_colors, colorscale='Turbo', 
                line=dict(width=1.5, color='DarkSlateGrey')
            ),
            text=[f"Duration: {c} steps" for c in counts],
            hoverinfo="text", showlegend=False
        ), row=1, col=col)

        # 3. Transitions (Arrows/Cones)
        cone_x, cone_y, cone_z, cone_u, cone_v, cone_w = [], [], [], [], [], []
        for j in range(len(states_c)-1):
            x0, y0, z0 = states_c[j]
            x1, y1, z1 = states_c[j+1]

            u, v, w = x1 - x0, y1 - y0, z1 - z0
            norm = np.linalg.norm([u, v, w])
            if norm > 0:
                cone_x.append(x0 + (u/norm) * norm * 0.6)
                cone_y.append(y0 + (v/norm) * norm * 0.6)
                cone_z.append(z0 + (w/norm) * norm * 0.6)
                cone_u.append(u/norm)
                cone_v.append(v/norm)
                cone_w.append(w/norm)

        if cone_x:
            fig.add_trace(go.Cone(
                x=cone_x, y=cone_y, z=cone_z, u=cone_u, v=cone_v, w=cone_w,
                sizemode="absolute", sizeref=0.15, showscale=False, 
                colorscale="Blues", anchor="center", hoverinfo='skip'
            ), row=1, col=col)

    # 4. Academic Report Layout Formatting (FIXED FONT ATTRIBUTES)
    axis_config = dict(
        range=[-1.5, 1.5], dtick=1, zeroline=False,
        showbackground=False, 
        gridcolor='lightgrey',
        linecolor='black',
        tickfont=dict(size=10, color='black')
    )
    
    title_font_config = dict(size=12, color='black')
    camera_angle = dict(eye=dict(x=1.6, y=1.6, z=1.2))

    scene_layout = dict(
        xaxis=dict(**axis_config, title=dict(text="Accel", font=title_font_config)),
        yaxis=dict(**axis_config, title=dict(text="Turn", font=title_font_config)),
        zaxis=dict(**axis_config, title=dict(text="Climb", font=title_font_config)),
        aspectratio=dict(x=1, y=1, z=1),
        camera=camera_angle
    )

    layout_update = {
        "title_text": "Fighter Jet Intent Profiles: Discrete State Transitions",
        "title_font": dict(family="Arial, sans-serif", size=20, color='black'),
        "font": dict(family="Arial, sans-serif", color='black'),
        "paper_bgcolor": 'white', 
        "plot_bgcolor": 'white',
        "margin": dict(l=10, r=10, b=10, t=70)
    }
    
    for i in range(num_plots):
        scene_key = f"scene{i+1}" if i > 0 else "scene"
        layout_update[scene_key] = scene_layout

    fig.update_layout(**layout_update)
    return fig

# ==========================================
# MAIN DATA INGESTION
# ==========================================
if __name__ == "__main__":
    print(f"[INFO] Loading real trajectory data from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find the dataset at {DATA_PATH}. Check your paths!")
        
    df = pd.read_parquet(DATA_PATH)
    
    all_flights = df['CTN_New'].unique()
    sample_ctns = np.random.choice(all_flights, min(NUM_SAMPLES, len(all_flights)), replace=False)
    
    print(f"[INFO] Selected Flight IDs for visualization: {sample_ctns}")
    
    trajectories_data = []
    
    for ctn in sample_ctns:
        flight_data = df[df['CTN_New'] == ctn].copy()
        
        states = []
        for _, row in flight_data.iterrows():
            a = discretize_kinematics(row['Acceleration'], ACCEL_DEADBAND)
            t = discretize_kinematics(row['Turn_Rate'], TURN_DEADBAND)
            v = discretize_kinematics(row['vz'], CLIMB_DEADBAND)
            states.append((a, t, v))
            
        states_compressed, durations = compress_states(states)
        trajectories_data.append((states_compressed, durations, f"CTN: {ctn}"))
        
    print(f"[INFO] Rendering 3D Visualization...")
    fig = plot_trajectory_comparison(trajectories_data)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.write_html(OUTPUT_PATH)
    print(f"[SUCCESS] Interactive HTML saved to: {OUTPUT_PATH}")