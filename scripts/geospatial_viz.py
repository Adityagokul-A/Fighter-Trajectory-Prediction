import os
import sys
import pathlib
import yaml
import glob
import numpy as np
import pandas as pd
import torch
import folium
from branca.element import Template, MacroElement
from typing import Optional

# Ensure repository root is on sys.path so `src` imports work when running the script
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import HybridTrajectoryDataset
# Import both possible model wrappers used in training
from src.models.predictor import BaselineTrajectoryModel
from src.models.cvae_predictor import TrajectoryCVAE
import os
import sys
import pathlib
import yaml
import glob
import numpy as np
import pandas as pd
import torch
import folium
from branca.element import Template, MacroElement
from typing import Optional

# Ensure repository root is on sys.path so `src` imports work when running the script
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import HybridTrajectoryDataset
# Import both possible model wrappers used in training
from src.models.predictor import BaselineTrajectoryModel
from src.models.cvae_predictor import TrajectoryCVAE

# Earth radius used in preprocessing (match smoothing.py)
EARTH_RADIUS = 6378137.0


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def enu_to_llh(x_e, y_n, lat0, lon0, alt0=0.0):
    """Approximate inverse of llh_to_enu_vectorized used in preprocessing.
    Uses small-angle equirectangular approx that matches preprocessing transform.
    x_e, y_n can be scalars or arrays (meters). lat0/lon0 in degrees.
    Returns (lat, lon) in degrees.
    """
    lat0_rad = np.deg2rad(lat0)
    dlat = y_n / EARTH_RADIUS
    dlon = x_e / (EARTH_RADIUS * np.cos(lat0_rad) + 1e-12)
    lat = np.rad2deg(lat0_rad + dlat)
    lon = np.rad2deg(np.deg2rad(lon0) + dlon)
    return lat, lon


def run_viz(sample_flight: Optional[str] = None, sample_index: Optional[int] = None, out_html: str = 'geovis.html', parquet_path: Optional[str] = None):
    # Load config
    model_cfg = load_yaml('config/model_config.yaml')['model']
    train_cfg = load_yaml('config/train_config.yaml')['training']

    # Determine parquet path (config override or explicit argument)
    default_parquet = 'dataset/processed/smoothed_kinematic_trajectories.parquet'
    parquet_path = parquet_path or train_cfg.get('parquet_path', default_parquet)

    # Helpful check: if parquet is missing, list candidate files and explain how to generate/process them
    if not os.path.exists(parquet_path):
        proc_dir = os.path.join('dataset', 'processed')
        cand = []
        if os.path.isdir(proc_dir):
            cand = [os.path.join(proc_dir, f) for f in os.listdir(proc_dir) if f.endswith('.parquet')]

        other_files = []
        if os.path.isdir('dataset') and not cand:
            other_files = [os.path.join('dataset', f) for f in os.listdir('dataset')]

        msg_lines = [f"Parquet file not found at: {parquet_path}"]
        if cand:
            msg_lines.append("Candidate parquet files in dataset/processed:")
            msg_lines += [f"  - {p}" for p in cand]
        elif other_files:
            msg_lines.append("Files in dataset/ (no processed parquet found):")
            msg_lines += [f"  - {p}" for p in other_files]
        else:
            msg_lines.append("No dataset files found. Ensure dataset has been prepared under the 'dataset' directory.")

        msg_lines.append("")
        msg_lines.append("If you have the processed parquet in a different location, run this script with:")
        msg_lines.append("    python scripts/geospatial_viz.py --parquet PATH/TO/your.parquet")
        msg = "\n".join(msg_lines)
        raise FileNotFoundError(msg)

    # Dataset to retrieve normalization and feature layout
    dataset = HybridTrajectoryDataset(
        parquet_path=parquet_path,
        input_window=model_cfg.get('input_window', 30),
        pred_window=model_cfg.get('pred_window', 10),
        normalize=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpts = glob.glob(os.path.join(train_cfg['checkpoint_dir'], '*.pt'))
    if not ckpts:
        raise FileNotFoundError('No checkpoints found in checkpoint_dir')
    latest = max(ckpts, key=os.path.getctime)
    print('Loading checkpoint:', latest)
    ck = torch.load(latest, map_location=device)

    # Create appropriate model wrapper based on checkpoint or best guess
    model = None
    ck_model_type = None
    if isinstance(ck, dict) and 'model_type' in ck:
        ck_model_type = str(ck['model_type']).lower()

    try:
        if ck_model_type and 'cvae' in ck_model_type:
            model = TrajectoryCVAE()
        else:
            model = BaselineTrajectoryModel()

        state = ck.get('model_state', ck)
        model.load_state_dict(state)
    except Exception:
        try:
            model = TrajectoryCVAE() if not isinstance(model, TrajectoryCVAE) else BaselineTrajectoryModel()
            state = ck.get('model_state', ck)
            model.load_state_dict(state)
        except Exception as e:
            keys = ck.keys() if isinstance(ck, dict) else 'not-dict'
            raise RuntimeError(f'Failed to load checkpoint into known model wrappers. Checkpoint keys: {keys}. Error: {e}')

    model.to(device)
    model.eval()

    # Load smoothed parquet for geodetic origins and lat/lon columns
    df = pd.read_parquet(parquet_path)

    # Choose a flight to visualize
    if sample_flight is None and sample_index is None:
        counts = df['CTN_New'].value_counts()
        candidates = counts[counts >= (model_cfg.get('input_window', 30) + model_cfg.get('pred_window', 10))].index
        if len(candidates) == 0:
            raise RuntimeError('No flights long enough for chosen windows')
        flight_id = candidates[0]
    elif sample_flight is not None:
        flight_id = sample_flight
    else:
        # pick by flight index via the parquet CTN_New values (dataset does not expose flight_ids)
        flight_ids = df['CTN_New'].unique().tolist()
        try:
            flight_id = flight_ids[sample_index]
        except Exception as e:
            raise IndexError(f"sample_index {sample_index} out of range (0..{len(flight_ids)-1})") from e

    print('Selected flight:', flight_id)
    flight_df = df[df['CTN_New'] == flight_id].sort_values('TIME').reset_index(drop=True)

    N = model_cfg.get('input_window', 30)
    K = model_cfg.get('pred_window', 10)

    if len(flight_df) < N + K:
        raise RuntimeError('Flight too short for windows')

    idx = max(0, len(flight_df) - (N + K))

    past_df = flight_df.iloc[idx: idx + N]
    future_df = flight_df.iloc[idx + N: idx + N + K]

    feat_cols = dataset.feature_cols
    x_past = past_df[feat_cols].values.astype(np.float32)
    x_past = (x_past - dataset.mean) / dataset.std if dataset.normalize else x_past
    x_past = torch.from_numpy(x_past).unsqueeze(0).to(device)

    input_lengths = torch.tensor([x_past.size(1)], dtype=torch.long, device=device)

    with torch.no_grad():
        if isinstance(model, TrajectoryCVAE):
            samples = model.inference(x_past, input_lengths, K, num_samples=1)
            pred_deltas = samples[:, 0, :, :]
        else:
            preds, _ = model(x_past, input_lengths, K, targets=None, tf_ratio=0.0)
            pred_deltas = preds

    spatial_std = dataset.std[0:3].astype(np.float32)

    x_past_np = x_past.cpu().numpy().squeeze()
    x_past_meters = (x_past_np * dataset.std) + dataset.mean
    path_past = x_past_meters[:, 0:3]

    y_gt_deltas = (future_df[['x_s','y_s','z_s']].values.astype(np.float32) - path_past[-1])

    y_pred_deltas = pred_deltas.cpu().numpy().squeeze() * spatial_std
    path_pred = np.cumsum(y_pred_deltas, axis=0)

    path_past = path_past - path_past[-1]
    path_gt = np.vstack([[0,0,0], np.cumsum(y_gt_deltas, axis=0)])
    path_pred = np.vstack([[0,0,0], path_pred])

    lat0 = flight_df.iloc[0]['Latitude']
    lon0 = flight_df.iloc[0]['Longitude']
    current_pos_enu = past_df[['x_s','y_s','z_s']].values[-1]

    abs_past = path_past + current_pos_enu
    abs_gt = path_gt + current_pos_enu
    abs_pred = path_pred + current_pos_enu

    past_lat, past_lon = enu_to_llh(abs_past[:,0], abs_past[:,1], lat0, lon0)
    gt_lat, gt_lon = enu_to_llh(abs_gt[:,0], abs_gt[:,1], lat0, lon0)
    pred_lat, pred_lon = enu_to_llh(abs_pred[:,0], abs_pred[:,1], lat0, lon0)

    cur_lat, cur_lon = enu_to_llh(current_pos_enu[0], current_pos_enu[1], lat0, lon0)
    m = folium.Map(location=[cur_lat, cur_lon], zoom_start=11)

    folium.PolyLine(list(zip(past_lat, past_lon)), color='cyan', weight=3, opacity=0.8, tooltip='History').add_to(m)
    folium.PolyLine(list(zip(gt_lat, gt_lon)), color='green', weight=4, opacity=0.9, tooltip='GT Future').add_to(m)
    folium.PolyLine(list(zip(pred_lat, pred_lon)), color='magenta', weight=4, opacity=0.9, tooltip='Predicted').add_to(m)

    folium.Marker(location=[cur_lat, cur_lon], popup='Current Position', icon=folium.Icon(color='white', icon_color='black')).add_to(m)

    # Add a small legend/key to the map (fixed position)
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 110px; 
                border:2px solid grey; z-index:9999; font-size:14px; background-color: white; 
                opacity: 0.9; padding: 8px;">
      <b>Legend</b><br>
      <div style="margin-top:6px">
        <span style="display:inline-block;width:12px;height:12px;background:cyan;margin-right:8px;"></span>History<br>
        <span style="display:inline-block;width:12px;height:12px;background:green;margin-right:8px;"></span>GT Future<br>
        <span style="display:inline-block;width:12px;height:12px;background:magenta;margin-right:8px;"></span>Predicted
      </div>
    </div>
    {% endmacro %}
    """

    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    m.save(out_html)
    print(f'Saved geospatial visualization to {out_html}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Geospatial visualization of a model prediction for one flight')
    parser.add_argument('--flight', type=str, default=None, help='CTN_New flight id to visualize')
    parser.add_argument('--index', type=int, default=None, help='Flight index (dataset.flight_ids) to visualize')
    parser.add_argument('--out', type=str, default='geovis.html', help='Output HTML file')
    parser.add_argument('--parquet', type=str, default=None, help='Path to processed parquet file (overrides config)')

    args = parser.parse_args()
    run_viz(sample_flight=args.flight, sample_index=args.index, out_html=args.out, parquet_path=args.parquet)

