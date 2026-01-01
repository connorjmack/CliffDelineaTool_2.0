#!/usr/bin/env python3
"""
Script to apply CliffDelineaTool v2.0 Deep Learning model 
to specific San Diego beach shapefiles using corresponding DEMs.

Usage:
    Place this script in v2/scripts/ and run:
    python predict_sandiego_v2.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point
from shapely.ops import substring
from PIL import Image

# Add parent directory to path to allow importing cliff_dl
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import the model
from cliff_dl.models.cnn_lstm import CNN_BiLSTM_CliffDetector

# ==========================================
# CONFIGURATION
# ==========================================

# Dictionary mapping Shapefiles (Transects) to their corresponding DEMs (TIFFs).
# UPDATE THE TIFF PATHS below with the actual locations of your DEMs.
DATA_PAIRS = [
    {
        "transects": "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/utilities/shape_files/transects/1m_transects/DelMarTransects595to620at1m/DelMarTransects595to620at1m.shp",
        "dem": "/Volumes/group/LiDAR/VMQLZ_Truck/LiDAR_Processed_Level2/20251204_00590_00636_1252_DelMar_NoWaves/Beach_And_Backshore/20251204_00590_00636_1252_DelMar_NoWaves_beach_cliff_ground.tif" # <--- REPLACE THIS
    },
    {
        "transects": "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/utilities/shape_files/transects/1m_transects/TorreyTransects567to581at1m/TorreyTransects567to581at1m.shp",
        "dem": "/Volumes/group/LiDAR/VMQLZ_Truck/LiDAR_Processed_Level2/20251204_00567_00590_1557_Torrey_NoWaves/Beach_And_Backshore/20251204_00567_00590_1557_Torrey_NoWaves_beach_cliff_ground.tif" # <--- REPLACE THIS
    },
    {
        "transects": "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/utilities/shape_files/transects/1m_transects/SolanaTransects637to666at1m/SolanaTransects637to666at1m.shp",
        "dem": "/Volumes/group/LiDAR/VMQLZ_Truck/LiDAR_Processed_Level2/20251117_00636_00683_1443_Solana_Cardiff_NoWaves/Beach_And_Backshore/20251117_00636_00683_1443_Solana_Cardiff_NoWaves_beach_cliff_ground.tif" # <--- REPLACE THIS
    }
]

# Path to your trained model weights
CHECKPOINT_PATH = parent_dir / "scripts" / "experiments" / "runs" / "checkpoints" / "best_model.pth"

OUTPUT_DIR = parent_dir / "outputs" / "sd_county"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLING_STEP = 1.0  # Sample every 1 meter

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_model(checkpoint_path, input_dim=13):
    """Load the trained CNN_BiLSTM_CliffDetector model"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model = CNN_BiLSTM_CliffDetector(
        input_dim=input_dim,
        cnn_channels=[64, 128, 64],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        attention_heads=8
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

def sample_elevation_from_raster(line_geom, raster_data, raster_transform, raster_nodata, step=1.0):
    """
    Interpolates points along a LineString at 'step' intervals
    and samples elevation from the provided raster data array.
    """
    length = line_geom.length
    distances = np.arange(0, length, step)

    data_points = []

    for dist in distances:
        # Get point at distance
        point = line_geom.interpolate(dist)

        # Sample raster - convert coordinates to array indices
        try:
            row, col = rasterio.transform.rowcol(raster_transform, point.x, point.y)

            # Check bounds
            if row < 0 or col < 0 or row >= raster_data.shape[0] or col >= raster_data.shape[1]:
                continue

            val = raster_data[row, col]

            # Filter NoData (often extremely negative or -9999)
            if val < -100 or (raster_nodata is not None and val == raster_nodata):
                continue

            data_points.append({
                'Distance': dist,
                'Elevation': float(val),
                'geometry': point
            })
        except (IndexError, ValueError):
            # Point outside raster bounds
            continue

    return pd.DataFrame(data_points)

def process_shapefile_with_dem(shp_path, dem_path):
    """
    Reads a LineString shapefile and extracts elevation profiles using the DEM.
    """
    print(f"Reading Transects: {os.path.basename(shp_path)}")
    print(f"Sampling DEM: {os.path.basename(dem_path)}")

    gdf = gpd.read_file(shp_path)

    # Find ID column - try multiple common names
    id_col = None
    possible_id_cols = ['TransectID', 'OBJECTID', 'FID', 'ID', 'Id', 'id', 'fid', 'objectid']
    for col in possible_id_cols:
        if col in gdf.columns:
            id_col = col
            break

    # If no ID column found, use the index
    if id_col is None:
        print(f"  Warning: No standard ID column found. Available columns: {list(gdf.columns)}")
        print(f"  Using row index as TransectID")
        gdf['_temp_id'] = range(len(gdf))
        id_col = '_temp_id'
    else:
        print(f"  Using '{id_col}' as TransectID column")

    transects = []

    with rasterio.open(dem_path) as src:
        # Read the entire raster once for efficiency
        print(f"  Reading DEM into memory...")
        raster_data = src.read(1)
        raster_transform = src.transform
        raster_nodata = src.nodata
        print(f"  DEM loaded: {raster_data.shape}")

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extracting Profiles"):
            geom = row.geometry
            tid = row[id_col]

            if geom.geom_type != 'LineString':
                continue

            # Extract profile
            df = sample_elevation_from_raster(geom, raster_data, raster_transform, raster_nodata, step=SAMPLING_STEP)

            # Check if transect is reversed (cliff on left instead of right)
            # In a normal transect, elevation should increase from sea to land
            # If the first half has higher mean elevation than second half, it's reversed
            if len(df) > 10:
                mid_idx = len(df) // 2
                first_half_elev = df.iloc[:mid_idx]['Elevation'].mean()
                second_half_elev = df.iloc[mid_idx:]['Elevation'].mean()

                # If first half is higher, transect is reversed - flip it
                if first_half_elev > second_half_elev:
                    # Reverse the dataframe
                    df = df.iloc[::-1].reset_index(drop=True)
                    # Recalculate distances from 0
                    max_dist = df['Distance'].max()
                    df['Distance'] = max_dist - df['Distance']
                    # Reverse geometry list
                    df['geometry'] = df['geometry'].tolist()[::-1]

            # Validation
            if len(df) > 10:
                transects.append({
                    'id': tid,
                    'data': df,
                    'geometry': df.geometry.tolist()
                })

    return transects

def extract_features_v2(df):
    """
    Feature extraction (Logic from v2 model description).
    """
    # 1. Normalize Elevation & Distance
    elev = df['Elevation'].values
    dist = df['Distance'].values
    
    elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
    
    # 2. Gradients
    grad = np.gradient(elev, dist)
    grad_norm = (grad - np.mean(grad)) / (np.std(grad) + 1e-6)
    
    # 3. Curvature
    curv = np.gradient(grad, dist)
    curv_norm = (curv - np.mean(curv)) / (np.std(curv) + 1e-6)
    
    # 4 & 5. Slopes (Simplified for inference speed)
    n_vert = 20
    sea_slope = np.zeros_like(elev)
    land_slope = np.zeros_like(elev)
    
    # Vectorized approximation for speed
    for i in range(len(elev)):
        if i >= n_vert:
            sea_slope[i] = np.degrees(np.arctan((elev[i] - elev[i-n_vert]) / (dist[i] - dist[i-n_vert] + 1e-6)))
        if i < len(elev) - n_vert:
            land_slope[i] = np.degrees(np.arctan((elev[i+n_vert] - elev[i]) / (dist[i+n_vert] - dist[i] + 1e-6)))
            
    sea_slope = np.clip(sea_slope / 90.0, 0, 1)
    land_slope = np.clip(land_slope / 90.0, 0, 1)
    
    # 6. Trendline Deviation
    trend = np.linspace(elev[0], elev[-1], len(elev))
    dev_norm = (elev - trend) / (elev.max() - elev.min() + 1e-6)
    
    # 7. Slope Change
    slope_change = land_slope - sea_slope
    
    # 8. Convexity
    convexity = curv / ((1 + grad**2)**1.5)
    
    # 9. Relative Elevation
    rel_elev = (elev - np.mean(elev)) / (np.std(elev) + 1e-6)
    
    # 10. Low Elevation Zone
    low_zone = (elev < 15).astype(float)
    
    # 11. Shore Proximity
    shore_prox = np.exp(-5.0 * dist_norm)
    
    # 12. Max Local Slope
    local_slope = np.abs(grad)
    max_local = local_slope / (local_slope.max() + 1e-6)

    features = np.stack([
        elev_norm, dist_norm, grad_norm, curv_norm,
        sea_slope, land_slope, dev_norm, slope_change,
        convexity, rel_elev, low_zone, shore_prox, max_local
    ], axis=1)
    
    return torch.FloatTensor(features).unsqueeze(0) 

def predict_transects(model, transects):
    """Run inference"""
    results = []
    
    with torch.no_grad():
        for t in tqdm(transects, desc="Predicting"):
            input_tensor = extract_features_v2(t['data']).to(DEVICE)

            outputs = model(input_tensor)
            seg_logits = outputs['segmentation']
            reg_offsets = outputs['regression']
            confidence = outputs['confidence']

            probs = torch.softmax(seg_logits, dim=-1)
            base_probs = probs[0, :, 1].cpu().numpy()
            top_probs = probs[0, :, 2].cpu().numpy()
            conf_score = confidence[0, 0].item()  # Already has sigmoid applied in model
            
            base_idx = np.argmax(base_probs)
            top_idx = np.argmax(top_probs)
            
            df = t['data']
            base_geom = t['geometry'][base_idx]
            top_geom = t['geometry'][top_idx]
            
            results.append({
                'TransectID': t['id'],
                'Base_Confidence': base_probs[base_idx],
                'Top_Confidence': top_probs[top_idx],
                'Transect_Confidence': conf_score,
                'Base_X': base_geom.x,
                'Base_Y': base_geom.y,
                'Base_Z': df.iloc[base_idx]['Elevation'],
                'Top_X': top_geom.x,
                'Top_Y': top_geom.y,
                'Top_Z': df.iloc[top_idx]['Elevation']
            })
            
    return pd.DataFrame(results)

def plot_transect_with_predictions(transect_data, predictions_row, output_path):
    """
    Create a visualization of a single transect with predicted cliff base and top.
    """
    df = transect_data['data']
    distances = df['Distance'].values
    elevation = df['Elevation'].values

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot elevation profile
    ax.plot(distances, elevation, 'k-', linewidth=1.5, label='Elevation Profile', alpha=0.7)
    ax.fill_between(distances, elevation.min() - 2, elevation, alpha=0.1, color='gray')

    # Get predicted positions from the predictions row
    base_x = predictions_row['Base_X']
    base_y = predictions_row['Base_Y']
    top_x = predictions_row['Top_X']
    top_y = predictions_row['Top_Y']

    # Find the distance values that match these coordinates
    base_dist = predictions_row.get('Base_Distance', None)
    top_dist = predictions_row.get('Top_Distance', None)

    # If distances aren't in the row, find the closest point
    if base_dist is None or pd.isna(base_dist):
        # Find closest geometry point to base_x, base_y
        geometries = transect_data['geometry']
        min_dist = float('inf')
        base_idx = 0
        for i, geom in enumerate(geometries):
            dist = ((geom.x - base_x)**2 + (geom.y - base_y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                base_idx = i
        base_dist = df.iloc[base_idx]['Distance']

    if top_dist is None or pd.isna(top_dist):
        # Find closest geometry point to top_x, top_y
        geometries = transect_data['geometry']
        min_dist = float('inf')
        top_idx = 0
        for i, geom in enumerate(geometries):
            dist = ((geom.x - top_x)**2 + (geom.y - top_y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                top_idx = i
        top_dist = df.iloc[top_idx]['Distance']

    # Mark predicted positions
    ax.axvline(base_dist, color='red', linestyle='-', linewidth=2.5,
               label=f'Pred Base ({base_dist:.1f}m)', alpha=0.8, zorder=6)
    ax.axvline(top_dist, color='orange', linestyle='-', linewidth=2.5,
               label=f'Pred Top ({top_dist:.1f}m)', alpha=0.8, zorder=6)

    # Shade cliff zone
    if base_dist < top_dist:
        ax.axvspan(base_dist, top_dist, alpha=0.15, color='orange',
                  label='Predicted Cliff Zone', zorder=1)

    # Add confidence scores
    conf_text = f'Base Conf: {predictions_row["Base_Confidence"]:.3f}\n'
    conf_text += f'Top Conf: {predictions_row["Top_Confidence"]:.3f}\n'
    conf_text += f'Transect Conf: {predictions_row["Transect_Confidence"]:.3f}'

    ax.text(0.02, 0.98, conf_text,
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Labels
    ax.set_xlabel('Distance from Sea (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Elevation (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Transect ID: {predictions_row["TransectID"]}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return True

def create_location_gif(transects, results_df, location_name, output_dir, sample_every=10):
    """
    Create a GIF visualization for all transects at a location.

    Args:
        transects: List of transect data
        results_df: DataFrame with predictions
        location_name: Name for the output file
        output_dir: Directory to save output
        sample_every: Include every Nth transect (default: 10)
    """
    print(f"  Creating GIF for {location_name}...")
    print(f"  Sampling every {sample_every} transects to avoid file limit issues...")

    frames_dir = output_dir / f'{location_name}_frames'
    frames_dir.mkdir(exist_ok=True)

    frames = []

    # Sample every Nth transect and result row together
    sampled_indices = list(range(0, len(transects), sample_every))

    for i, idx in enumerate(tqdm(sampled_indices,
                                 desc=f"  Generating {location_name} frames")):
        if idx >= len(transects) or idx >= len(results_df):
            continue

        transect_data = transects[idx]
        result_row = results_df.iloc[idx]
        frame_path = frames_dir / f"frame_{i:04d}.png"

        try:
            success = plot_transect_with_predictions(transect_data, result_row, frame_path)
            if success:
                frames.append(frame_path)
        except Exception as e:
            print(f"    Warning: Could not create frame for transect index {idx}: {e}")
            continue

    if not frames:
        print(f"    Warning: No frames generated for {location_name}")
        frames_dir.rmdir()
        return None

    print(f"    Generated {len(frames)} frames")
    print(f"    Creating GIF...")

    # Create GIF - open images one at a time to avoid "too many open files" error
    gif_path = output_dir / f"{location_name}.gif"

    # Load first image
    first_image = Image.open(frames[0])

    # Load remaining images
    remaining_images = []
    for frame_path in frames[1:]:
        img = Image.open(frame_path)
        remaining_images.append(img.copy())  # Copy to memory
        img.close()  # Close file handle

    # Save GIF
    first_image.save(
        gif_path,
        save_all=True,
        append_images=remaining_images,
        duration=500,  # 0.5 second per frame
        loop=0
    )
    first_image.close()

    print(f"    ✓ Saved GIF to {gif_path}")
    print(f"    Size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Clean up frames
    for frame in frames:
        frame.unlink()
    frames_dir.rmdir()

    return gif_path

# ==========================================
# MAIN
# ==========================================

def main():
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading model from {CHECKPOINT_PATH}...")
    try:
        model = load_model(CHECKPOINT_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for pair in DATA_PAIRS:
        shp_file = pair["transects"]
        dem_file = pair["dem"]

        if not os.path.exists(shp_file):
            print(f"Missing Transect file: {shp_file}")
            continue
        if not os.path.exists(dem_file) or "path/to" in dem_file:
            print(f"Missing or invalid DEM path: {dem_file}")
            print("Please update the DATA_PAIRS list in the script.")
            continue

        # 1. Load & Extract
        try:
            transects = process_shapefile_with_dem(shp_file, dem_file)
            print(f"  Extracted {len(transects)} valid profiles.")
        except Exception as e:
            print(f"  Error processing files: {e}")
            continue

        # 2. Predict
        if len(transects) > 0:
            results_df = predict_transects(model, transects)

            # 3. Save
            base_name = os.path.basename(shp_file).replace('.shp', '')

            # CSV
            results_df.to_csv(OUTPUT_DIR / f"{base_name}_predictions.csv", index=False)

            # Shapefiles
            base_gdf = gpd.GeoDataFrame(
                results_df,
                geometry=gpd.points_from_xy(results_df.Base_X, results_df.Base_Y),
                crs="EPSG:26911"
            )
            base_gdf.to_file(OUTPUT_DIR / f"{base_name}_base_predicted.shp")

            top_gdf = gpd.GeoDataFrame(
                results_df,
                geometry=gpd.points_from_xy(results_df.Top_X, results_df.Top_Y),
                crs="EPSG:26911"
            )
            top_gdf.to_file(OUTPUT_DIR / f"{base_name}_top_predicted.shp")

            print(f"  Saved results to {OUTPUT_DIR}")

            # 4. Create GIF visualization
            try:
                create_location_gif(transects, results_df, base_name, OUTPUT_DIR)
            except Exception as e:
                print(f"  Error creating GIF: {e}")
        else:
            print("  No valid transects found (check DEM overlap).")

if __name__ == "__main__":
    main()