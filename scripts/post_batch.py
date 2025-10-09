#!/usr/bin/env python3
"""
Batch ROS Bag Post-Processing Script

Features:
- Recursively read .bag files under given inputs
- Infer (map, method) grouping from path or args
- For each bag:
  * Reuse single-bag logic: sync GT and VINS, rigid alignment, compute position/orientation errors
  * With a position error threshold, find first exceed time and record exploration rate (interpolated)
  * Compute total duration and whether it's a successful run
  * Build R2D2 and OV-MSCKF 2D heatmaps normalized per message
- Aggregate to (map, method):
  * Mean/std of normalized success exploration rate
  * Mean/std exploration time for successful runs
  * Save average per-message heatmaps and per-map comparison figures
- Save CSV/JSON summaries
"""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import rosbag
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Utilities reused from single-bag logic

def calculate_position_error(gt_pos, est_pos) -> float:
    dx = gt_pos.x - est_pos.x
    dy = gt_pos.y - est_pos.y
    dz = gt_pos.z - est_pos.z
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def calculate_orientation_error(gt_quat, est_quat) -> float:
    if hasattr(gt_quat, 'x'):
        gt_quat_array = [gt_quat.x, gt_quat.y, gt_quat.z, gt_quat.w]
    else:
        gt_quat_array = gt_quat
    if hasattr(est_quat, 'x'):
        est_quat_array = [est_quat.x, est_quat.y, est_quat.z, est_quat.w]
    else:
        est_quat_array = est_quat
    r_gt = R.from_quat(gt_quat_array)
    r_est = R.from_quat(est_quat_array)
    r_rel = r_gt * r_est.inv()
    angle = r_rel.magnitude()
    return float(angle)

# Added helpers for clarity and reuse

def compute_rigid_alignment_from_first_odoms(gt_first_msg, vins_first_msg) -> Tuple[R, np.ndarray]:
    """Compute rotation (R_gt * R_vins^{-1}) and translation aligning VINS to GT at first common timestamp."""
    gt_initial_pos = gt_first_msg.pose.pose.position
    vins_initial_pos = vins_first_msg.pose.pose.position
    gt_initial_quat = gt_first_msg.pose.pose.orientation
    vins_initial_quat = vins_first_msg.pose.pose.orientation

    r_gt_initial = R.from_quat([gt_initial_quat.x, gt_initial_quat.y, gt_initial_quat.z, gt_initial_quat.w])
    r_vins_initial = R.from_quat([vins_initial_quat.x, vins_initial_quat.y, vins_initial_quat.z, vins_initial_quat.w])
    r_transform = r_gt_initial * r_vins_initial.inv()

    gt_pos_array = np.array([gt_initial_pos.x, gt_initial_pos.y, gt_initial_pos.z], dtype=float)
    vins_pos_array = np.array([vins_initial_pos.x, vins_initial_pos.y, vins_initial_pos.z], dtype=float)
    vins_pos_rotated = r_transform.apply(vins_pos_array)
    translation = gt_pos_array - vins_pos_rotated
    return r_transform, translation


def process_rosbag(bag_path: str):
    print(f"Processing ROS bag: {bag_path}")
    exploration_data = []
    gt_odom_data = []
    vins_odom_data = []
    r2d2_pointcloud_data = []
    ov_msckf_pointcloud_data = []
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            info = bag.get_type_and_topic_info()
            topics = list(info.topics.keys())
            print(f"Available topics: {topics}")
            for topic, msg, t in bag.read_messages():
                timestamp = t.to_sec()
                if topic == '/exploration_rate':
                    rate = msg.data if hasattr(msg, 'data') else float(msg)
                    exploration_data.append([timestamp, rate])
                elif topic == '/kingfisher/ground_truth/odometry':
                    gt_odom_data.append([timestamp, msg])
                elif topic == '/ov_msckf/loop_pose':
                    vins_odom_data.append([timestamp, msg])
                elif topic == '/r2d2/visible_features_uv':
                    r2d2_pointcloud_data.append([timestamp, msg])
                elif topic == '/ov_msckf/loop_feats':
                    ov_msckf_pointcloud_data.append([timestamp, msg])
    except Exception as e:
        print(f"Error reading bag file: {e}")
        return None, None, None, None, None
    print(f"Found {len(exploration_data)} exploration rate messages")
    print(f"Found {len(gt_odom_data)} ground truth odometry messages")
    print(f"Found {len(vins_odom_data)} VINS odometry messages")
    print(f"Found {len(r2d2_pointcloud_data)} R2D2 point cloud messages")
    print(f"Found {len(ov_msckf_pointcloud_data)} OV-MSCKF loop feature messages")
    return exploration_data, gt_odom_data, vins_odom_data, r2d2_pointcloud_data, ov_msckf_pointcloud_data


def synchronize_and_calculate_errors(exploration_data, gt_odom_data, vins_odom_data):
    if not all([exploration_data, gt_odom_data, vins_odom_data]):
        print("Error: Missing data for one or more topics")
        return (None,) * 12
    exploration_times = np.array([d[0] for d in exploration_data])
    exploration_rates = np.array([d[1] for d in exploration_data])
    gt_times = np.array([d[0] for d in gt_odom_data])
    vins_times = np.array([d[0] for d in vins_odom_data])
    start_time = max(exploration_times.min(), gt_times.min(), vins_times.min())
    end_time = min(exploration_times.max(), gt_times.max(), vins_times.max())
    print(f"Common time range: {start_time:.2f} to {end_time:.2f} seconds")
    
    # Calculate speed thresholds for GT trajectory
    def compute_speed_thresholds(times, x, y, z, speed_threshold=0.1):
        """Find start and end times based on speed thresholds"""
        if len(times) < 2:
            return start_time, end_time
            
        t = times.astype(np.float64)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        dt = np.diff(t)
        dt[dt == 0] = np.nan
        speed = np.sqrt(dx*dx + dy*dy + dz*dz) / dt
        t_mid = (t[1:] + t[:-1]) / 2.0
        
        # Find first time when speed exceeds threshold
        speed_start_idx = np.where(speed >= speed_threshold)[0]
        if len(speed_start_idx) > 0:
            speed_start_time = t_mid[speed_start_idx[0]]
        else:
            speed_start_time = start_time
            
        # Find last time when speed exceeds threshold (going backwards)
        speed_end_idx = np.where(speed >= speed_threshold)[0]
        if len(speed_end_idx) > 0:
            speed_end_time = t_mid[speed_end_idx[-1]]
        else:
            speed_end_time = end_time
            
        return speed_start_time, speed_end_time
    
    # Extract GT trajectory positions for speed calculation
    gt_positions = []
    gt_pos_times = []
    for timestamp, msg in gt_odom_data:
        if start_time <= timestamp <= end_time:
            gt_positions.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
            gt_pos_times.append(timestamp)
    
    if len(gt_positions) >= 2:
        gt_positions = np.array(gt_positions)
        gt_pos_times = np.array(gt_pos_times)
        speed_start_time, speed_end_time = compute_speed_thresholds(
            gt_pos_times, gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2]
        )
        print(f"Speed-based time range: {speed_start_time:.2f} to {speed_end_time:.2f} seconds")
        # Update time range based on speed thresholds
        start_time = max(start_time, speed_start_time)
        end_time = min(end_time, speed_end_time)
        print(f"Adjusted time range: {start_time:.2f} to {end_time:.2f} seconds")
    else:
        print("Warning: Insufficient GT trajectory data for speed threshold calculation")
    
    exp_mask = (exploration_times >= start_time) & (exploration_times <= end_time)
    gt_mask = (gt_times >= start_time) & (gt_times <= end_time)
    vins_mask = (vins_times >= start_time) & (vins_times <= end_time)
    exploration_times_filtered = exploration_times[exp_mask]
    exploration_rates_filtered = exploration_rates[exp_mask]
    first_gt_time = gt_times[gt_mask][0] if np.any(gt_mask) else None
    first_vins_time = vins_times[vins_mask][0] if np.any(vins_mask) else None
    if first_gt_time is None or first_vins_time is None:
        print("Error: Cannot find first odometry messages")
        return (None,) * 12
    first_common_time = max(first_gt_time, first_vins_time)
    print(f"First common odometry timestamp: {first_common_time:.2f} seconds")
    gt_first_idx = int(np.argmin(np.abs(gt_times - first_common_time)))
    vins_first_idx = int(np.argmin(np.abs(vins_times - first_common_time)))
    gt_first_msg = gt_odom_data[gt_first_idx][1]
    vins_first_msg = vins_odom_data[vins_first_idx][1]
    # Use helper for rigid alignment
    r_transform, translation = compute_rigid_alignment_from_first_odoms(gt_first_msg, vins_first_msg)
    print(f"Initial position offset: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}) m")
    print(f"Initial orientation offset: {np.degrees(r_transform.magnitude()):.2f}°")
    error_times = []
    position_errors = []
    orientation_errors = []
    gt_trajectory_times = []
    gt_trajectory_x = []
    gt_trajectory_y = []
    gt_trajectory_z = []
    vins_trajectory_times = []
    vins_trajectory_x = []
    vins_trajectory_y = []
    vins_trajectory_z = []
    for vins_time, vins_msg in vins_odom_data:
        if vins_time < first_common_time or vins_time > end_time:
            continue
        gt_diffs = np.abs(gt_times - vins_time)
        gt_idx = int(np.argmin(gt_diffs))
        if gt_diffs[gt_idx] > 0.1:
            continue
        gt_msg = gt_odom_data[gt_idx][1]
        vins_pos_array = np.array([
            vins_msg.pose.pose.position.x,
            vins_msg.pose.pose.position.y,
            vins_msg.pose.pose.position.z,
        ])
        vins_pos_rotated = r_transform.apply(vins_pos_array)
        vins_pos_transformed = vins_pos_rotated + translation
        aligned_vins_pos = Point()
        aligned_vins_pos.x = float(vins_pos_transformed[0])
        aligned_vins_pos.y = float(vins_pos_transformed[1])
        aligned_vins_pos.z = float(vins_pos_transformed[2])
        r_vins_current = R.from_quat([
            vins_msg.pose.pose.orientation.x,
            vins_msg.pose.pose.orientation.y,
            vins_msg.pose.pose.orientation.z,
            vins_msg.pose.pose.orientation.w,
        ])
        r_vins_aligned = r_transform * r_vins_current
        pos_error = calculate_position_error(gt_msg.pose.pose.position, aligned_vins_pos)
        orient_error = calculate_orientation_error(
            gt_msg.pose.pose.orientation,
            r_vins_aligned.as_quat(),
        )
        error_times.append(vins_time)
        position_errors.append(pos_error)
        orientation_errors.append(orient_error)
        gt_trajectory_times.append(gt_times[gt_idx])
        gt_trajectory_x.append(gt_msg.pose.pose.position.x)
        gt_trajectory_y.append(gt_msg.pose.pose.position.y)
        gt_trajectory_z.append(gt_msg.pose.pose.position.z)
        vins_trajectory_times.append(vins_time)
        vins_trajectory_x.append(aligned_vins_pos.x)
        vins_trajectory_y.append(aligned_vins_pos.y)
        vins_trajectory_z.append(aligned_vins_pos.z)
    error_times = np.array(error_times)
    position_errors = np.array(position_errors)
    orientation_errors = np.array(orientation_errors)
    gt_trajectory_times = np.array(gt_trajectory_times)
    gt_trajectory_x = np.array(gt_trajectory_x)
    gt_trajectory_y = np.array(gt_trajectory_y)
    gt_trajectory_z = np.array(gt_trajectory_z)
    vins_trajectory_times = np.array(vins_trajectory_times)
    vins_trajectory_x = np.array(vins_trajectory_x)
    vins_trajectory_y = np.array(vins_trajectory_y)
    vins_trajectory_z = np.array(vins_trajectory_z)
    print(f"Calculated {len(error_times)} error measurements after alignment")
    return (
        exploration_times_filtered, exploration_rates_filtered,
        error_times, position_errors, orientation_errors,
        gt_trajectory_times, gt_trajectory_x, gt_trajectory_y, gt_trajectory_z,
        vins_trajectory_times, vins_trajectory_x, vins_trajectory_y, vins_trajectory_z,
    )


def process_pointcloud_data(r2d2_pointcloud_data, ov_msckf_pointcloud_data):
    r2d2_u_coords = []
    r2d2_v_coords = []
    r2d2_intensities = []
    ov_msckf_u_coords = []
    ov_msckf_v_coords = []
    # Per-message stats
    r2d2_weight_per_msg = []  # sum of intensities (or count if no intensity) per message
    ov_count_per_msg = []     # count of features per message
    for _, msg in r2d2_pointcloud_data:
        try:
            msg_sum = 0.0
            msg_count = 0
            if hasattr(msg, 'fields'):
                u_field = v_field = intensity_field = None
                for field in msg.fields:
                    if field.name == 'u':
                        u_field = field
                    elif field.name == 'v':
                        v_field = field
                    elif field.name == 'intensity':
                        intensity_field = field
                if u_field is not None and v_field is not None:
                    u_offset = u_field.offset
                    v_offset = v_field.offset
                    intensity_offset = intensity_field.offset if intensity_field else None
                    point_step = msg.point_step
                    data = np.frombuffer(msg.data, dtype=np.uint8)
                    for i in range(msg.width * msg.height):
                        start_idx = i * point_step
                        u = np.frombuffer(data[start_idx + u_offset:start_idx + u_offset + 4], dtype=np.float32)[0]
                        v = np.frombuffer(data[start_idx + v_offset:start_idx + v_offset + 4], dtype=np.float32)[0]
                        intensity = 1.0
                        if intensity_field is not None:
                            intensity = np.frombuffer(data[start_idx + intensity_offset:start_idx + intensity_offset + 4], dtype=np.float32)[0]
                        r2d2_u_coords.append(float(u))
                        r2d2_v_coords.append(float(v))
                        r2d2_intensities.append(float(intensity))
                        msg_sum += float(intensity)
                        msg_count += 1
            else:
                if hasattr(msg, 'channels') and len(msg.channels) > 0:
                    u_channel = v_channel = intensity_channel = None
                    for channel in msg.channels:
                        if channel.name == 'u':
                            u_channel = channel
                        elif channel.name == 'v':
                            v_channel = channel
                        elif channel.name == 'intensity':
                            intensity_channel = channel
                    if u_channel is not None and v_channel is not None:
                        for i in range(len(msg.points)):
                            u = u_channel.values[i]
                            v = v_channel.values[i]
                            intensity = intensity_channel.values[i] if intensity_channel else 1.0
                            r2d2_u_coords.append(float(u))
                            r2d2_v_coords.append(float(v))
                            r2d2_intensities.append(float(intensity))
                            msg_sum += float(intensity)
                            msg_count += 1
        except Exception as e:
            print(f"Error processing R2D2 point cloud message: {e}")
            msg_sum = msg_sum if 'msg_sum' in locals() else 0.0
            msg_count = msg_count if 'msg_count' in locals() else 0
        # Record per-message weight sum
        if msg_count > 0:
            r2d2_weight_per_msg.append(msg_sum)
    for _, msg in ov_msckf_pointcloud_data:
        try:
            msg_count = 0
            if hasattr(msg, 'fields'):
                u_field = v_field = None
                for field in msg.fields:
                    if field.name == 'u':
                        u_field = field
                    elif field.name == 'v':
                        v_field = field
                if u_field is not None and v_field is not None:
                    u_offset = u_field.offset
                    v_offset = v_field.offset
                    point_step = msg.point_step
                    data = np.frombuffer(msg.data, dtype=np.uint8)
                    for i in range(msg.width * msg.height):
                        start_idx = i * point_step
                        u = np.frombuffer(data[start_idx + u_offset:start_idx + u_offset + 4], dtype=np.float32)[0]
                        v = np.frombuffer(data[start_idx + v_offset:start_idx + v_offset + 4], dtype=np.float32)[0]
                        ov_msckf_u_coords.append(float(u))
                        ov_msckf_v_coords.append(float(v))
                        msg_count += 1
            else:
                if hasattr(msg, 'channels') and len(msg.channels) > 0:
                    for channel in msg.channels:
                        if len(channel.values) >= 5:
                            u = channel.values[2]
                            v = channel.values[3]
                            ov_msckf_u_coords.append(float(u))
                            ov_msckf_v_coords.append(float(v))
                            msg_count += 1
        except Exception as e:
            print(f"Error processing OV-MSCKF point cloud message: {e}")
            msg_count = msg_count if 'msg_count' in locals() else 0
        # Record per-message count
        if msg_count > 0:
            ov_count_per_msg.append(msg_count)
    print(f"Extracted {len(r2d2_u_coords)} R2D2 feature points")
    print(f"Extracted {len(ov_msckf_u_coords)} OV-MSCKF loop feature points")
    return (
        r2d2_u_coords, r2d2_v_coords, r2d2_intensities,
        ov_msckf_u_coords, ov_msckf_v_coords,
        r2d2_weight_per_msg, ov_count_per_msg,
    )


def interpolate_exploration_rate(exploration_times, exploration_rates, target_times):
    if len(exploration_times) < 2:
        return np.full_like(target_times, np.nan)
    interp_func = interp1d(
        exploration_times, exploration_rates, kind='linear', bounds_error=False, fill_value=np.nan
    )
    return interp_func(target_times)

def default_edges(image_width: int, image_height: int, bin_size: int) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(0, image_width, image_width // bin_size + 1),
        np.linspace(0, image_height, image_height // bin_size + 1),
    )


def accumulate_heatmap(sum_array: Optional[np.ndarray], heatmap: np.ndarray, message_count: int) -> np.ndarray:
    if sum_array is None:
        return np.array(heatmap, dtype=np.float64) * max(1, int(message_count))
    return sum_array + np.array(heatmap, dtype=np.float64) * max(1, int(message_count))


def average_heatmap(sum_array: Optional[np.ndarray], total_messages: int, image_width: int, image_height: int, bin_size: int) -> np.ndarray:
    if sum_array is None or int(total_messages) == 0:
        return np.zeros((image_height // bin_size, image_width // bin_size))
    return sum_array / float(total_messages)


def compute_histogram(u_list, v_list, weights_list, width: int, height: int, bin_size: int, total_count: Optional[int]):
    if len(u_list) == 0:
        heatmap = np.zeros((height // bin_size, width // bin_size))
        x_edges = np.linspace(0, width, width // bin_size + 1)
        y_edges = np.linspace(0, height, height // bin_size + 1)
        return heatmap, x_edges, y_edges
    heatmap, x_edges, y_edges = np.histogram2d(
        u_list, v_list,
        bins=[width // bin_size, height // bin_size],
        range=[[0, width], [0, height]],
        weights=weights_list if (weights_list is not None and len(weights_list) == len(u_list)) else None,
    )
    if total_count is not None and total_count > 0:
        heatmap = heatmap / float(total_count)
    return heatmap, x_edges, y_edges


def save_heatmap_image(heatmap: np.ndarray, x_edges, y_edges, title: str, out_path: str, vmin: float, vmax: float):
    plt.figure(figsize=(8, 4))
    plt.title(title)
    img = np.clip(heatmap, 0.0, 1.0)
    plt.imshow(
        img.T,
        origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect='auto',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label='Density per message')
    plt.xlabel('Image U (px)')
    plt.ylabel('Image V (px)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def combined_compare_figure(map_name: str, method_to_heatmap: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            fixed_scale: Optional[Tuple[float, float]], out_path: str):
    methods = sorted(method_to_heatmap.keys())
    fixed_scale = None
    if not methods:
        return
    # Unified scale across methods
    if fixed_scale is not None:
        vmin, vmax = fixed_scale
    else:
        nonzeros = []
        for m in methods:
            data = method_to_heatmap[m][0]
            nonzero = data[data > 0]
            if nonzero.size > 0:
                nonzeros.append(np.percentile(nonzero, 99))
        vmax = float(max(nonzeros)) if nonzeros else 1.0
        vmin = 0.0
    cols = min(3, len(methods))
    rows = int(np.ceil(len(methods) / cols))
    fig = plt.figure(figsize=(6 * cols, 4.5 * rows))
    fig.suptitle(f'{map_name} Feature Density (per-message)', fontsize=14, fontweight='bold')
    gs = GridSpec(rows, cols, hspace=0.25, wspace=0.2)
    for idx, method in enumerate(methods):
        r = idx // cols
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])
        heatmap, x_edges, y_edges = method_to_heatmap[method]
        img = np.clip(heatmap, 0.0, 1.0)
        im = ax.imshow(
            img.T,
            origin='lower',
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
        )
        ax.set_title(method)
        ax.set_xlabel('U (px)')
        ax.set_ylabel('V (px)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Density per message')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_error_vs_exploration_compare(map_name: str,
                                      method_to_pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                                      num_bins: int,
                                      y_max: float,
                                      out_path: str):
    """Plot Position Error vs Normalized Exploration Rate curve.
    - X-axis x: normalized exploration rate (0-1), uniformly binned
    - Y-axis y: conditional mean E[y|x] of position error
    - Variance bands: transparent bands ±1σ
    Data structure compatible:
      * New format samples: (pos_err_array, norm_rate_array, fallback_scalar) -> use norm_rate_array as x, pos_err_array as y
      * Old format samples: (x_array, y_array) -> directly use x as normalized rate, y as pos err
    """
    methods = sorted(method_to_pairs.keys())
    if not methods:
        return

    # x uniformly binned to [0,1]
    bin_edges = np.linspace(0.0, 1.0, int(max(5, num_bins)) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8, 5))
    plt.title(f'{map_name} Position Error vs Normalized Exploration Rate')

    for method in methods:
        samples = method_to_pairs[method]
        if not samples:
            continue
        x_all = []
        y_all = []
        for sample in samples:
            if len(sample) >= 3:
                # (pos_err_arr, norm_rate_arr, fallback)
                pos_arr = np.asarray(sample[0])
                rate_arr = np.asarray(sample[1])
                x_arr = rate_arr
                y_arr = pos_arr
            else:
                # Old format: (x_arr, y_arr)
                x_arr = np.asarray(sample[0])
                y_arr = np.asarray(sample[1])
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if np.any(mask):
                x_all.append(x_arr[mask])
                y_all.append(y_arr[mask])
        if not x_all:
            continue
        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
        # Bin and calculate mean and variance
        bin_index = np.digitize(x_all, bin_edges) - 1
        valid_mask = (bin_index >= 0) & (bin_index < len(bin_centers))
        bin_index = bin_index[valid_mask]
        y_valid = y_all[valid_mask]
        mean_values = np.full_like(bin_centers, np.nan, dtype=float)
        std_values = np.full_like(bin_centers, np.nan, dtype=float)
        for b in range(len(bin_centers)):
            in_bin = (bin_index == b)
            if np.any(in_bin):
                y_bin = y_valid[in_bin]
                mean_values[b] = float(np.mean(y_bin))
                std_values[b] = float(np.std(y_bin))
        if np.all(~np.isfinite(mean_values)):
            continue
        plt.plot(bin_centers, mean_values, label=method)
        lower = np.clip(mean_values - std_values, 0.0, y_max)
        upper = np.clip(mean_values + std_values, 0.0, y_max)
        plt.fill_between(bin_centers, lower, upper, alpha=0.2)

    plt.xlabel('Normalized Exploration Rate')
    plt.ylabel('Position Error (m)')
    plt.ylim(0.0, float(y_max))
    plt.xlim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

# Path parsing: infer map/method

DEFAULT_METHOD_KEYWORDS = [
    'rrt', 'rrtstar', 'rrt_star', 'informedrrt', 'informed_rrt',
    'astar', 'a_star', 'apf', 'teb', 'mpc', 'dwa', 'lqr', 'prm', 'rrtconnect',
    'gbplanner', 'explorer', 'frontier', 'random', 'mcts', 'rl', 'sac', 'ppo',
]


def tokenize_path(relpath: str) -> List[str]:
    base = os.path.basename(relpath)
    stem = os.path.splitext(base)[0]
    parts = re.split(r'[\\/\._\-]+', relpath.lower()) + re.split(r'[\\/\._\-]+', stem.lower())
    return [p for p in parts if p]


def map_name_from_filename(filename: str, exp_token: str) -> str:
    """Extract map name from bag filename. Prefer the part before exp_token; otherwise the stem."""
    stem = os.path.splitext(filename)[0]
    idx = stem.lower().find(exp_token.lower())
    if idx > 0:
        return stem[:idx]
    return stem


def infer_map_method(root: str, bag_path: str, assume_structure: str, method_keywords: List[str],
                     map_regex: Optional[str], method_regex: Optional[str],
                     experiments_root: Optional[str] = None,
                     map_from_filename: bool = True,
                     map_exp_token: str = '_exp') -> Tuple[str, str]:
    """
    Infer (map, method).
    Priority: ./experiments/{method}/*.bag
      - method: direct subfolder under experiments root
      - map: bag filename stem; if like name_exp_xxx, take the part before exp_token
    Fallback: assume_structure/regex/keywords.
    """
    abs_path = os.path.abspath(bag_path)
    parts = abs_path.replace('\\', '/').split('/')
    method = 'unknown'
    map_name = None

    # Priority: experiments/{method}/*.bag
    if experiments_root:
        try:
            if experiments_root in parts:
                idx = parts.index(experiments_root)
                if idx + 1 < len(parts):
                    method = parts[idx + 1]
                if map_from_filename:
                    map_name = map_name_from_filename(os.path.basename(abs_path), map_exp_token)
        except Exception:
            pass

    # Fallback to old logic if map/method not found
    rel = os.path.relpath(bag_path, root)
    rel_norm = rel.replace('\\', '/')

    if map_name is None:
        if map_regex:
            m = re.search(map_regex, rel_norm)
            if m and m.groups():
                map_name = m.group(1)
        if map_name is None:
            # Default to filename, if map_from_filename is enabled, take token part
            if map_from_filename:
                map_name = map_name_from_filename(os.path.basename(rel_norm), map_exp_token)
            else:
                map_name = os.path.splitext(os.path.basename(rel_norm))[0]

    if method == 'unknown':
        if assume_structure in ('map/method', 'method/map'):
            parts_rel = rel_norm.split('/')
            if len(parts_rel) >= 2:
                if assume_structure == 'map/method':
                    method = parts_rel[1]
                else:
                    method = parts_rel[0]
                    map_name = parts_rel[1]
        else:
            tokens = tokenize_path(rel_norm)
            for t in tokens:
                if t in method_keywords:
                    method = t
                    break
            if method_regex:
                mth = re.search(method_regex, rel_norm)
                if mth and mth.groups():
                    method = mth.group(1)

    return map_name, method


def build_success_ref_map(group_results: Dict[Tuple[str, str], List[dict]],
                          normalize_by_method: Optional[Dict[str, str]] = None) -> Dict[Tuple[str, str], float]:
    """Compute success reference per (map, method); optionally override target methods by reference method values per map."""
    success_ref_map: Dict[Tuple[str, str], float] = {}
    # Base computation
    for key, results in group_results.items():
        success_candidates: List[float] = []
        fallback_max_last_rate = 0.0
        for res in results:
            last_rate = float(res.get('last_exploration_rate', float('nan')))
            if np.isfinite(last_rate) and last_rate > fallback_max_last_rate:
                fallback_max_last_rate = last_rate
            if res.get('success_run', False) and np.isfinite(last_rate) and last_rate > 0:
                success_candidates.append(last_rate)
        if success_candidates:
            success_ref = float(max(success_candidates))
        else:
            success_ref = float(fallback_max_last_rate) if np.isfinite(fallback_max_last_rate) and fallback_max_last_rate > 0 else 1.0
        success_ref_map[key] = success_ref
    # Optional overrides per map
    if normalize_by_method:
        map_to_methods: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (map_name, method), value in success_ref_map.items():
            map_to_methods[map_name][method] = value
        for map_name in map_to_methods.keys():
            for target_method, ref_method in normalize_by_method.items():
                ref_value = map_to_methods[map_name].get(ref_method)
                if ref_value is not None:
                    success_ref_map[(map_name, target_method)] = ref_value
    return success_ref_map


def process_single_bag(bag_path: str, args):
    (
        exploration_data, gt_odom_data, vins_odom_data,
        r2d2_pointcloud_data, ov_msckf_pointcloud_data,
    ) = process_rosbag(bag_path)
    if not all([exploration_data, gt_odom_data, vins_odom_data]):
        return None
    result = synchronize_and_calculate_errors(exploration_data, gt_odom_data, vins_odom_data)
    if result[0] is None:
        return None
    (
        exploration_times, exploration_rates, error_times, position_errors, orientation_errors,
        gt_trajectory_times, gt_x, gt_y, gt_z,
        vins_trajectory_times, vins_x, vins_y, vins_z,
    ) = result
    # Bag duration (on exploration timeline)
    total_time = float(exploration_times[-1] - exploration_times[0]) if len(exploration_times) > 1 else 0.0
    # Interpolate exploration rate at error timestamps
    exp_rate_on_error_t = interpolate_exploration_rate(exploration_times, exploration_rates, error_times)
    failure_idx = None
    failure_time = None
    if len(position_errors) > 0:
        mask = position_errors >= args.threshold
        if np.any(mask):
            failure_idx = int(np.argmax(mask))
            failure_time = float(error_times[failure_idx])
    if failure_idx is not None:
        success_exploration_rate = float(exp_rate_on_error_t[failure_idx]) if not np.isnan(exp_rate_on_error_t[failure_idx]) else float('nan')
    else:
        # No failure: take last exploration rate
        success_exploration_rate = float(exploration_rates[-1]) if len(exploration_rates) > 0 else float('nan')
    # Success run condition
    success_run = (failure_idx is None) and (total_time < args.success_time_limit)
    # Normalize success rate to 100%
    # Defer normalization to global post-processing; keep raw value
    success_exploration_rate_raw = success_exploration_rate
    # Compute normalized exploration rate for curve x-axis
    # Defer x-axis normalization to global stage; return raw interpolation
    curve_x_norm = exp_rate_on_error_t
    # Point cloud heatmaps and per-message stats
    r2d2_u, r2d2_v, r2d2_intensity, ov_u, ov_v, r2d2_weight_per_msg, ov_count_per_msg = process_pointcloud_data(
        r2d2_pointcloud_data, ov_msckf_pointcloud_data
    )
    r2d2_msg_count = int(len(r2d2_weight_per_msg))
    ov_msg_count = int(len(ov_count_per_msg))
    r2d2_heat, r2d2_xe, r2d2_ye = compute_histogram(
        r2d2_u, r2d2_v, r2d2_intensity, args.image_width, args.image_height, args.bin_size, r2d2_msg_count
    )
    ov_heat, ov_xe, ov_ye = compute_histogram(
        ov_u, ov_v, None, args.image_width, args.image_height, args.bin_size, ov_msg_count
    )
    # Per-message averages
    ov_mean_per_msg = float(np.mean(ov_count_per_msg)) if len(ov_count_per_msg) > 0 else float('nan')
    r2d2_weight_mean_per_msg = float(np.mean(r2d2_weight_per_msg)) if len(r2d2_weight_per_msg) > 0 else float('nan')
    return {
        'total_time': total_time,
        'failure_time': failure_time,
        'success_exploration_rate': success_exploration_rate,
        'success_exploration_rate_raw': success_exploration_rate_raw,
        'success_run': success_run,
        'r2d2_heat': r2d2_heat,
        'r2d2_edges': (r2d2_xe, r2d2_ye),
        'ov_heat': ov_heat,
        'ov_edges': (ov_xe, ov_ye),
        'ov_mean_per_msg': ov_mean_per_msg,
        'r2d2_weight_mean_per_msg': r2d2_weight_mean_per_msg,
        'r2d2_msg_count': r2d2_msg_count,
        'ov_msg_count': ov_msg_count,
        # Curve plotting: x=normalized exploration rate, y=position error
        'curve_x': curve_x_norm,
        'curve_y': position_errors,
        'raw_exploration_times': exploration_times,
        'raw_exploration_rates': exploration_rates,
        'raw_exp_rate_on_error_t': exp_rate_on_error_t,
        'last_exploration_rate': float(exploration_rates[-1]) if len(exploration_rates) > 0 else float('nan'),
    }

# Main flow

def discover_bag_files(paths: List[str], recursive: bool = True) -> List[str]:
    bags = []
    for p in paths:
        if os.path.isfile(p) and p.endswith('.bag'):
            bags.append(os.path.abspath(p))
        elif os.path.isdir(p):
            if recursive:
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.endswith('.bag'):
                            bags.append(os.path.join(root, f))
            else:
                for f in os.listdir(p):
                    if f.endswith('.bag'):
                        bags.append(os.path.join(p, f))
    return sorted(list(set(bags)))


def main():
    parser = argparse.ArgumentParser(description='Batch process ROS bag files and aggregate statistics')
    parser.add_argument('inputs', nargs='+', help='Input bag files or directories')
    parser.add_argument('--threshold', type=float, default=2.0, help='Position error threshold (m)')
    parser.add_argument('--success-time-limit', type=float, default=200.0, help='Max time (s) to consider a run successful if no failure')
    parser.add_argument('--image-width', type=int, default=1280, help='Image width for heatmaps')
    parser.add_argument('--image-height', type=int, default=640, help='Image height for heatmaps')
    parser.add_argument('--bin-size', type=int, default=20, help='Bin size for heatmaps (px)')
    parser.add_argument('--r2d2-scale', nargs=2, type=float, metavar=('MIN', 'MAX'), default=[0.0, 0.1], help='Fixed scale for R2D2 heatmaps')
    parser.add_argument('--ov-scale', nargs=2, type=float, metavar=('MIN', 'MAX'), default=[0.0, 0.6], help='Fixed scale for OV-MSCKF heatmaps')
    parser.add_argument('--assume-structure', choices=['map/method', 'method/map', 'auto'], default='auto', help='How to infer map/method from path')
    parser.add_argument('--map-regex', type=str, default=None, help='Regex with group(1) to extract map from relative path')
    parser.add_argument('--method-regex', type=str, default=None, help='Regex with group(1) to extract method from relative path')
    parser.add_argument('--method-keywords', nargs='*', default=DEFAULT_METHOD_KEYWORDS, help='Keywords to detect methods when auto')
    parser.add_argument('--experiments-root', type=str, default='experiments', help='Folder name that contains method subfolders, e.g., experiments/{method}/*.bag')
    parser.add_argument('--map-from-filename', dest='map_from_filename', action='store_true', default=True, help='Infer map name from bag filename (default)')
    parser.add_argument('--no-map-from-filename', dest='map_from_filename', action='store_false', help='Disable inferring map name from filename')
    parser.add_argument('--map-exp-token', type=str, default='_exp', help="Token in filename that separates map name and experiment suffix, e.g., '_exp'")
    parser.add_argument('--normalize-success-rate', dest='normalize_success_rate', action='store_true', default=True, help='Normalize exploration rate so success equals 100% (default on)')
    parser.add_argument('--no-normalize-success-rate', dest='normalize_success_rate', action='store_false', help='Disable success-rate normalization')
    parser.add_argument('--only-map', dest='only_maps', action='append', default=None, help='Filter by specific map name(s); allow multiple or comma-separated')
    parser.add_argument('--output-dir', '-o', type=str, default='results_batch', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show figures')
    parser.add_argument('--no-recursive', action='store_true', help='Do not recurse into subdirectories')
    parser.add_argument('--curve-ymax', type=float, default=10.0, help='Y-axis max for position error vs exploration rate plot')
    parser.add_argument('--curve-bins', type=int, default=10, help='Number of bins along exploration rate for curve averaging')
    parser.add_argument('--completion-percent', type=float, default=0.8, help='Completion ratio (0-1) to define exploration time cutoff (e.g., 0.95)')
    args = parser.parse_args()

    # Filter by map name
    only_maps_set = None
    if args.only_maps:
        only_maps = []
        for token in args.only_maps:
            only_maps.extend([s for s in re.split(r'[,\s]+', token) if s])
        only_maps_set = set(only_maps)

    bag_files = discover_bag_files(args.inputs, recursive=(not args.no_recursive))
    if not bag_files:
        print('No .bag files found')
        return
    print(f'Found {len(bag_files)} bag files')

    # Compute overall_root for relative path fallback
    dir_inputs = [os.path.abspath(p) for p in args.inputs if os.path.isdir(p)]
    if dir_inputs:
        overall_root = os.path.commonpath(dir_inputs)
    else:
        overall_root = os.path.commonpath([os.path.dirname(b) for b in bag_files])

    # Grouping: (map, method) -> list of bag paths
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for bag in bag_files:
        map_name, method = infer_map_method(
            overall_root, bag, args.assume_structure, [k.lower() for k in args.method_keywords],
            args.map_regex, args.method_regex, experiments_root=args.experiments_root,
            map_from_filename=args.map_from_filename, map_exp_token=args.map_exp_token,
        )
        if only_maps_set is not None and map_name not in only_maps_set:
            continue
        groups[(map_name, method)].append(bag)

    # Containers
    stats = {}
    heatmaps_r2d2 = defaultdict(dict)
    heatmaps_ov = defaultdict(dict)
    # Curve data container: map -> method -> List[(pos_err_array, norm_rate_array, fallback_scalar)]
    curve_points = defaultdict(lambda: defaultdict(list))

    rows_success_rate = []  # map, method, n, mean, std
    rows_success_time = []  # map, method, n_success, mean_time, std_time
    rows_feat_per_ts = []   # map, method, n, ov_mean, ov_std, r2d2_weight_mean, r2d2_weight_std

    # Pass 1: 处理所有 bag，缓存原始结果
    group_results = defaultdict(list)
    print("\n-- Pass 1: 解析所有 bag 并缓存原始结果 --")
    for (map_name, method), bags in sorted(groups.items()):
        print(f"\n=== Group: map={map_name}, method={method}, count={len(bags)} ===")
        for bag in bags:
            print(f"-- Processing: {bag}")
            res = process_single_bag(bag, args)
            if res is None:
                print("Skip due to processing failure")
                continue
            group_results[(map_name, method)].append(res)

    # Pass 1.5: 预计算 (map, method) -> success_ref 字典
    success_ref_map: Dict[Tuple[str, str], float] = build_success_ref_map(group_results, normalize_by_method={'la_planner': 'fuel'})
    for (map_name, method), _results in group_results.items():
        print(f"Success ref for map={map_name}, method={method}: {success_ref_map[(map_name, method)]:.3f}")

    # Pass 2: 使用 success_ref_map 统一归一化并统计
    print("\n-- Pass 2: 全局归一化与统计 --")
    for (map_name, method), results in sorted(group_results.items()):
        success_ref = success_ref_map[(map_name, method)]
        success_rates = []
        success_times = []
        ov_means = []
        r2d2_weight_means = []
        # Heatmap 聚合（按消息数加权）
        sum_r2d2 = None
        sum_ov = None
        total_r2d2_msgs = 0
        total_ov_msgs = 0
        r2d2_edges = None
        ov_edges = None
        # 曲线样本
        method_curve_pairs = []

        for res in results:
            # 记录每消息特征统计
            if 'ov_mean_per_msg' in res and not np.isnan(res['ov_mean_per_msg']):
                ov_means.append(res['ov_mean_per_msg'])
            if 'r2d2_weight_mean_per_msg' in res and not np.isnan(res['r2d2_weight_mean_per_msg']):
                r2d2_weight_means.append(res['r2d2_weight_mean_per_msg'])

            # Heatmap 累积（按消息数加权）
            sum_r2d2 = accumulate_heatmap(sum_r2d2, res['r2d2_heat'], res.get('r2d2_msg_count', 0))
            sum_ov = accumulate_heatmap(sum_ov, res['ov_heat'], res.get('ov_msg_count', 0))
            if r2d2_edges is None:
                r2d2_edges = res['r2d2_edges']
            if ov_edges is None:
                ov_edges = res['ov_edges']
            total_r2d2_msgs += int(res.get('r2d2_msg_count', 0))
            total_ov_msgs += int(res.get('ov_msg_count', 0))

            # 统一归一化的成功探索率
            raw_success_rate = float(res.get('success_exploration_rate_raw', np.nan))
            if np.isfinite(raw_success_rate):
                rate_val = raw_success_rate
                if args.normalize_success_rate:
                    rate_val = float(np.clip(rate_val / success_ref, 0.0, 1.0))
                success_rates.append(rate_val)

            # 完成时间（到达 success_ref * completion_percent 的时间）
            t_arr = np.asarray(res.get('raw_exploration_times', []), dtype=float)
            r_arr = np.asarray(res.get('raw_exploration_rates', []), dtype=float)
            if t_arr.size > 0 and r_arr.size > 0 and np.isfinite(success_ref) and success_ref > 0:
                cutoff = float(success_ref * float(args.completion_percent))
                idxs = np.where(r_arr >= cutoff)[0]
                if idxs.size > 0:
                    t_comp = float(t_arr[int(idxs[0])] - t_arr[0])
                    if np.isfinite(t_comp) and t_comp >= 0:
                        success_times.append(t_comp)

            # 曲线样本（pos_err vs 归一化 rate）
            if method != 'la_planner':
                if 'raw_exp_rate_on_error_t' in res and 'curve_y' in res:
                    try:
                        norm_rate_arr = np.clip(np.asarray(res['raw_exp_rate_on_error_t'], dtype=float) / success_ref, 0.0, 1.0)
                    except Exception:
                        norm_rate_arr = np.asarray(res['curve_x'])
                    pos_err_arr = np.asarray(res['curve_y'])
                    fallback_val = float(np.clip(raw_success_rate / success_ref, 0.0, 1.0)) if np.isfinite(raw_success_rate) else float('nan')
                    if norm_rate_arr is not None and norm_rate_arr.size > 0 and pos_err_arr.size > 0:
                        method_curve_pairs.append((pos_err_arr, norm_rate_arr, fallback_val))

        # 每消息密度平均
        avg_r2d2 = average_heatmap(sum_r2d2, total_r2d2_msgs, args.image_width, args.image_height, args.bin_size)
        avg_ov = average_heatmap(sum_ov, total_ov_msgs, args.image_width, args.image_height, args.bin_size)
        if r2d2_edges is None:
            r2d2_edges = default_edges(args.image_width, args.image_height, args.bin_size)
        if ov_edges is None:
            ov_edges = default_edges(args.image_width, args.image_height, args.bin_size)

        heatmaps_r2d2[map_name][method] = (avg_r2d2, r2d2_edges[0], r2d2_edges[1])
        heatmaps_ov[map_name][method] = (avg_ov, ov_edges[0], ov_edges[1])
        # 存曲线点
        if method_curve_pairs:
            curve_points[map_name][method] = method_curve_pairs
        # 记录统计
        if success_rates:
            rows_success_rate.append([
                map_name, method, len(success_rates), float(np.mean(success_rates)), float(np.std(success_rates))
            ])
        else:
            rows_success_rate.append([map_name, method, 0, float('nan'), float('nan')])
        if success_times:
            rows_success_time.append([
                map_name, method, len(success_times), float(np.mean(success_times)), float(np.std(success_times))
            ])
        else:
            rows_success_time.append([map_name, method, 0, float('nan'), float('nan')])
        if ov_means or r2d2_weight_means:
            n_feat = max(len(ov_means), len(r2d2_weight_means))
            rows_feat_per_ts.append([
                map_name, method,
                n_feat,
                float(np.mean(ov_means)) if ov_means else float('nan'),
                float(np.std(ov_means)) if ov_means else float('nan'),
                float(np.mean(r2d2_weight_means)) if r2d2_weight_means else float('nan'),
                float(np.std(r2d2_weight_means)) if r2d2_weight_means else float('nan'),
            ])
        else:
            rows_feat_per_ts.append([map_name, method, 0, float('nan'), float('nan'), float('nan'), float('nan')])

    # Output directory
    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)

    # Save heatmap comparison figures
    for map_name, method_map in heatmaps_r2d2.items():
        combined_r2d2_out = os.path.join(outdir, f"{map_name}__compare__r2d2.png")
        combined_ov_out = os.path.join(outdir, f"{map_name}__compare__ov.png")
        combined_compare_figure(map_name + ' - R2D2', method_map, tuple(args.r2d2_scale), combined_r2d2_out)
        combined_compare_figure(map_name + ' - OV', heatmaps_ov[map_name], tuple(args.ov_scale), combined_ov_out)

    # Plot Position Error vs Normalized Exploration Rate (per map)
    for map_name, method_pairs in curve_points.items():
        out_path_curve = os.path.join(outdir, f"{map_name}__compare__poserr_vs_exploration.png")
        plot_error_vs_exploration_compare(map_name, method_pairs, args.curve_bins, args.curve_ymax, out_path_curve)

    # Save CSV/JSON
    import csv
    csv_rate = os.path.join(outdir, 'success_exploration_rate.csv')
    with open(csv_rate, 'w', newline='') as f:
        writer = csv.writer(f)
        header_rate = 'mean_success_exploration_rate_norm' if args.normalize_success_rate else 'mean_success_exploration_rate'
        writer.writerow(['map', 'method', 'n', header_rate, 'std'])
        writer.writerows(rows_success_rate)
    csv_time = os.path.join(outdir, 'success_exploration_time.csv')
    with open(csv_time, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['map', 'method', 'n_success', 'mean_time_s', 'std'])
        writer.writerows(rows_success_time)
    csv_feat = os.path.join(outdir, 'features_per_timestamp.csv')
    with open(csv_feat, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['map', 'method', 'n_bags', 'ov_mean_per_timestamp', 'ov_std', 'r2d2_weight_mean_per_timestamp', 'r2d2_weight_std'])
        writer.writerows(rows_feat_per_ts)
    meta = {
        'threshold_m': args.threshold,
        'success_time_limit_s': args.success_time_limit,
        'image': {
            'width': args.image_width,
            'height': args.image_height,
            'bin_size': args.bin_size,
            'r2d2_scale': args.r2d2_scale,
            'ov_scale': args.ov_scale,
        },
        'groups': {f'{m}__{d}': len(bags) for (m, d), bags in groups.items()},
    }
    with open(os.path.join(outdir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Visualization (log)
    if args.show:
        for map_name in heatmaps_r2d2.keys():
            r2d2_cmp = os.path.join(outdir, f"{map_name}__compare__r2d2.png")
            ov_cmp = os.path.join(outdir, f"{map_name}__compare__ov.png")
            print(f"Saved compare figures: {r2d2_cmp}, {ov_cmp}")
            curve_cmp = os.path.join(outdir, f"{map_name}__compare__poserr_vs_exploration.png")
            print(f"Saved curve figure: {curve_cmp}")
            break

    print(f"\nAll done. Results saved to: {outdir}")


if __name__ == '__main__':
    main() 