#!/usr/bin/env python3
"""
Batch ROS Bag Post-Processing Script

功能:
- 递归读取一个或多个输入路径下的所有 .bag 文件
- 依据路径推断 map 与 planning method 分组: 默认假设目录结构为 map/method/*.bag，可通过参数调整
- 对每个 bag:
  * 复用单包后处理逻辑: 同步 GT 与 VINS，刚体配准对齐，计算位置/姿态误差
  * 设定位置误差阈值 threshold，找到首次超过阈值的时间，记下失效前的 exploration rate（插值）
  * 统计总时长、是否在限定时间内无失效（成功探索）
  * 生成 R2D2 与 OV-MSCKF 的 2D 热力图计数（按总时长进行每秒归一化）
- 聚合到 (map, method) 维度:
  * 输出“失效前成功 exploration rate”的均值/方差
  * 对成功探索（总时长 < success_time_limit 且未失效）的样本，统计平均探索时间
  * 生成并保存总体平均热力图（每秒密度），并提供按地图对比的并排图
- 输出 CSV/JSON 汇总

依赖:
- rosbag, rospy (只读)
- numpy, scipy, matplotlib

示例:
python3 scripts/post_batch.py /data/rosbags \
  --threshold 2.0 --success-time-limit 200 \
  --assume-structure map/method \
  --output-dir results_batch
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

# -----------------------------
# 复用的基础函数（改自 scripts/post.py）
# -----------------------------

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
                elif topic == '/r2d2/point_cloud':
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
    gt_initial_pos = gt_first_msg.pose.pose.position
    vins_initial_pos = vins_first_msg.pose.pose.position
    gt_initial_quat = gt_first_msg.pose.pose.orientation
    vins_initial_quat = vins_first_msg.pose.pose.orientation
    r_gt_initial = R.from_quat([gt_initial_quat.x, gt_initial_quat.y, gt_initial_quat.z, gt_initial_quat.w])
    r_vins_initial = R.from_quat([vins_initial_quat.x, vins_initial_quat.y, vins_initial_quat.z, vins_initial_quat.w])
    r_transform = r_gt_initial * r_vins_initial.inv()
    gt_pos_array = np.array([gt_initial_pos.x, gt_initial_pos.y, gt_initial_pos.z])
    vins_pos_array = np.array([vins_initial_pos.x, vins_initial_pos.y, vins_initial_pos.z])
    vins_pos_rotated = r_transform.apply(vins_pos_array)
    translation = gt_pos_array - vins_pos_rotated
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
    # 每条消息的统计
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
        # 记录本条消息的加权和
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
        # 记录本条消息的数量
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

# -----------------------------
# 聚合/分组与绘图辅助
# -----------------------------

def compute_histogram(u_list, v_list, weights_list, width: int, height: int, bin_size: int, total_time: Optional[float]):
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
    if total_time is not None and total_time > 0:
        heatmap = heatmap / total_time
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
    plt.colorbar(label='Density per Second')
    plt.xlabel('Image U (px)')
    plt.ylabel('Image V (px)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def combined_compare_figure(map_name: str, method_to_heatmap: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            fixed_scale: Optional[Tuple[float, float]], out_path: str):
    methods = sorted(method_to_heatmap.keys())
    if not methods:
        return
    # 统一尺度
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
    fig.suptitle(f'{map_name} Feature Density (per-second)', fontsize=14, fontweight='bold')
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
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_error_vs_exploration_compare(map_name: str,
                                      method_to_pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                                      num_bins: int,
                                      x_max: float,
                                      out_path: str):
    """绘制 y(归一化 exploration rate) vs x(position error) 曲线。
    统计口径：对每个阈值 x，按每个 bag 的“首次 position error ≥ x”的时刻取归一化 exploration rate；
    若 bag 从未越过该 x，则回退为该 bag 的成功归一化值（或末帧/最大归一化规则的一致值）。
    这样在 x=阈值 时的均值等效于 mean_success_exploration_rate_norm。
    注意：此函数假定传入的数据结构为每方法的样本三元组列表：(pos_err_array, norm_rate_array, fallback_scalar)。
    """
    # 为了兼容原调用，此处把 method_to_pairs 视为 List[(pos_err, norm_rate, fallback)]
    methods = sorted(method_to_pairs.keys())
    if not methods:
        return
    # 以 [0, x_max] 均匀分箱作为阈值集合
    x_min = 0.0
    if not np.isfinite(x_max) or x_max <= x_min:
        return
    bin_edges = np.linspace(x_min, float(x_max), int(max(5, num_bins)) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8, 5))
    plt.title(f'{map_name} Normalized Exploration Rate vs Position Error')

    for method in methods:
        samples = method_to_pairs[method]
        if not samples:
            continue
        # 对每个阈值，收集每个 bag 的首次越阈值的 y；若未越阈，用 fallback
        mean_values = np.full_like(bin_centers, np.nan, dtype=float)
        std_values = np.full_like(bin_centers, np.nan, dtype=float)
        for b_idx in range(len(bin_centers)):
            thr = bin_centers[b_idx]
            values = []
            for sample in samples:
                # 兼容两种形态：(pos, rate, fallback) 或旧的 (x, y)
                if len(sample) >= 3:
                    pos_arr, rate_arr, fallback_val = sample[0], sample[1], sample[2]
                else:
                    # 旧数据结构：x=norm_rate, y=pos_err；无法等效，只能跳过
                    continue
                if pos_arr is None or rate_arr is None:
                    continue
                pos_flat = np.asarray(pos_arr).ravel()
                rate_flat = np.asarray(rate_arr).ravel()
                mask = np.isfinite(pos_flat) & np.isfinite(rate_flat)
                pos_flat = pos_flat[mask]
                rate_flat = rate_flat[mask]
                if pos_flat.size == 0:
                    continue
                crossed = np.where(pos_flat >= thr)[0]
                if crossed.size > 0:
                    idx = int(crossed[0])
                    values.append(float(rate_flat[idx]))
                else:
                    # 回退值（成功归一化率）
                    values.append(float(fallback_val))
            if values:
                mean_values[b_idx] = float(np.mean(values))
                std_values[b_idx] = float(np.std(values))
        if np.all(~np.isfinite(mean_values)):
            continue
        plt.plot(bin_centers, mean_values, label=method)
        lower = np.clip(mean_values - std_values, 0.0, 1.0)
        upper = np.clip(mean_values + std_values, 0.0, 1.0)
        plt.fill_between(bin_centers, lower, upper, alpha=0.2)

    plt.xlabel('Position Error (m)')
    plt.ylabel('Normalized Exploration Rate')
    plt.xlim(0.0, float(x_max))
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

# -----------------------------
# 路径解析: 推断 map / method
# -----------------------------

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
    """从 bag 文件名中提取地图名，优先取 exp_token 之前的部分，否则为去扩展名的完整文件名"""
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
    推断 (map, method)。
    优先支持结构: ./experiments/{method}/*.bag
      - method: experiments 根目录下的直接子目录名
      - map: 默认使用 bag 文件名（去扩展名），若命名形如 name_exp_xxx 则取 exp_token 之前部分
    失败时回退到原有逻辑（assume_structure/regex/keywords）。
    """
    abs_path = os.path.abspath(bag_path)
    parts = abs_path.replace('\\', '/').split('/')
    method = 'unknown'
    map_name = None

    # 优先: experiments/{method}/*.bag
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

    # 若未得到 map/method，回退到旧逻辑
    rel = os.path.relpath(bag_path, root)
    rel_norm = rel.replace('\\', '/')

    if map_name is None:
        if map_regex:
            m = re.search(map_regex, rel_norm)
            if m and m.groups():
                map_name = m.group(1)
        if map_name is None:
            # 默认用文件名，若启用 map_from_filename 则按 token 截取
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

# -----------------------------
# 单包计算与分组聚合
# -----------------------------

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
    # bag 总时长（以 exploration 时间轴）
    total_time = float(exploration_times[-1] - exploration_times[0]) if len(exploration_times) > 1 else 0.0
    # 在误差时间戳上插值 exploration rate
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
        # 未失效：取最后一个 exploration rate
        success_exploration_rate = float(exploration_rates[-1]) if len(exploration_rates) > 0 else float('nan')
    # 成功探索判定：在限定时间内（< success_time_limit）且未失效
    success_run = (failure_idx is None) and (total_time < args.success_time_limit)
    # 归一化：成功等于100%
    success_exploration_rate_raw = success_exploration_rate
    success_ref = None
    if args.normalize_success_rate:
        if success_run:
            success_ref = float(exploration_rates[-1]) if len(exploration_rates) > 0 else None
        else:
            success_ref = float(np.nanmax(exploration_rates)) if len(exploration_rates) > 0 else None
        if success_ref is None or not np.isfinite(success_ref) or success_ref <= 0:
            success_ref = 1.0
        success_exploration_rate = float(np.clip(success_exploration_rate_raw / success_ref, 0.0, 1.0)) if np.isfinite(success_exploration_rate_raw) else float('nan')
    # 计算曲线横轴需要的“归一化 exploration rate”（独立于是否输出归一化成功率）
    curve_ref = None
    if len(exploration_rates) > 0:
        if success_run:
            curve_ref = float(exploration_rates[-1])
        else:
            curve_ref = float(np.nanmax(exploration_rates))
    if curve_ref is None or not np.isfinite(curve_ref) or curve_ref <= 0:
        curve_ref = 1.0
    curve_x_norm = None
    try:
        curve_x_norm = np.clip(np.asarray(exp_rate_on_error_t, dtype=float) / float(curve_ref), 0.0, 1.0)
    except Exception:
        curve_x_norm = exp_rate_on_error_t
    # 点云热力图与每消息统计
    r2d2_u, r2d2_v, r2d2_intensity, ov_u, ov_v, r2d2_weight_per_msg, ov_count_per_msg = process_pointcloud_data(
        r2d2_pointcloud_data, ov_msckf_pointcloud_data
    )
    r2d2_heat, r2d2_xe, r2d2_ye = compute_histogram(
        r2d2_u, r2d2_v, r2d2_intensity, args.image_width, args.image_height, args.bin_size, total_time
    )
    ov_heat, ov_xe, ov_ye = compute_histogram(
        ov_u, ov_v, None, args.image_width, args.image_height, args.bin_size, total_time
    )
    # 每时间戳平均: OV 为每消息特征数均值；R2D2 为每消息加权和均值
    ov_mean_per_ts = float(np.mean(ov_count_per_msg)) if len(ov_count_per_msg) > 0 else float('nan')
    r2d2_weight_mean_per_ts = float(np.mean(r2d2_weight_per_msg)) if len(r2d2_weight_per_msg) > 0 else float('nan')
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
        'ov_mean_per_ts': ov_mean_per_ts,
        'r2d2_weight_mean_per_ts': r2d2_weight_mean_per_ts,
        'curve_x': curve_x_norm,
        'curve_y': position_errors,
    }

# -----------------------------
# 主流程
# -----------------------------

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
    parser.add_argument('--r2d2-scale', nargs=2, type=float, metavar=('MIN', 'MAX'), default=[0.0, 0.2], help='Fixed scale for R2D2 heatmaps')
    parser.add_argument('--ov-scale', nargs=2, type=float, metavar=('MIN', 'MAX'), default=[0.0, 0.8], help='Fixed scale for OV-MSCKF heatmaps')
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
    parser.add_argument('--only-map', dest='only_maps', action='append', default=None, help='仅处理指定 map 名（可多次指定或用逗号分隔）')
    parser.add_argument('--output-dir', '-o', type=str, default='results_batch', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show figures')
    parser.add_argument('--no-recursive', action='store_true', help='Do not recurse into subdirectories')
    # 新增：曲线绘图参数
    parser.add_argument('--curve-ymax', type=float, default=200.0, help='Y-axis max for position error vs exploration rate plot')
    parser.add_argument('--curve-bins', type=int, default=20, help='Number of bins along exploration rate for curve averaging')
    args = parser.parse_args()

    # 仅处理指定 map 名过滤
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

    # 计算 overall_root 作为回退相对根
    dir_inputs = [os.path.abspath(p) for p in args.inputs if os.path.isdir(p)]
    if dir_inputs:
        overall_root = os.path.commonpath(dir_inputs)
    else:
        overall_root = os.path.commonpath([os.path.dirname(b) for b in bag_files])

    # 分组: (map, method) -> list of bag paths
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

    # 统计容器
    stats = {}
    heatmaps_r2d2 = defaultdict(dict)  # map -> method -> (heat, x_edges, y_edges)
    heatmaps_ov = defaultdict(dict)
    # 新增：曲线数据容器 map -> method -> List[(pos_err_array, norm_rate_array, fallback_scalar)]
    curve_points = defaultdict(lambda: defaultdict(list))

    rows_success_rate = []  # map, method, n, mean, std
    rows_success_time = []  # map, method, n_success, mean_time, std_time
    rows_feat_per_ts = []   # map, method, n, ov_mean, ov_std, r2d2_weight_mean, r2d2_weight_std

    for (map_name, method), bags in sorted(groups.items()):
        print(f"\n=== Group: map={map_name}, method={method}, count={len(bags)} ===")
        success_rates = []
        success_times = []
        ov_means = []
        r2d2_weight_means = []
        # 聚合热力图：按总时间权重的每秒密度平均
        sum_r2d2 = None
        sum_ov = None
        total_time_sum = 0.0
        r2d2_edges = None
        ov_edges = None
        method_curve_pairs = []
        for bag in bags:
            print(f"-- Processing: {bag}")
            res = process_single_bag(bag, args)
            if res is None:
                print("Skip due to processing failure")
                continue
            # 使用归一化或原始的成功探索率
            rate_val = res['success_exploration_rate'] if args.normalize_success_rate else res.get('success_exploration_rate_raw', res['success_exploration_rate'])
            if not np.isnan(rate_val):
                success_rates.append(rate_val)
            if res['success_run']:
                success_times.append(res['total_time'])
            if 'ov_mean_per_ts' in res and not np.isnan(res['ov_mean_per_ts']):
                ov_means.append(res['ov_mean_per_ts'])
            if 'r2d2_weight_mean_per_ts' in res and not np.isnan(res['r2d2_weight_mean_per_ts']):
                r2d2_weight_means.append(res['r2d2_weight_mean_per_ts'])
            # 热力图累计（已是每秒密度，需用时间加权求平均: sum(counts) 与 sum(time)）
            if sum_r2d2 is None:
                sum_r2d2 = np.array(res['r2d2_heat'], dtype=np.float64)
                r2d2_edges = res['r2d2_edges']
            else:
                sum_r2d2 += np.array(res['r2d2_heat'], dtype=np.float64)
            if sum_ov is None:
                sum_ov = np.array(res['ov_heat'], dtype=np.float64)
                ov_edges = res['ov_edges']
            else:
                sum_ov += np.array(res['ov_heat'], dtype=np.float64)
            total_time_sum += max(res['total_time'], 1e-6)
            # 曲线数据收集：存为(position_error, normalized_rate, fallback)
            if 'curve_x' in res and 'curve_y' in res:
                norm_rate_arr = np.asarray(res['curve_x'])  # normalized exploration rate over error timestamps
                pos_err_arr = np.asarray(res['curve_y'])     # position error over error timestamps
                fallback_val = float(res.get('success_exploration_rate', np.nan))
                if norm_rate_arr.size > 0 and pos_err_arr.size > 0:
                    method_curve_pairs.append((pos_err_arr, norm_rate_arr, fallback_val))
        # 形成平均每秒密度
        if sum_r2d2 is None:
            avg_r2d2 = np.zeros((args.image_height // args.bin_size, args.image_width // args.bin_size))
            r2d2_edges = (
                np.linspace(0, args.image_width, args.image_width // args.bin_size + 1),
                np.linspace(0, args.image_height, args.image_height // args.bin_size + 1),
            )
        else:
            n_valid = max(1, len(bags))
            avg_r2d2 = sum_r2d2 / n_valid
        if sum_ov is None:
            avg_ov = np.zeros((args.image_height // args.bin_size, args.image_width // args.bin_size))
            ov_edges = (
                np.linspace(0, args.image_width, args.image_width // args.bin_size + 1),
                np.linspace(0, args.image_height, args.image_height // args.bin_size + 1),
            )
        else:
            n_valid = max(1, len(bags))
            avg_ov = sum_ov / n_valid
        heatmaps_r2d2[map_name][method] = (avg_r2d2, r2d2_edges[0], r2d2_edges[1])
        heatmaps_ov[map_name][method] = (avg_ov, ov_edges[0], ov_edges[1])
        # 保存该组的曲线点
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

    # 输出目录
    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)

    # 保存热力图（仅对比图）
    for map_name, method_map in heatmaps_r2d2.items():
        # 仅生成地图内方法对比的并排图
        combined_r2d2_out = os.path.join(outdir, f"{map_name}__compare__r2d2.png")
        combined_ov_out = os.path.join(outdir, f"{map_name}__compare__ov.png")
        combined_compare_figure(map_name + ' - R2D2', method_map, tuple(args.r2d2_scale), combined_r2d2_out)
        combined_compare_figure(map_name + ' - OV', heatmaps_ov[map_name], tuple(args.ov_scale), combined_ov_out)

    # 新增：绘制 Position Error vs Exploration Rate 对比曲线
    for map_name, method_pairs in curve_points.items():
        out_path_curve = os.path.join(outdir, f"{map_name}__compare__poserr_vs_exploration.png")
        # 使用曲线y轴上限参数作为position error的x轴上限（保持可控），也可另增参数
        plot_error_vs_exploration_compare(map_name, method_pairs, args.curve_bins, args.curve_ymax, out_path_curve)

    # 保存 CSV/JSON
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

    # 可视化
    if args.show:
        # 简单展示任意一个并排对比图
        for map_name in heatmaps_r2d2.keys():
            r2d2_cmp = os.path.join(outdir, f"{map_name}__compare__r2d2.png")
            ov_cmp = os.path.join(outdir, f"{map_name}__compare__ov.png")
            print(f"Saved compare figures: {r2d2_cmp}, {ov_cmp}")
            # 新增：曲线对比图路径提示
            curve_cmp = os.path.join(outdir, f"{map_name}__compare__poserr_vs_exploration.png")
            print(f"Saved curve figure: {curve_cmp}")
            break

    print(f"\nAll done. Results saved to: {outdir}")


if __name__ == '__main__':
    main() 