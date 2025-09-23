#!/usr/bin/env python3
"""
ROS Bag Post-Processing Script

This script processes a ROS bag file containing:
- /exploration_rate (std_msgs/Float64 or similar float message)
- /kingfisher/ground_truth/odometry (nav_msgs/Odometry)
- /ov_msckf/odometry (nav_msgs/Odometry)
- /r2d2/point_cloud (sensor_msgs/PointCloud2)
- /ov_msckf/loop_feats (sensor_msgs/PointCloud)

It generates five plots:
1. Exploration rate vs time
2. Odometry error vs time
3. Odometry error vs exploration rate
4. 2D heatmap of feature point density
5. Trajectory comparison (xyz positions)
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def quaternion_to_euler(quat):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    return r.as_euler('xyz', degrees=False)

def calculate_position_error(gt_pos, est_pos):
    """Calculate Euclidean distance error between two positions"""
    dx = gt_pos.x - est_pos.x
    dy = gt_pos.y - est_pos.y
    dz = gt_pos.z - est_pos.z
    return np.sqrt(dx**2 + dy**2 + dz**2)

def calculate_orientation_error(gt_quat, est_quat):
    """Calculate orientation error in radians"""
    # Handle both quaternion objects and numpy arrays
    if hasattr(gt_quat, 'x'):
        # ROS quaternion object
        gt_quat_array = [gt_quat.x, gt_quat.y, gt_quat.z, gt_quat.w]
    else:
        # Numpy array
        gt_quat_array = gt_quat
    
    if hasattr(est_quat, 'x'):
        # ROS quaternion object
        est_quat_array = [est_quat.x, est_quat.y, est_quat.z, est_quat.w]
    else:
        # Numpy array
        est_quat_array = est_quat
    
    # Convert quaternions to rotation matrices
    r_gt = R.from_quat(gt_quat_array)
    r_est = R.from_quat(est_quat_array)
    
    # Calculate relative rotation
    r_rel = r_gt * r_est.inv()
    
    # Get angle of rotation (magnitude of rotation)
    angle = r_rel.magnitude()
    return angle

def process_rosbag(bag_path):
    """Process ROS bag and extract data"""
    print(f"Processing ROS bag: {bag_path}")
    
    # Data containers
    exploration_data = []
    gt_odom_data = []
    vins_odom_data = []
    r2d2_pointcloud_data = []
    ov_msckf_pointcloud_data = []
    
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Get bag info
            info = bag.get_type_and_topic_info()
            topics = info.topics.keys()
            print(f"Available topics: {list(topics)}")
            
            # Read messages
            for topic, msg, t in bag.read_messages():
                timestamp = t.to_sec()
                
                if topic == '/exploration_rate':
                    # Handle different float message types
                    if hasattr(msg, 'data'):
                        rate = msg.data
                    else:
                        rate = float(msg)
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
    """Synchronize data and calculate odometry errors"""
    if not all([exploration_data, gt_odom_data, vins_odom_data]):
        print("Error: Missing data for one or more topics")
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    # Convert to numpy arrays for easier processing
    exploration_times = np.array([d[0] for d in exploration_data])
    exploration_rates = np.array([d[1] for d in exploration_data])
    
    gt_times = np.array([d[0] for d in gt_odom_data])
    vins_times = np.array([d[0] for d in vins_odom_data])
    
    # Find common time range
    start_time = max(exploration_times.min(), gt_times.min(), vins_times.min())
    end_time = min(exploration_times.max(), gt_times.max(), vins_times.max())
    
    print(f"Common time range: {start_time:.2f} to {end_time:.2f} seconds")
    
    # Filter data to common time range
    exp_mask = (exploration_times >= start_time) & (exploration_times <= end_time)
    gt_mask = (gt_times >= start_time) & (gt_times <= end_time)
    vins_mask = (vins_times >= start_time) & (vins_times <= end_time)
    
    exploration_times_filtered = exploration_times[exp_mask]
    exploration_rates_filtered = exploration_rates[exp_mask]
    
    # Find the first timestamp when both odometries are available
    first_gt_time = gt_times[gt_mask][0] if np.any(gt_mask) else None
    first_vins_time = vins_times[vins_mask][0] if np.any(vins_mask) else None
    
    if first_gt_time is None or first_vins_time is None:
        print("Error: Cannot find first odometry messages")
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    # Find the first timestamp when both are available (use the later one)
    first_common_time = max(first_gt_time, first_vins_time)
    print(f"First common odometry timestamp: {first_common_time:.2f} seconds")
    
    # Find the corresponding messages at the first common time
    gt_first_idx = np.argmin(np.abs(gt_times - first_common_time))
    vins_first_idx = np.argmin(np.abs(vins_times - first_common_time))
    
    # Get the first messages for alignment
    gt_first_msg = gt_odom_data[gt_first_idx][1]
    vins_first_msg = vins_odom_data[vins_first_idx][1]
    
    # Calculate initial offset for alignment
    # We want to transform VINS trajectory to align with GT trajectory
    # The transformation should map VINS initial pose to GT initial pose
    
    # Calculate the transformation from VINS initial pose to GT initial pose
    # This is a rigid body transformation (rotation + translation)
    
    # First, get the initial poses
    gt_initial_pos = gt_first_msg.pose.pose.position
    vins_initial_pos = vins_first_msg.pose.pose.position
    
    gt_initial_quat = gt_first_msg.pose.pose.orientation
    vins_initial_quat = vins_first_msg.pose.pose.orientation
    
    # Convert to rotation matrices
    r_gt_initial = R.from_quat([gt_initial_quat.x, gt_initial_quat.y, 
                                gt_initial_quat.z, gt_initial_quat.w])
    r_vins_initial = R.from_quat([vins_initial_quat.x, vins_initial_quat.y,
                                  vins_initial_quat.z, vins_initial_quat.w])
    
    # Calculate the rotation that transforms VINS orientation to GT orientation
    # R_transform * R_vins_initial = R_gt_initial
    # Therefore: R_transform = R_gt_initial * R_vins_initial.inv()
    r_transform = r_gt_initial * r_vins_initial.inv()
    
    # Calculate the translation that transforms VINS position to GT position
    # We need to apply the rotation first, then add translation
    # Let's calculate this step by step
    
    # The transformation equation is:
    # GT_position = R_transform * VINS_position + translation
    # Therefore: translation = GT_position - R_transform * VINS_position
    
    # Convert positions to numpy arrays for easier calculation
    gt_pos_array = np.array([gt_initial_pos.x, gt_initial_pos.y, gt_initial_pos.z])
    vins_pos_array = np.array([vins_initial_pos.x, vins_initial_pos.y, vins_initial_pos.z])
    
    # Apply rotation to VINS position
    vins_pos_rotated = r_transform.apply(vins_pos_array)
    
    # Calculate translation
    translation = gt_pos_array - vins_pos_rotated
    
    print(f"Initial position offset: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}) m")
    print(f"Initial orientation offset: {np.degrees(r_transform.magnitude()):.2f}°")
    
    # Calculate errors at VINS timestamps (starting from the first common time)
    error_times = []
    position_errors = []
    orientation_errors = []
    
    # Store trajectory data for plotting
    gt_trajectory_times = []
    gt_trajectory_x = []
    gt_trajectory_y = []
    gt_trajectory_z = []
    vins_trajectory_times = []
    vins_trajectory_x = []
    vins_trajectory_y = []
    vins_trajectory_z = []
    
    for i, (vins_time, vins_msg) in enumerate(vins_odom_data):
        if vins_time < first_common_time or vins_time > end_time:
            continue
            
        # Find closest ground truth message
        gt_diffs = np.abs(gt_times - vins_time)
        gt_idx = np.argmin(gt_diffs)
        
        # Skip if ground truth is too far in time (> 0.1 seconds)
        if gt_diffs[gt_idx] > 0.1:
            continue
            
        gt_msg = gt_odom_data[gt_idx][1]
        
        # Apply alignment transformation to VINS message
        # Transform VINS position: apply rotation first, then add translation
        vins_pos_array = np.array([vins_msg.pose.pose.position.x,
                                  vins_msg.pose.pose.position.y,
                                  vins_msg.pose.pose.position.z])
        
        # Apply rotation
        vins_pos_rotated = r_transform.apply(vins_pos_array)
        
        # Apply translation
        vins_pos_transformed = vins_pos_rotated + translation
        
        # Create aligned position object
        aligned_vins_pos = Point()
        aligned_vins_pos.x = vins_pos_transformed[0]
        aligned_vins_pos.y = vins_pos_transformed[1]
        aligned_vins_pos.z = vins_pos_transformed[2]
        
        # Transform VINS orientation using the rotation transform
        r_vins_current = R.from_quat([vins_msg.pose.pose.orientation.x,
                                     vins_msg.pose.pose.orientation.y,
                                     vins_msg.pose.pose.orientation.z,
                                     vins_msg.pose.pose.orientation.w])
        
        # Apply the rotation transform
        r_vins_aligned = r_transform * r_vins_current
        
        # Calculate errors between aligned VINS and ground truth
        pos_error = calculate_position_error(gt_msg.pose.pose.position, aligned_vins_pos)
        
        # For orientation, compare the aligned VINS with GT
        orient_error = calculate_orientation_error(
            gt_msg.pose.pose.orientation,
            r_vins_aligned.as_quat()
        )
        
        error_times.append(vins_time)
        position_errors.append(pos_error)
        orientation_errors.append(orient_error)
        
        # Store trajectory data
        gt_trajectory_times.append(gt_times[gt_idx])
        gt_trajectory_x.append(gt_msg.pose.pose.position.x)
        gt_trajectory_y.append(gt_msg.pose.pose.position.y)
        gt_trajectory_z.append(gt_msg.pose.pose.position.z)
        
        vins_trajectory_times.append(vins_time)
        vins_trajectory_x.append(aligned_vins_pos.x)
        vins_trajectory_y.append(aligned_vins_pos.y)
        vins_trajectory_z.append(aligned_vins_pos.z)
    
    # Convert to numpy arrays
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
    
    return (exploration_times_filtered, exploration_rates_filtered,
            error_times, position_errors, orientation_errors,
            gt_trajectory_times, gt_trajectory_x, gt_trajectory_y, gt_trajectory_z,
            vins_trajectory_times, vins_trajectory_x, vins_trajectory_y, vins_trajectory_z)

def process_pointcloud_data(r2d2_pointcloud_data, ov_msckf_pointcloud_data, 
                           exploration_times, exploration_rates):
    """Process PointCloud data and create density heatmaps"""
    print("Processing PointCloud data for heatmap generation...")
    
    # Initialize arrays to store all u, v coordinates and intensities
    r2d2_u_coords = []
    r2d2_v_coords = []
    r2d2_intensities = []
    ov_msckf_u_coords = []
    ov_msckf_v_coords = []
    
    # Process R2D2 point cloud data
    for timestamp, msg in r2d2_pointcloud_data:
        try:
            # Check if this is PointCloud or PointCloud2
            if hasattr(msg, 'fields'):
                # PointCloud2 message
                u_field = None
                v_field = None
                intensity_field = None
                
                for field in msg.fields:
                    if field.name == 'u':
                        u_field = field
                    elif field.name == 'v':
                        v_field = field
                    elif field.name == 'intensity':
                        intensity_field = field
                
                if u_field is not None and v_field is not None:
                    # Calculate byte offsets
                    u_offset = u_field.offset
                    v_offset = v_field.offset
                    intensity_offset = intensity_field.offset if intensity_field else None
                    
                    # Extract data for each point
                    point_step = msg.point_step
                    data = np.frombuffer(msg.data, dtype=np.uint8)
                    
                    for i in range(msg.width * msg.height):
                        start_idx = i * point_step
                        
                        # Extract u, v coordinates
                        u = np.frombuffer(data[start_idx + u_offset:start_idx + u_offset + 4], dtype=np.float32)[0]
                        v = np.frombuffer(data[start_idx + v_offset:start_idx + v_offset + 4], dtype=np.float32)[0]
                        
                        # Extract intensity if available
                        intensity = 1.0  # Default weight
                        if intensity_field is not None:
                            intensity = np.frombuffer(data[start_idx + intensity_offset:start_idx + intensity_offset + 4], dtype=np.float32)[0]
                        
                        r2d2_u_coords.append(u)
                        r2d2_v_coords.append(v)
                        r2d2_intensities.append(intensity)
            else:
                # PointCloud message - check if it has channels
                if hasattr(msg, 'channels') and len(msg.channels) > 0:
                    # Find u, v, and intensity channels
                    u_channel = None
                    v_channel = None
                    intensity_channel = None
                    
                    for channel in msg.channels:
                        if channel.name == 'u':
                            u_channel = channel
                        elif channel.name == 'v':
                            v_channel = channel
                        elif channel.name == 'intensity':
                            intensity_channel = channel
                    
                    if u_channel is not None and v_channel is not None:
                        # Extract coordinates and intensity from channels
                        for i in range(len(msg.points)):
                            u = u_channel.values[i]
                            v = v_channel.values[i]
                            intensity = intensity_channel.values[i] if intensity_channel else 1.0
                            
                            r2d2_u_coords.append(u)
                            r2d2_v_coords.append(v)
                            r2d2_intensities.append(intensity)
                else:
                    print(f"Warning: R2D2 PointCloud message has no channels or u/v coordinates")
                    
        except Exception as e:
            print(f"Error processing R2D2 point cloud message: {e}")
            continue
    
    # Process OV-MSCKF loop features data
    for timestamp, msg in ov_msckf_pointcloud_data:
        try:
            # Check if this is PointCloud or PointCloud2
            if hasattr(msg, 'fields'):
                # PointCloud2 message
                u_field = None
                v_field = None
                
                for field in msg.fields:
                    if field.name == 'u':
                        u_field = field
                    elif field.name == 'v':
                        v_field = field
                
                if u_field is not None and v_field is not None:
                    # Calculate byte offsets
                    u_offset = u_field.offset
                    v_offset = v_field.offset
                    
                    # Extract data for each point
                    point_step = msg.point_step
                    data = np.frombuffer(msg.data, dtype=np.uint8)
                    
                    for i in range(msg.width * msg.height):
                        start_idx = i * point_step
                        
                        # Extract u, v coordinates
                        u = np.frombuffer(data[start_idx + u_offset:start_idx + u_offset + 4], dtype=np.float32)[0]
                        v = np.frombuffer(data[start_idx + v_offset:start_idx + v_offset + 4], dtype=np.float32)[0]
                        
                        ov_msckf_u_coords.append(u)
                        ov_msckf_v_coords.append(v)
            else:
                # PointCloud message - check if it has channels
                if hasattr(msg, 'channels') and len(msg.channels) > 0:
                    # For OV-MSCKF, each channel contains [x, y, z, u, v] values
                    # We need to extract u and v from the 3rd and 4th positions
                    for channel in msg.channels:
                        if len(channel.values) >= 5:  # Ensure we have at least 5 values
                            # Extract u and v coordinates (3rd and 4th values)
                            u = channel.values[2]  # 3rd value (index 2)
                            v = channel.values[3]  # 4th value (index 3)
                            
                            ov_msckf_u_coords.append(u)
                            ov_msckf_v_coords.append(v)
                        else:
                            print(f"Warning: Channel has insufficient values: {len(channel.values)}")
                else:
                    print(f"Warning: OV-MSCKF PointCloud message has no channels")
                    
        except Exception as e:
            print(f"Error processing OV-MSCKF point cloud message: {e}")
            continue
    
    print(f"Extracted {len(r2d2_u_coords)} R2D2 feature points")
    print(f"Extracted {len(ov_msckf_u_coords)} OV-MSCKF loop feature points")
    
    return (r2d2_u_coords, r2d2_v_coords, r2d2_intensities,
            ov_msckf_u_coords, ov_msckf_v_coords)

def create_heatmap(r2d2_u_coords, r2d2_v_coords, r2d2_intensities,
                   ov_msckf_u_coords, ov_msckf_v_coords, 
                   r2d2_width=1280, r2d2_height=640,
                   ov_width=1920, ov_height=1080,
                   total_time=None, bin_size=20, 
                   fixed_scale_r2d2=None, fixed_scale_ov=None):
    """Create 2D heatmap showing feature point density normalized by time
    
    Args:
        bin_size: Size of each bin in pixels (default: 20)
        fixed_scale_r2d2: Fixed scale range for R2D2 heatmap [min, max] (optional)
        fixed_scale_ov: Fixed scale range for OV-MSCKF heatmap [min, max] (optional)
    """
    
    # Create 2D histogram for R2D2 features (weighted by intensity)
    if len(r2d2_u_coords) > 0:
        r2d2_heatmap, r2d2_x_edges, r2d2_y_edges = np.histogram2d(
            r2d2_u_coords, r2d2_v_coords, 
            bins=[r2d2_width//bin_size, r2d2_height//bin_size],  # 可调整的分辨率
            range=[[0, r2d2_width], [0, r2d2_height]],
            weights=r2d2_intensities
        )
        
        # 如果提供了总时间，将热力图除以总时间进行归一化
        if total_time is not None and total_time > 0:
            r2d2_heatmap = r2d2_heatmap / total_time
    else:
        r2d2_heatmap = np.zeros((r2d2_height//bin_size, r2d2_width//bin_size))
        r2d2_x_edges = np.linspace(0, r2d2_width, r2d2_width//bin_size + 1)
        r2d2_y_edges = np.linspace(0, r2d2_height, r2d2_height//bin_size + 1)
    
    # Create 2D histogram for OV-MSCKF features (unweighted)
    if len(ov_msckf_u_coords) > 0:
        ov_msckf_heatmap, ov_x_edges, ov_y_edges = np.histogram2d(
            ov_msckf_u_coords, ov_msckf_v_coords,
            bins=[ov_width//bin_size, ov_height//bin_size],  # 可调整的分辨率
            range=[[0, ov_width], [0, ov_height]]
        )
        
        # 如果提供了总时间，将热力图除以总时间进行归一化
        if total_time is not None and total_time > 0:
            ov_msckf_heatmap = ov_msckf_heatmap / total_time
    else:
        ov_msckf_heatmap = np.zeros((ov_height//bin_size, ov_width//bin_size))
        ov_x_edges = np.linspace(0, ov_width, ov_width//bin_size + 1)
        ov_y_edges = np.linspace(0, ov_height, ov_height//bin_size + 1)
    
    # 使用 GridSpec: 顶部一行放两幅图，底部一行放滑动条
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f'Feature Point Density Heatmaps (Time Normalized, Bin Size: {bin_size}px)', fontsize=16, fontweight='bold')
    gs = GridSpec(2, 2, height_ratios=[1, 0.12], hspace=0.28, wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    
    # 计算数据统计信息用于颜色映射
    r2d2_nonzero = r2d2_heatmap[r2d2_heatmap > 0]
    ov_nonzero = ov_msckf_heatmap[ov_msckf_heatmap > 0]
    
    # 计算百分位数用于更好的颜色映射
    r2d2_p95 = np.percentile(r2d2_nonzero, 95) if len(r2d2_nonzero) > 0 else 1.0
    r2d2_p99 = np.percentile(r2d2_nonzero, 99) if len(r2d2_nonzero) > 0 else 1.0
    ov_p95 = np.percentile(ov_nonzero, 95) if len(ov_nonzero) > 0 else 1.0
    ov_p99 = np.percentile(ov_nonzero, 99) if len(ov_nonzero) > 0 else 1.0
    
    # 确定颜色映射范围 - 线性映射到 [0,1]
    if fixed_scale_r2d2 is not None:
        r2d2_vmin, r2d2_vmax = fixed_scale_r2d2
        r2d2_title_suffix = f" (Fixed Scale: {r2d2_vmin:.2f}-{r2d2_vmax:.2f})"
    else:
        r2d2_vmin, r2d2_vmax = 0.0, 1.0
        r2d2_title_suffix = " (Linear 0-1)"
    
    if fixed_scale_ov is not None:
        ov_vmin, ov_vmax = fixed_scale_ov
        ov_title_suffix = f" (Fixed Scale: {ov_vmin:.2f}-{ov_vmax:.2f})"
    else:
        ov_vmin, ov_vmax = 0.0, 1.0
        ov_title_suffix = " (Linear 0-1)"
    
    # 线性显示，直接裁剪到 [0,1]
    r2d2_display = np.clip(r2d2_heatmap, 0.0, 1.0)
    im1 = axes[0].imshow(r2d2_display.T, origin='lower', 
                        extent=[r2d2_x_edges[0], r2d2_x_edges[-1], r2d2_y_edges[0], r2d2_y_edges[-1]],
                        aspect='auto', cmap='viridis', vmin=r2d2_vmin, vmax=r2d2_vmax)
    axes[0].set_title(f'R2D2 Feature Density{r2d2_title_suffix}')
    axes[0].set_xlabel('Image U Coordinate (pixels)')
    axes[0].set_ylabel('Image V Coordinate (pixels)')
    plt.colorbar(im1, ax=axes[0], label='Weighted Density per Second' if (total_time is not None and total_time > 0) else 'Weighted Density')
 
    ov_display = np.clip(ov_msckf_heatmap, 0.0, 1.0)
    im2 = axes[1].imshow(ov_display.T, origin='lower',
                        extent=[ov_x_edges[0], ov_x_edges[-1], ov_y_edges[0], ov_y_edges[-1]],
                        aspect='auto', cmap='viridis', vmin=ov_vmin, vmax=ov_vmax)
    axes[1].set_title(f'OV-MSCKF Feature Density{ov_title_suffix}')
    axes[1].set_xlabel('Image U Coordinate (pixels)')
    axes[1].set_ylabel('Image V Coordinate (pixels)')
    plt.colorbar(im2, ax=axes[1], label='Feature Count per Second' if (total_time is not None and total_time > 0) else 'Feature Count')

    # 统计打印保持不变
    print("\n=== Feature Point Statistics ===")
    if len(r2d2_u_coords) > 0:
        print(f"R2D2 Features - Total: {len(r2d2_u_coords)}, "
              f"Mean Intensity: {np.mean(r2d2_intensities):.3f}")
        r2d2_nonzero_mean = np.mean(r2d2_nonzero) if len(r2d2_nonzero) > 0 else 0.0
        print(f"R2D2 Density - Mean: {r2d2_nonzero_mean:.3f}, "
              f"Max: {np.max(r2d2_heatmap):.3f}, "
              f"95th percentile: {r2d2_p95:.3f}, "
              f"99th percentile: {r2d2_p99:.3f}")
    else:
        print("R2D2 Features - No data available")

    if len(ov_msckf_u_coords) > 0:
        ov_nonzero_mean = np.mean(ov_nonzero) if len(ov_nonzero) > 0 else 0.0
        print(f"OV-MSCKF Features - Total: {len(ov_msckf_u_coords)}")
        print(f"OV-MSCKF Density - Mean: {ov_nonzero_mean:.3f}, "
              f"Max: {np.max(ov_msckf_heatmap):.3f}, "
              f"95th percentile: {ov_p95:.3f}, "
              f"99th percentile: {ov_p99:.3f}")
    else:
        print("OV-MSCKF Features - No data available")
    
    return fig

def interpolate_exploration_rate(exploration_times, exploration_rates, target_times):
    """Interpolate exploration rate to target timestamps"""
    if len(exploration_times) < 2:
        return np.full_like(target_times, np.nan)
    
    # Create interpolation function
    interp_func = interp1d(exploration_times, exploration_rates, 
                          kind='linear', bounds_error=False, fill_value=np.nan)
    
    return interp_func(target_times)

def create_plots(exploration_times, exploration_rates, error_times, 
                position_errors, orientation_errors, gt_trajectory_times,
                gt_trajectory_x, gt_trajectory_y, gt_trajectory_z,
                vins_trajectory_times, vins_trajectory_x, vins_trajectory_y, vins_trajectory_z):
    """Create the required plots including trajectory comparison"""
    
    # Check if we have valid error data
    if len(error_times) == 0:
        print("Warning: No error data available for plotting")
        return None
    
    # Convert times to relative times (start from 0)
    exp_time_rel = exploration_times - exploration_times[0]
    error_time_rel = error_times - exploration_times[0]
    gt_traj_time_rel = gt_trajectory_times - exploration_times[0]
    vins_traj_time_rel = vins_trajectory_times - exploration_times[0]
    
    # Interpolate exploration rates to error timestamps
    exploration_rates_interp = interpolate_exploration_rate(
        exploration_times, exploration_rates, error_times
    )
    
    # Create figure with subplots (3x3 layout to add velocity plot)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('ROS Bag Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Exploration rate vs time
    axes[0, 0].plot(exp_time_rel, exploration_rates, 'b-', linewidth=1.5)
    # 移除停止处理相关的垂直线
    # if error_time_rel[-1] < exp_time_rel[-1]:
    #     axes[0, 0].axvline(x=error_time_rel[-1], color='red', linestyle='--', 
    #                        label=f'Processing stopped at {error_time_rel[-1]:.1f}s')
    #     axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Exploration Rate')
    axes[0, 0].set_title('Exploration Rate vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Position error vs time
    axes[0, 1].plot(error_time_rel, position_errors, 'r-', linewidth=1.5, label='Position Error')
    # Add horizontal line at 2.0m threshold
    axes[0, 1].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2.0m threshold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position Error (m)')
    axes[0, 1].set_title('Position Error vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Orientation error vs time
    axes[0, 2].plot(error_time_rel, np.degrees(orientation_errors), 'g-', 
                   linewidth=1.5, label='Orientation Error')
    # 移除停止处理相关的垂直线
    # if error_time_rel[-1] < exp_time_rel[-1]:
    #     axes[0, 2].axvline(x=error_time_rel[-1], color='red', linestyle='--', 
    #                        label=f'Processing stopped at {error_time_rel[-1]:.1f}s')
    #     axes[0, 2].legend()
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Orientation Error (degrees)')
    axes[0, 2].set_title('Orientation Error vs Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Position error vs exploration rate
    valid_mask = ~np.isnan(exploration_rates_interp)
    if np.any(valid_mask):
        axes[1, 0].scatter(exploration_rates_interp[valid_mask], 
                          position_errors[valid_mask], 
                          alpha=0.6, s=20, c='purple')
        # Add horizontal line at 2.0m threshold
        axes[1, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2.0m threshold')
        axes[1, 0].set_xlabel('Exploration Rate')
        axes[1, 0].set_ylabel('Position Error (m)')
        axes[1, 0].set_title('Position Error vs Exploration Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No overlapping data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Position Error vs Exploration Rate')
    
    # Plot 5: X trajectory comparison
    axes[1, 1].plot(gt_traj_time_rel, gt_trajectory_x, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[1, 1].plot(vins_traj_time_rel, vins_trajectory_x, 'r--', linewidth=2, label='VINS (Aligned)', alpha=0.8)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('X Position (m)')
    axes[1, 1].set_title('X Trajectory Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Y trajectory comparison
    axes[1, 2].plot(gt_traj_time_rel, gt_trajectory_y, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[1, 2].plot(vins_traj_time_rel, vins_trajectory_y, 'r--', linewidth=2, label='VINS (Aligned)', alpha=0.8)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Y Position (m)')
    axes[1, 2].set_title('Y Trajectory Comparison')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # Plot 7: Flight velocity vs time (GT and VINS)
    def compute_speed(times, x, y, z):
        if len(times) < 2:
            return times[:0], np.array([])
        t = times.astype(np.float64)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        dt = np.diff(t)
        dt[dt == 0] = np.nan
        speed = np.sqrt(dx*dx + dy*dy + dz*dz) / dt
        t_mid = (t[1:] + t[:-1]) / 2.0
        return t_mid - exploration_times[0], speed
    
    gt_speed_t, gt_speed = compute_speed(gt_trajectory_times, gt_trajectory_x, gt_trajectory_y, gt_trajectory_z)
    
    axes[2, 0].plot(gt_speed_t, gt_speed, 'b-', linewidth=1.5, label='GT speed')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Speed (m/s)')
    axes[2, 0].set_title('GT Velocity vs Time')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Hide unused subplots (if any)
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    # 移除停止处理相关的注释
    # if error_time_rel[-1] < exp_time_rel[-1]:
    #     fig.text(0.5, 0.02, f'Note: Processing stopped at {error_time_rel[-1]:.1f}s due to position error > 2.0m', 
    #             ha='center', va='bottom', fontsize=10, style='italic', color='red')
    
    plt.tight_layout()
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Exploration Rate - Mean: {np.mean(exploration_rates):.4f}, "
          f"Std: {np.std(exploration_rates):.4f}")
    print(f"Position Error - Mean: {np.mean(position_errors):.4f}m, "
          f"Std: {np.std(position_errors):.4f}m, Max: {np.max(position_errors):.4f}m")
    print(f"Orientation Error - Mean: {np.degrees(np.mean(orientation_errors)):.2f}°, "
          f"Std: {np.degrees(np.std(orientation_errors)):.2f}°, "
          f"Max: {np.degrees(np.max(orientation_errors)):.2f}°")
    
    # 移除停止处理相关的打印信息
    # if error_time_rel[-1] < exp_time_rel[-1]:
    #     print(f"Data processing stopped at {error_time_rel[-1]:.1f}s due to large position error")
    
    return fig

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process ROS bag file and generate plots')
    parser.add_argument('bag_file', help='Path to the ROS bag file')
    parser.add_argument('--output', '-o', help='Output plot file (optional)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--image-width', type=int, default=1280, help='Image width for R2D2 heatmap (default: 1280)')
    parser.add_argument('--image-height', type=int, default=640, help='Image height for R2D2 heatmap (default: 640)')
    parser.add_argument('--bin-size', type=int, default=20, help='Bin size for heatmap (default: 20)')
    parser.add_argument('--r2d2-scale', nargs=2, type=float, metavar=('MIN', 'MAX'), 
                       default=[0.0, 0.2], help='Fixed scale range for R2D2 heatmap [min max] (default: 0.0 0.5)')
    parser.add_argument('--ov-scale', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       default=[0.0, 0.6], help='Fixed scale range for OV-MSCKF heatmap [min max] (default: 0.0 0.5)')
    
    args = parser.parse_args()
    
    # Process the bag file
    exploration_data, gt_odom_data, vins_odom_data, r2d2_pointcloud_data, ov_msckf_pointcloud_data = process_rosbag(args.bag_file)
    
    if not all([exploration_data, gt_odom_data, vins_odom_data]):
        print("Error: Failed to extract required data from bag file")
        return
    
    # Synchronize data and calculate errors
    result = synchronize_and_calculate_errors(exploration_data, gt_odom_data, vins_odom_data)
    
    if result[0] is None:
        print("Error: Failed to synchronize data and calculate errors")
        return
    
    (exploration_times, exploration_rates, error_times, position_errors, orientation_errors,
     gt_trajectory_times, gt_trajectory_x, gt_trajectory_y, gt_trajectory_z,
     vins_trajectory_times, vins_trajectory_x, vins_trajectory_y, vins_trajectory_z) = result
    
    # Process PointCloud data and create heatmaps
    pointcloud_result = process_pointcloud_data(r2d2_pointcloud_data, ov_msckf_pointcloud_data,
                                              exploration_times, exploration_rates)
    
    if pointcloud_result[0] is not None:  # Check if we have any R2D2 data
        r2d2_u_coords, r2d2_v_coords, r2d2_intensities, ov_msckf_u_coords, ov_msckf_v_coords = pointcloud_result
        
        # 计算总时间（从exploration_times获取）
        total_time = exploration_times[-1] - exploration_times[0] if len(exploration_times) > 1 else None
        print(f"Total bag time: {total_time:.2f} seconds")
        
        # Create heatmap with different image sizes for R2D2 and OV-MSCKF
        heatmap_fig = create_heatmap(r2d2_u_coords, r2d2_v_coords, r2d2_intensities,
                                    ov_msckf_u_coords, ov_msckf_v_coords,
                                    r2d2_width=args.image_width, r2d2_height=args.image_height,
                                    ov_width=args.image_width, ov_height=args.image_height,
                                    total_time=total_time, bin_size=args.bin_size,
                                    fixed_scale_r2d2=args.r2d2_scale, fixed_scale_ov=args.ov_scale)
        
        # 直接显示热力图
        plt.figure(heatmap_fig.number)
        plt.show()
    
    # Create main plots
    fig = create_plots(exploration_times, exploration_rates, error_times, 
                      position_errors, orientation_errors,
                      gt_trajectory_times, gt_trajectory_x, gt_trajectory_y, gt_trajectory_z,
                      vins_trajectory_times, vins_trajectory_x, vins_trajectory_y, vins_trajectory_z)
    
    if fig is None:
        print("Error: Failed to create plots")
        return
    
    # 直接显示所有图
    plt.show()

if __name__ == '__main__':
    main()
