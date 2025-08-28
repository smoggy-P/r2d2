#!/usr/bin/env python3
"""
ROS Bag Post-Processing Script

This script processes a ROS bag file containing:
- /exploration_rate (std_msgs/Float64 or similar float message)
- /kingfisher/ground_truth/odometry (nav_msgs/Odometry)
- /ov_msckf/odometry (nav_msgs/Odometry)
- /r2d2/point_cloud (sensor_msgs/PointCloud2)
- /ov_msckf/loop_feats (sensor_msgs/PointCloud2)

It generates four plots:
1. Exploration rate vs time
2. Odometry error vs time
3. Odometry error vs exploration rate
4. 2D heatmap of feature point density
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
        return None, None, None, None, None, None
    
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
        return None, None, None, None, None, None
    
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
    
    # Flag to track if we should stop processing due to large position error
    stop_processing = False
    stop_timestamp = None
    
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
        
        # Check if position error is too large (> 2.0m)
        if pos_error > 2.0:
            print(f"Position error {pos_error:.3f}m exceeds 2.0m at timestamp {vins_time:.2f}s. Stopping processing.")
            stop_processing = True
            stop_timestamp = vins_time
            break
        
        # For orientation, compare the aligned VINS with GT
        orient_error = calculate_orientation_error(
            gt_msg.pose.pose.orientation,
            r_vins_aligned.as_quat()
        )
        
        error_times.append(vins_time)
        position_errors.append(pos_error)
        orientation_errors.append(orient_error)
    
    # Convert to numpy arrays
    error_times = np.array(error_times)
    position_errors = np.array(position_errors)
    orientation_errors = np.array(orientation_errors)
    
    if stop_processing:
        print(f"Processing stopped at timestamp {stop_timestamp:.2f}s due to large position error")
        print(f"Final data range: {error_times[0]:.2f}s to {error_times[-1]:.2f}s")
    else:
        print(f"Calculated {len(error_times)} error measurements after alignment")
    
    return (exploration_times_filtered, exploration_rates_filtered,
            error_times, position_errors, orientation_errors)

def process_pointcloud_data(r2d2_pointcloud_data, ov_msckf_pointcloud_data, 
                           exploration_times, exploration_rates):
    """Process PointCloud2 data and create density heatmaps"""
    print("Processing PointCloud2 data for heatmap generation...")
    
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
                    # Find u and v channels
                    u_channel = None
                    v_channel = None
                    
                    for channel in msg.channels:
                        if channel.name == 'u':
                            u_channel = channel
                        elif channel.name == 'v':
                            v_channel = channel
                    
                    if u_channel is not None and v_channel is not None:
                        # Extract coordinates from channels
                        for i in range(len(msg.points)):
                            u = u_channel.values[i]
                            v = v_channel.values[i]
                            
                            ov_msckf_u_coords.append(u)
                            ov_msckf_v_coords.append(v)
                else:
                    print(f"Warning: OV-MSCKF PointCloud message has no channels or u/v coordinates")
                    
        except Exception as e:
            print(f"Error processing OV-MSCKF point cloud message: {e}")
            continue
    
    print(f"Extracted {len(r2d2_u_coords)} R2D2 feature points")
    print(f"Extracted {len(ov_msckf_u_coords)} OV-MSCKF loop feature points")
    
    return (r2d2_u_coords, r2d2_v_coords, r2d2_intensities,
            ov_msckf_u_coords, ov_msckf_v_coords)

def create_heatmap(r2d2_u_coords, r2d2_v_coords, r2d2_intensities,
                   ov_msckf_u_coords, ov_msckf_v_coords, image_width=1920, image_height=1080):
    """Create 2D heatmap showing feature point density"""
    
    # Create 2D histogram for R2D2 features (weighted by intensity)
    if len(r2d2_u_coords) > 0:
        r2d2_heatmap, x_edges, y_edges = np.histogram2d(
            r2d2_u_coords, r2d2_v_coords, 
            bins=[image_width//20, image_height//20],  # Adjust bin size as needed
            range=[[0, image_width], [0, image_height]],
            weights=r2d2_intensities
        )
    else:
        r2d2_heatmap = np.zeros((image_height//20, image_width//20))
        x_edges = np.linspace(0, image_width, image_width//20 + 1)
        y_edges = np.linspace(0, image_height, image_height//20 + 1)
    
    # Create 2D histogram for OV-MSCKF features (unweighted)
    if len(ov_msckf_u_coords) > 0:
        ov_msckf_heatmap, _, _ = np.histogram2d(
            ov_msckf_u_coords, ov_msckf_v_coords,
            bins=[image_width//20, image_height//20],
            range=[[0, image_width], [0, image_height]]
        )
    else:
        ov_msckf_heatmap = np.zeros((image_height//20, image_width//20))
    
    # Create the heatmap plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Feature Point Density Heatmaps', fontsize=16, fontweight='bold')
    
    # R2D2 heatmap (weighted by intensity)
    im1 = axes[0].imshow(r2d2_heatmap.T, origin='lower', 
                         extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                         aspect='auto', cmap='viridis')
    axes[0].set_title('R2D2 Feature Density (Intensity Weighted)')
    axes[0].set_xlabel('Image U Coordinate (pixels)')
    axes[0].set_ylabel('Image V Coordinate (pixels)')
    plt.colorbar(im1, ax=axes[0], label='Weighted Density')
    
    # OV-MSCKF heatmap
    im2 = axes[1].imshow(ov_msckf_heatmap.T, origin='lower',
                         extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                         aspect='auto', cmap='viridis')
    axes[1].set_title('OV-MSCKF Loop Feature Density')
    axes[1].set_xlabel('Image U Coordinate (pixels)')
    axes[1].set_ylabel('Image V Coordinate (pixels)')
    plt.colorbar(im2, ax=axes[1], label='Feature Count')
    
    plt.tight_layout()
    
    # Print statistics
    print("\n=== Feature Point Statistics ===")
    if len(r2d2_u_coords) > 0:
        print(f"R2D2 Features - Total: {len(r2d2_u_coords)}, "
              f"Mean Intensity: {np.mean(r2d2_intensities):.3f}, "
              f"Max Density: {np.max(r2d2_heatmap):.3f}")
    else:
        print("R2D2 Features - No data available")
    
    if len(ov_msckf_u_coords) > 0:
        print(f"OV-MSCKF Features - Total: {len(ov_msckf_u_coords)}, "
              f"Max Density: {np.max(ov_msckf_heatmap):.0f}")
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
                position_errors, orientation_errors):
    """Create the three required plots"""
    
    # Check if we have valid error data
    if len(error_times) == 0:
        print("Warning: No error data available for plotting")
        return None
    
    # Convert times to relative times (start from 0)
    exp_time_rel = exploration_times - exploration_times[0]
    error_time_rel = error_times - exploration_times[0]
    
    # Interpolate exploration rates to error timestamps
    exploration_rates_interp = interpolate_exploration_rate(
        exploration_times, exploration_rates, error_times
    )
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ROS Bag Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Exploration rate vs time
    axes[0, 0].plot(exp_time_rel, exploration_rates, 'b-', linewidth=1.5)
    # Add vertical line at the last error timestamp if different from exploration end
    if error_time_rel[-1] < exp_time_rel[-1]:
        axes[0, 0].axvline(x=error_time_rel[-1], color='red', linestyle='--', 
                           label=f'Processing stopped at {error_time_rel[-1]:.1f}s')
        axes[0, 0].legend()
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
    axes[1, 0].plot(error_time_rel, np.degrees(orientation_errors), 'g-', 
                   linewidth=1.5, label='Orientation Error')
    # Add vertical line at the last error timestamp
    if error_time_rel[-1] < exp_time_rel[-1]:
        axes[1, 0].axvline(x=error_time_rel[-1], color='red', linestyle='--', 
                           label=f'Processing stopped at {error_time_rel[-1]:.1f}s')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Orientation Error (degrees)')
    axes[1, 0].set_title('Orientation Error vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Position error vs exploration rate
    valid_mask = ~np.isnan(exploration_rates_interp)
    if np.any(valid_mask):
        axes[1, 1].scatter(exploration_rates_interp[valid_mask], 
                          position_errors[valid_mask], 
                          alpha=0.6, s=20, c='purple')
        # Add horizontal line at 2.0m threshold
        axes[1, 1].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2.0m threshold')
        axes[1, 1].set_xlabel('Exploration Rate')
        axes[1, 1].set_ylabel('Position Error (m)')
        axes[1, 1].set_title('Position Error vs Exploration Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No overlapping data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Position Error vs Exploration Rate')
    
    # Add overall information about data range
    if error_time_rel[-1] < exp_time_rel[-1]:
        fig.text(0.5, 0.02, f'Note: Processing stopped at {error_time_rel[-1]:.1f}s due to position error > 2.0m', 
                ha='center', va='bottom', fontsize=10, style='italic', color='red')
    
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
    
    if error_time_rel[-1] < exp_time_rel[-1]:
        print(f"Data processing stopped at {error_time_rel[-1]:.1f}s due to large position error")
    
    return fig

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process ROS bag file and generate plots')
    parser.add_argument('bag_file', help='Path to the ROS bag file')
    parser.add_argument('--output', '-o', help='Output plot file (optional)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--image-width', type=int, default=1280, help='Image width for heatmap (default: 1920)')
    parser.add_argument('--image-height', type=int, default=640, help='Image height for heatmap (default: 1080)')
    
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
    
    exploration_times, exploration_rates, error_times, position_errors, orientation_errors = result
    
    # Process PointCloud data and create heatmaps
    pointcloud_result = process_pointcloud_data(r2d2_pointcloud_data, ov_msckf_pointcloud_data,
                                              exploration_times, exploration_rates)
    
    if pointcloud_result[0] is not None:  # Check if we have any R2D2 data
        r2d2_u_coords, r2d2_v_coords, r2d2_intensities, ov_msckf_u_coords, ov_msckf_v_coords = pointcloud_result
        
        # Create heatmap
        heatmap_fig = create_heatmap(r2d2_u_coords, r2d2_v_coords, r2d2_intensities,
                                    ov_msckf_u_coords, ov_msckf_v_coords,
                                    args.image_width, args.image_height)
        
        # Save heatmap separately
        heatmap_output = args.bag_file.replace('.bag', '_heatmap.png')
        heatmap_fig.savefig(heatmap_output, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {heatmap_output}")
        
        if args.show:
            plt.figure(heatmap_fig.number)
            plt.show()
    
    # Create main plots
    fig = create_plots(exploration_times, exploration_rates, error_times, 
                      position_errors, orientation_errors)
    
    if fig is None:
        print("Error: Failed to create plots")
        return
    
    # Save or show plots
    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {args.output}")
    
    if args.show:
        plt.show()
    elif not args.output:
        # Default: save to current directory
        output_file = args.bag_file.replace('.bag', '_analysis.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_file}")

if __name__ == '__main__':
    main()