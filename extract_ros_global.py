#!/usr/bin/env python3

"""
R2D2 Global Feature Extraction Node

This ROS node performs real-time feature extraction using R2D2 (Reliable and Repeatable
Detector and Descriptor) and builds a global feature map. It supports both depth-based
and octomap-based raycasting for 3D reconstruction.

Key Features:
- Synchronized RGB and depth image processing
- Global feature map building with point merging
- Octomap-based raycasting for 3D localization
- Visibility testing with occlusion checking
- Real-time visualization of features and visible rays
"""

# ROS imports
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from octomap_msgs.msg import Octomap
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as GeoPoint
from cv_bridge import CvBridge
import message_filters
import sensor_msgs.point_cloud2 as pc2

# Python standard library
import time
from threading import Lock
from collections import deque

# Third-party libraries
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image as PILImage
from scipy.spatial import cKDTree
import tf

# Local imports
from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
from extract import load_network, NonMaxSuppression, extract_multiscale


class R2D2GlobalFeatureNode:
    """
    ROS node for global feature map building using R2D2 features.
    
    This node processes synchronized RGB and depth images, extracts R2D2 features,
    and maintains a global 3D feature map. It supports both depth-based and
    octomap-based raycasting for 3D reconstruction.
    """
    
    # Camera to baselink transformation (x, y, z) in meters
    CAMERA_TO_BASELINK_TRANSLATION = np.array([0.05, 0.325, 0.1475])
    
    def __init__(self):
        """Initialize the R2D2 Global Feature Node with ROS parameters and subscribers."""
        rospy.init_node('r2d2_global_feature_node')
        
        # Load ROS parameters
        self._load_parameters()
        
        # Initialize state variables
        self._init_state()
        
        # Initialize R2D2 network before subscribers can invoke callbacks
        self._init_network()
        
        # Initialize ROS communication
        self._init_ros_communication()
        
        # Start periodic publishing
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_global_map)
        
        rospy.loginfo("R2D2 Global Feature Node initialized successfully")

    def _load_parameters(self):
        """Load all ROS parameters with default values."""
        # Model parameters
        self.model_path = rospy.get_param('~model_path', 'models/r2d2_WASF_N16.pt')
        self.num_keypoints = rospy.get_param('~num_keypoints', 1000)
        self.reliability_thr = rospy.get_param('~reliability_thr', 0.7)
        self.repeatability_thr = rospy.get_param('~repeatability_thr', 0.7)
        
        # Map parameters
        self.map_range_x = rospy.get_param("~map_range_x", 100.0)
        self.map_range_y = rospy.get_param("~map_range_y", 100.0)
        self.map_range_z = rospy.get_param("~map_range_z", 50.0)
        self.voxel_size = rospy.get_param("~voxel_size", 0.1)
        self.merge_distance = rospy.get_param("~merge_distance", 0.1)
        self.publish_rate = rospy.get_param("~publish_rate", 1.0)
        
        # Frame IDs
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.camera_frame = rospy.get_param("~camera_frame", "D435i_camera_color_frame")
        
        # Camera intrinsics
        self.fx = rospy.get_param('~fx', 695.99511719)
        self.fy = rospy.get_param('~fy', 695.99511719)
        self.cx = rospy.get_param('~cx', 640)
        self.cy = rospy.get_param('~cy', 360)
        self.img_width = rospy.get_param('~img_width', 1280)
        self.img_height = rospy.get_param('~img_height', 720)
        
        # Vision parameters
        self.max_view_distance = rospy.get_param('~max_view_distance', 10.0)
        self.score_threshold = rospy.get_param('~score_threshold', 0.8)
        # Octomap parameters
        self.octomap_resolution = rospy.get_param('~octomap_resolution', 0.1)
        self.ray_step_factor = rospy.get_param('~ray_step_factor', 0.8)
        self.occupied_check_distance = rospy.get_param('~occupied_check_distance', 0.15)
        self.enable_occupied_check = rospy.get_param('~enable_occupied_check', True)
        self.use_raycasting = rospy.get_param('~use_raycasting', True)
        
        # Visualization parameters
        self.line_width = rospy.get_param('~visible_line_width', 0.02)
        self.line_color = rospy.get_param('~visible_line_color', [0.1, 0.8, 1.0, 0.9])
        
        # Synchronization parameters
        self.sync_queue_size = rospy.get_param('~sync_queue_size', 10)
        self.sync_slop = rospy.get_param('~sync_slop', 0.1)

    def _init_ros_communication(self):
        """Initialize ROS publishers and subscribers."""
        self.bridge = CvBridge()
        
        # Synchronized RGB and depth image subscribers
        self.image_sub = message_filters.Subscriber('/D435i_camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber("/D435i_camera/depth/image_rect_raw", Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop
        )
        self.ts.registerCallback(self.synced_callback)
        
        # Other subscribers
        self.odom_sub = rospy.Subscriber("/drone/odom", Odometry, self.odom_callback)
        self.octomap_sub = rospy.Subscriber('/octomap_binary', Octomap, self.octomap_callback)
        self.octo_centers_sub = rospy.Subscriber('/octomap_point_cloud_centers', PointCloud2,
                                                 self.octomap_centers_callback, queue_size=1)
        
        # Publishers
        self.vis_pub = rospy.Publisher('/r2d2/visualization', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('/r2d2/point_cloud', PointCloud2, queue_size=1)
        self.global_pub = rospy.Publisher("/r2d2/global_feature_map", PointCloud2, queue_size=1)
        self.visible_pub = rospy.Publisher('/r2d2/visible_features_uv', PointCloud2, queue_size=1)
        self.rays_pub = rospy.Publisher('/r2d2/visible_rays', Marker, queue_size=1)

    def _init_network(self):
        """Initialize the R2D2 neural network."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        self.net = load_network(self.model_path)
        self.net = self.net.to(self.device)
        self.net.eval()
        
        self.detector = NonMaxSuppression(
            rel_thr=self.reliability_thr,
            rep_thr=self.repeatability_thr
        )

    def _init_state(self):
        """Initialize state variables."""
        self.current_pose = np.eye(4)
        self.global_points = []
        self.kdtree = None
        self.lock = Lock()
        self.odom_buffer = deque(maxlen=100)
        
        # Octomap data
        self.occ_points = None
        self.occ_kdtree = None

    # ==================== Odometry Handling ====================
    
    def odom_callback(self, msg):
        """
        Handle incoming odometry messages.
        
        Args:
            msg: Odometry message containing pose information
        """
        with self.lock:
            self.odom_buffer.append((msg.header.stamp.to_sec(), msg))
            
            # Update current pose
            pose = msg.pose.pose
            q = pose.orientation
            t = pose.position
            
            rot = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            rot[0, 3] = t.x
            rot[1, 3] = t.y
            rot[2, 3] = t.z
            self.current_pose = rot

    def get_pose_at(self, stamp):
        """
        Get camera pose at a specific timestamp.
        
        This method finds the closest odometry message to the given timestamp,
        extracts the baselink pose, and transforms it to camera frame.
        
        Args:
            stamp: ROS Time stamp
            
        Returns:
            4x4 transformation matrix representing camera pose in world frame
        """
        with self.lock:
            if not self.odom_buffer:
                return np.eye(4)
            
            # Find closest odometry message
            times = [abs(t - stamp.to_sec()) for t, _ in self.odom_buffer]
            idx = np.argmin(times)
            msg = self.odom_buffer[idx][1]
            
            # Extract baselink pose
            pose = msg.pose.pose
            q = pose.orientation
            t = pose.position
            
            T_world_baselink = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            T_world_baselink[0, 3] = t.x
            T_world_baselink[1, 3] = t.y
            T_world_baselink[2, 3] = t.z
            
            # Apply baselink to camera transformation
            T_baselink_camera = np.eye(4)
            T_baselink_camera[0:3, 3] = self.CAMERA_TO_BASELINK_TRANSLATION
            
            # Compute camera pose in world frame
            T_world_camera = T_world_baselink @ T_baselink_camera
            return T_world_camera

    # ==================== Octomap Handling ====================
    
    def octomap_callback(self, msg: Octomap):
        """
        Handle incoming octomap messages to extract resolution information.
        
        Args:
            msg: Octomap message
        """
        try:
            self.octomap_resolution = float(msg.resolution)
        except Exception as e:
            rospy.logwarn(f"Failed to parse octomap message: {e}")

    def octomap_centers_callback(self, msg: PointCloud2):
        """
        Handle incoming occupied voxel center point cloud for raycasting.
        
        Args:
            msg: PointCloud2 message containing occupied voxel centers
        """
        try:
            pts = []
            for p in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                pts.append([p[0], p[1], p[2]])
            
            if len(pts) > 0:
                self.occ_points = np.asarray(pts, dtype=np.float32)
                self.occ_kdtree = cKDTree(self.occ_points)
        except Exception as e:
            rospy.logwarn(f"Failed to parse octomap centers point cloud: {e}")

    # ==================== Raycasting and Visibility ====================
    
    def raycast_to_octomap(self, u, v, T_wc):
        """
        Perform raycasting from a pixel to find intersection with occupied voxels.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            T_wc: 4x4 camera pose in world frame
            
        Returns:
            np.array([x, y, z]) of the hit point in world frame, or None if no hit
        """
        if self.occ_kdtree is None or self.occ_points is None:
            rospy.logwarn_throttle(10.0, "Octomap data not available for raycasting")
            return None
        
        # Get camera origin in world frame
        cam_origin = T_wc[0:3, 3]
        
        # Compute ray direction in camera frame
        x_cam = 1.0
        y_cam = -(u - self.cx) * x_cam / self.fx
        z_cam = -(v - self.cy) * x_cam / self.fy
        
        ray_dir_cam = np.array([x_cam, y_cam, z_cam, 0.0])
        
        # Transform ray direction to world frame
        ray_dir_world = (T_wc @ ray_dir_cam)[0:3]
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)
        
        # Sample along ray to find first occupied voxel
        step = max(0.05, self.octomap_resolution * self.ray_step_factor)
        max_dist = self.max_view_distance
        n_steps = int(max_dist / step)
        
        for i in range(1, n_steps):
            current_point = cam_origin + ray_dir_world * (i * step)
            
            try:
                dist, idx = self.occ_kdtree.query(current_point, k=1)
                
                # Check if we hit an occupied voxel
                if dist < (self.octomap_resolution * 0.5):
                    return self.occ_points[idx]
            except Exception as e:
                rospy.logwarn_throttle(10.0, f"Raycast query failed: {e}")
                return None
        
        return None

    def is_point_near_occupied(self, point_world):
        """
        Check if a world point is near occupied voxels.
        
        Args:
            point_world: np.array([x, y, z]) point in world frame
            
        Returns:
            bool: True if point is near occupied voxels (should be added),
                  False if surrounded by free space (should be filtered)
        """
        if not self.enable_occupied_check:
            return True
        
        if self.occ_kdtree is None:
            return True
        
        try:
            dist, _ = self.occ_kdtree.query(point_world, k=1)
            return dist < self.occupied_check_distance
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"Occupied check failed: {e}")
            return True

    def is_visible_world(self, pw: np.ndarray, T_wc: np.ndarray):
        """
        Test if a world point is visible from the given camera pose.
        
        This function checks:
        1. If the point is in front of the camera
        2. If the point projects within the image bounds
        3. If the point is within maximum viewing distance
        4. If the point is not occluded by other objects
        
        Args:
            pw: World point [x, y, z]
            T_wc: 4x4 camera pose in world frame
            
        Returns:
            tuple: (is_visible, u, v) where u, v are pixel coordinates
        """
        # Transform point to camera frame
        T_cw = np.linalg.inv(T_wc)
        pw_h = np.array([pw[0], pw[1], pw[2], 1.0], dtype=np.float64)
        pc = (T_cw @ pw_h)[0:3]

        # Check if point is in front of camera
        if pc[0] <= 0.0:
            return False, 0.0, 0.0

        # Project to image plane
        u = -self.fx * (pc[1] / pc[0]) + self.cx
        v = -self.fy * (pc[2] / pc[0]) + self.cy

        # Check if within image bounds
        if (u < 0) or (u >= self.img_width) or (v < 0) or (v >= self.img_height):
            return False, 0.0, 0.0

        # Check distance
        cam_o = T_wc[0:3, 3]
        dir_w = pw - cam_o
        dist = np.linalg.norm(dir_w)
        
        if dist > self.max_view_distance:
            return False, 0.0, 0.0
        
        # Occlusion check using raycasting
        if self.occ_kdtree is not None and self.octomap_resolution > 0.0:
            if dist < 1e-6:
                return True, u, v
            
            dir_w = dir_w / dist
            step = max(0.05, self.octomap_resolution * self.ray_step_factor)
            n_steps = int(dist / step)
            
            # Sample along ray from camera to point
            for i in range(1, max(1, n_steps)):
                s = cam_o + dir_w * (i * step)
                d, _ = self.occ_kdtree.query(s, k=1)
                
                if d < (self.octomap_resolution * 0.5):
                    return False, 0.0, 0.0

        return True, float(u), float(v)

    # ==================== Image Processing ====================
    
    def synced_callback(self, image_msg, depth_msg):
        """
        Handle synchronized RGB and depth images.
        
        Args:
            image_msg: RGB image message
            depth_msg: Depth image message
        """
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            self.process_images(image_msg, depth_img)
        except Exception as e:
            rospy.logerr(f"Error in synced callback: {str(e)}")

    def process_images(self, msg, depth_img):
        """
        Process RGB and depth images to extract and map features.
        
        Args:
            msg: RGB image message
            depth_img: Depth image as numpy array
        """
        try:
            # Convert to format needed by R2D2
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            pil_image = PILImage.fromarray(cv_image)
            
            img = norm_RGB(pil_image)[None]
            img = img.to(self.device)
            
            # Extract R2D2 features
            rospy.loginfo("Starting feature extraction")
            start_time = time.time()
            
            xys, desc, scores = extract_multiscale(
                self.net, img, self.detector,
                scale_f=2**0.25,
                min_scale=0.0,
                max_scale=1.0,
                min_size=256,
                max_size=1024
            )
            
            extraction_time = time.time() - start_time
            rospy.loginfo(f"Feature extraction completed in {extraction_time:.3f}s")
            
            # Select top keypoints
            xys = xys.cpu().numpy()
            scores = scores.cpu().numpy()
            
            idxs = scores.argsort()[-self.num_keypoints:]
            selected_kpts = xys[idxs]
            selected_scores = scores[idxs]
            
            # Normalize scores for visualization
            vis_img = cv_image.copy()
            norm_scores = (selected_scores - 0.9958) / (0.9992 - 0.9958)
            
            # Reconstruct 3D positions
            points = self._reconstruct_3d_points(
                selected_kpts, norm_scores, depth_img, msg.header.stamp
            )
            
            # Publish local point cloud
            if len(points) > 0:
                self._publish_local_pointcloud(points, msg.header)
                
                # Update global map
                added_indices = self._update_global_map(points, msg.header.stamp)
                
                # Visualize added features
                self._visualize_features(vis_img, points, added_indices)
            
            # Publish visualization
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "rgb8")
            self.vis_pub.publish(vis_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def _reconstruct_3d_points(self, keypoints, scores, depth_img, stamp):
        """
        Reconstruct 3D points from keypoints using depth or raycasting.
        
        Args:
            keypoints: Nx3 array of (x, y, scale)
            scores: N array of feature scores
            depth_img: Depth image
            stamp: Image timestamp
            
        Returns:
            List of [x, y, z, score, u, v] in camera frame
        """
        points = []
        
        if self.use_raycasting:
            rospy.loginfo_throttle(10.0, "Using octomap raycasting for 3D reconstruction")
            
            T_wc = self.get_pose_at(stamp)
            raycast_success = 0
            raycast_total = 0
            
            for kp, score in zip(keypoints, scores):
                x, y, scale = kp
                
                if score < self.score_threshold:
                    continue
                
                if x < 0 or x >= self.img_width or y < 0 or y >= self.img_height:
                    continue
                
                raycast_total += 1
                point_world = self.raycast_to_octomap(x, y, T_wc)
                
                if point_world is not None:
                    raycast_success += 1
                    
                    # Transform to camera frame for local pointcloud
                    T_cw = np.linalg.inv(T_wc)
                    pw_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
                    pc = (T_cw @ pw_h)[0:3]
                    
                    points.append([pc[0], pc[1], pc[2], score, x, y])
            
            if raycast_total > 0:
                success_rate = 100 * raycast_success / raycast_total
                rospy.loginfo_throttle(2.0, 
                    f"Raycasting: {raycast_success}/{raycast_total} successful ({success_rate:.1f}%)")
        else:
            rospy.loginfo_throttle(10.0, "Using depth image for 3D reconstruction")
            
            if depth_img is not None:
                for kp, score in zip(keypoints, scores):
                    x, y, scale = kp
                    
                    if score < self.score_threshold:
                        continue
                    
                    if x < depth_img.shape[1] and y < depth_img.shape[0]:
                        depth = depth_img[int(y), int(x)]
                        
                        if 0 <= depth <= 5000:
                            depth_meters = depth / 1000.0
                            
                            x_cam = depth_meters
                            y_cam = -(x - self.cx) * depth_meters / self.fx
                            z_cam = -(y - self.cy) * depth_meters / self.fy
                            
                            points.append([x_cam, y_cam, z_cam, score, x, y])
            else:
                rospy.logwarn_throttle(5.0, "Depth image mode enabled but no depth available")
        
        return points

    def _publish_local_pointcloud(self, points, header):
        """
        Publish local point cloud in camera frame.
        
        Args:
            points: List of [x, y, z, intensity, u, v]
            header: ROS header for the point cloud
        """
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('u', 16, PointField.FLOAT32, 1),
            PointField('v', 20, PointField.FLOAT32, 1)
        ]
        
        header.frame_id = self.camera_frame
        pc_msg = pc2.create_cloud(header, fields, points)
        self.pc_pub.publish(pc_msg)

    def _update_global_map(self, points, stamp):
        """
        Update global feature map with new points.
        
        Args:
            points: List of [x, y, z, intensity, u, v] in camera frame
            stamp: Timestamp for pose lookup
            
        Returns:
            List of indices of points that were added/updated
        """
        T = self.get_pose_at(stamp)
        points_np = np.array(points)
        
        if points_np.shape[0] == 0:
            return []
        
        # Transform to world frame
        points_xyz = points_np[:, :3]
        points_xyz_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
        points_world = (T @ points_xyz_h.T).T[:, :3]
        points_intensity = points_np[:, 3:4]
        points_world = np.hstack([points_world, points_intensity])

        added_indices = []
        
        if len(self.global_points) == 0:
            # First batch of points
            filtered_points = []
            for i, pt in enumerate(points_world):
                if self.is_point_near_occupied(pt[:3]):
                    filtered_points.append(pt)
                    added_indices.append(i)
            
            if len(filtered_points) > 0:
                self.global_points = filtered_points
                self.kdtree = cKDTree(np.array(filtered_points)[:, :3])
        else:
            # Merge with existing points
            existing = np.array(self.global_points)
            self.kdtree = cKDTree(existing[:, :3])
            
            for i, pt in enumerate(points_world):
                # Check if point is near occupied voxels
                if not self.is_point_near_occupied(pt[:3]):
                    continue
                
                # Check if point should be merged with existing
                dist, idx = self.kdtree.query(pt[:3], k=1)
                
                if dist < self.merge_distance:
                    # Update existing point if new intensity is higher
                    if pt[3] > existing[idx, 3]:
                        existing[idx, 3] = pt[3]
                        added_indices.append(i)
                else:
                    # Add as new point
                    existing = np.vstack([existing, pt])
                    added_indices.append(i)
            
            self.global_points = existing.tolist()
            self.kdtree = cKDTree(np.array(self.global_points)[:, :3])
        
        return added_indices

    def _visualize_features(self, vis_img, points, added_indices):
        """
        Draw feature points on visualization image.
        
        Args:
            vis_img: Image to draw on
            points: List of [x, y, z, intensity, u, v]
            added_indices: Indices of points that were added to global map
        """
        points_np = np.array(points)
        
        for idx in added_indices:
            u, v = int(points_np[idx, 4]), int(points_np[idx, 5])
            intensity = points_np[idx, 3]
            
            # Normalize intensity for color
            normalized_intensity = np.clip((intensity - 0.8) / 0.2, 0.0, 1.0)
            color_intensity = int(normalized_intensity * 255)
            color = (0, 255, color_intensity)
            
            cv2.circle(vis_img, (u, v), 3, color, -1)
            cv2.circle(vis_img, (u, v), 5, (0, 255, 0), 1)

    # ==================== Publishing ====================
    
    def publish_global_map(self, event):
        """
        Publish global feature map and visible features periodically.
        
        Args:
            event: ROS timer event
        """
        with self.lock:
            if not self.global_points:
                return
            points = np.array(self.global_points)
        
        # Filter ground points
        points = points[points[:, 2] > 0.2]
        
        # Expand feature points (optional, currently disabled)
        expanded_points = self._expand_feature_points(points)
        
        # Publish global map
        self._publish_global_pointcloud(expanded_points)
        
        # Publish visible features
        self._publish_visible_features(points)

    def _expand_feature_points(self, points):
        """
        Expand feature points in multiple directions (currently disabled).
        
        Args:
            points: Nx4 array of [x, y, z, intensity]
            
        Returns:
            Expanded points array
        """
        expanded_points = []
        resolution = 0.1
        
        # Define expansion directions (currently none)
        directions = np.array([])
        
        for point in points:
            x, y, z, intensity = point
            expanded_points.append([x, y, z, intensity])
            
            for direction in directions:
                dx, dy, dz = direction
                expanded_points.append([x + dx, y + dy, z + dz, intensity])
        
        return np.array(expanded_points)

    def _publish_global_pointcloud(self, points):
        """
        Publish global feature map as point cloud.
        
        Args:
            points: Nx4 array of [x, y, z, intensity]
        """
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
        
        pc_msg = pc2.create_cloud(header, fields, points)
        self.global_pub.publish(pc_msg)

    def _publish_visible_features(self, points):
        """
        Publish features visible from current camera pose.
        
        Args:
            points: Nx4 array of [x, y, z, intensity] in world frame
        """
        try:
            T_wc = self.current_pose.copy()
            uv_intensity = []
            visible_world_points = []
            
            # Limit number of points to check for performance
            max_check = rospy.get_param('~max_visible_check', 5000)
            
            if points.shape[0] > max_check:
                indices = np.random.choice(points.shape[0], max_check, replace=False)
                pts_for_check = points[indices]
            else:
                pts_for_check = points

            # Check visibility for each point
            for p in pts_for_check:
                pw = p[:3]
                inten = float(p[3])
                ok, u, v = self.is_visible_world(pw, T_wc)
                
                if ok:
                    uv_intensity.append([u, v, inten])
                    visible_world_points.append((pw, inten))

            if len(uv_intensity) > 0:
                # Publish UV coordinates
                self._publish_uv_pointcloud(uv_intensity)
                
                # Publish visibility rays for RViz
                self._publish_visibility_rays(visible_world_points, T_wc)

        except Exception as e:
            rospy.logwarn(f"Failed to publish visible features: {e}")

    def _publish_uv_pointcloud(self, uv_intensity):
        """
        Publish UV coordinates of visible features.
        
        Args:
            uv_intensity: List of [u, v, intensity]
        """
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_frame

        fields = [
            PointField('u', 0, PointField.FLOAT32, 1),
            PointField('v', 4, PointField.FLOAT32, 1),
            PointField('intensity', 8, PointField.FLOAT32, 1)
        ]
        
        uv_msg = pc2.create_cloud(header, fields, uv_intensity)
        self.visible_pub.publish(uv_msg)

    def _publish_visibility_rays(self, visible_points, T_wc):
        """
        Publish line markers showing rays from camera to visible features.
        
        Args:
            visible_points: List of (world_point, intensity) tuples
            T_wc: 4x4 camera pose in world frame
        """
        try:
            if len(visible_points) == 0:
                return
            
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "r2d2_visible_rays"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = float(self.line_width)
            
            # Set color
            marker.color.r = float(self.line_color[0])
            marker.color.g = float(self.line_color[1])
            marker.color.b = float(self.line_color[2])
            marker.color.a = float(self.line_color[3])
            
            # Build line list: camera origin -> visible point
            cam_o = T_wc[0:3, 3]
            cam_pt = GeoPoint(x=float(cam_o[0]), y=float(cam_o[1]), z=float(cam_o[2]))
            
            marker.points = []
            max_lines = rospy.get_param('~max_visible_lines', 
                                       rospy.get_param('~max_visible_check', 5000))
            
            for i, (pw, inten) in enumerate(visible_points):
                if i >= max_lines:
                    break
                
                world_pt = GeoPoint(x=float(pw[0]), y=float(pw[1]), z=float(pw[2]))
                marker.points.append(cam_pt)
                marker.points.append(world_pt)
            
            # Set lifetime based on publish rate
            marker.lifetime = rospy.Duration(1.0 / max(1e-3, self.publish_rate))
            self.rays_pub.publish(marker)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish visible rays marker: {e}")


def main():
    """Main entry point for the ROS node."""
    try:
        node = R2D2GlobalFeatureNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
