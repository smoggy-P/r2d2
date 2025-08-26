#!/usr/bin/env python3

# 首先导入ROS相关包
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge

# 然后导入conda环境中的包
import torch
import numpy as np
import cv2
from PIL import Image as PILImage
import time
import torch.nn.functional as F

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
from extract import load_network, NonMaxSuppression, extract_multiscale
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from threading import Lock
from scipy.spatial import cKDTree
from collections import deque

class R2D2GlobalFeatureNode:
    def __init__(self):
        rospy.init_node('r2d2_global_feature_node')
        
        # 获取ROS参数
        self.model_path = rospy.get_param('~model_path', 'models/r2d2_WASF_N16.pt')
        self.num_keypoints = rospy.get_param('~num_keypoints', 1000)
        self.reliability_thr = rospy.get_param('~reliability_thr', 0.2)
        self.repeatability_thr = rospy.get_param('~repeatability_thr', 0.2)
        self.map_range_x = rospy.get_param("~map_range_x", 100.0)
        self.map_range_y = rospy.get_param("~map_range_y", 100.0)
        self.map_range_z = rospy.get_param("~map_range_z", 50.0)
        self.voxel_size = rospy.get_param("~voxel_size", 0.1)
        self.merge_distance = rospy.get_param("~merge_distance", 0.2)
        self.publish_rate = rospy.get_param("~publish_rate", 1.0)
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.fx = rospy.get_param('~fx', 695.99511719)
        self.fy = rospy.get_param('~fy', 695.99511719)
        self.cx = rospy.get_param('~cx', 640)
        self.cy = rospy.get_param('~cy', 360)
        
        # 设置发布者和订阅者
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/D435i_camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/D435i_camera/depth/image_rect_raw", Image, self.depth_callback)
        self.odom_sub = rospy.Subscriber("/drone/odom", Odometry, self.odom_callback)
        self.vis_pub = rospy.Publisher('/r2d2/visualization', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('/r2d2/point_cloud', PointCloud2, queue_size=1)
        self.global_pub = rospy.Publisher("/r2d2/global_feature_map", PointCloud2, queue_size=1)
        
        # 加载网络
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device: ", self.device)
        self.net = load_network(self.model_path)
        self.net = self.net.to(self.device)
        self.net.eval()
        
        # 创建非极大值抑制检测器
        self.detector = NonMaxSuppression(
            rel_thr=self.reliability_thr,
            rep_thr=self.repeatability_thr
        )
        
        # 创建状态变量
        self.depth_img = None
        self.current_pose = np.eye(4)
        self.global_points = []  # List of [x, y, z, intensity]
        self.kdtree = None
        self.lock = Lock()
        self.odom_buffer = deque(maxlen=100)  # Store recent odometry messages
        
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_global_map)
        rospy.loginfo("R2D2 Global Feature Node initialized")

    def odom_callback(self, msg):
        with self.lock:
            self.odom_buffer.append((msg.header.stamp.to_sec(), msg))
            pose = msg.pose.pose
            q = pose.orientation
            t = pose.position
            # Quaternion to rotation matrix
            import tf
            rot = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            rot[0, 3] = t.x
            rot[1, 3] = t.y
            rot[2, 3] = t.z
            self.current_pose = rot

    def get_pose_at(self, stamp):
        # Find the odometry message closest to 'stamp'
        with self.lock:
            if not self.odom_buffer:
                return np.eye(4)
            times = [abs(t - stamp.to_sec()) for t, _ in self.odom_buffer]
            idx = np.argmin(times)
            msg = self.odom_buffer[idx][1]
            # Convert msg to 4x4 matrix as before
            pose = msg.pose.pose
            q = pose.orientation
            t = pose.position
            import tf
            rot = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            rot[0, 3] = t.x
            rot[1, 3] = t.y
            rot[2, 3] = t.z
            return rot

    def depth_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {str(e)}")

    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            pil_image = PILImage.fromarray(cv_image)
            
            # 准备输入
            img = norm_RGB(pil_image)[None]
            img = img.to(self.device)
            
            # 提取特征点
            print("start extract")
            start_time = time.time()
            xys, desc, scores = extract_multiscale(
                self.net, img, self.detector,
                scale_f=2**0.25,
                min_scale=0.0,
                max_scale=1.0,
                min_size=256,
                max_size=1024
            )
            print("extract done, consuming time: ", time.time() - start_time)
            # 转换为numpy数组
            xys = xys.cpu().numpy()
            scores = scores.cpu().numpy()
            
            # 选择前N个特征点
            idxs = scores.argsort()[-self.num_keypoints:]
            selected_kpts = xys[idxs]
            selected_scores = scores[idxs]
            
            # 可视化
            vis_img = cv_image.copy()
            
            # 将分数归一化到0-1
            norm_scores = (selected_scores - 0.9958) / (0.9992 - 0.9958)
            print("min score: ", norm_scores.min(), " max score: ", norm_scores.max())
            
            points = []
            if self.depth_img is not None:
                for kp, score in zip(selected_kpts, norm_scores):
                    x, y, scale = kp
                    # Project the keypoint to the depth image and publish point cloud
                    if x < self.depth_img.shape[1] and y < self.depth_img.shape[0]:
                        depth = self.depth_img[int(y), int(x)]
                        if depth > 0 and score > 0.9:
                            depth_meters = depth / 1000.0

                            x_world = depth_meters
                            y_world = -(x - self.cx) * depth_meters / self.fx
                            z_world = -(y - self.cy) * depth_meters / self.fy
                            points.append([x_world, y_world, z_world, score])

            # Publish local point cloud
            if len(points) > 0:
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1)
                ]
                header = msg.header
                header.frame_id = "D435i_camera_depth_frame"
                pc_msg = pc2.create_cloud(header, fields, points)
                self.pc_pub.publish(pc_msg)

                # Transform to world and merge into global map
                T = self.get_pose_at(msg.header.stamp)
                points_np = np.array(points)
                if points_np.shape[0] == 0:
                    return
                points_xyz = points_np[:, :3]
                points_xyz_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
                points_world = (T @ points_xyz_h.T).T[:, :3]
                points_intensity = points_np[:, 3:4]
                points_world = np.hstack([points_world, points_intensity])

                if len(self.global_points) == 0:
                    self.global_points = points_world.tolist()
                    self.kdtree = cKDTree(points_world[:, :3])
                else:
                    existing = np.array(self.global_points)
                    self.kdtree = cKDTree(existing[:, :3])
                    for pt in points_world:
                        dist, idx = self.kdtree.query(pt[:3], k=1)
                        if dist < self.merge_distance:
                            if pt[3] > existing[idx, 3]:
                                existing[idx, 3] = pt[3]
                        else:
                            existing = np.vstack([existing, pt])
                    self.global_points = existing.tolist()
                    self.kdtree = cKDTree(np.array(self.global_points)[:, :3])

            # Publish visualization
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "rgb8")
            self.vis_pub.publish(vis_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def publish_global_map(self, event):
        with self.lock:
            if not self.global_points:
                return
            points = np.array(self.global_points)
        
        # Filter ground points
        points = points[points[:, 2] > 0.1]
        
        # Expand feature points in 6 directions with 0.1 resolution
        expanded_points = []
        resolution = 0.1
        
        # Define 6 directions: +x, -x, +y, -y, +z, -z
        directions = np.array([
            # [resolution, 0, 0],    # +x
            # [-resolution, 0, 0],   # -x
            # [0, resolution, 0],    # +y
            # [0, -resolution, 0],   # -y
            # [0, 0, resolution],    # +z
            # [0, 0, -resolution]    # -z
        ])
        
        for point in points:
            x, y, z, intensity = point
            
            # Add the original point
            expanded_points.append([x, y, z, intensity])
            
            # Add expanded points in 6 directions
            for direction in directions:
                dx, dy, dz = direction
                expanded_points.append([x + dx, y + dy, z + dz, intensity])
        
        expanded_points = np.array(expanded_points)
        
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
        pc_msg = pc2.create_cloud(header, fields, expanded_points)
        self.global_pub.publish(pc_msg)

if __name__ == '__main__':
    try:
        node = R2D2GlobalFeatureNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
