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
# 可视化
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as GeoPoint


# === NEW: OctoMap 消息
from octomap_msgs.msg import Octomap  # /octomap_binary
# 可选：订阅占据体素中心点云（若 octomap_server 开启 publish_point_cloud_centers）
# 话题名通常是 /octomap_point_cloud_centers

class R2D2GlobalFeatureNode:
    def __init__(self):
        rospy.init_node('r2d2_global_feature_node')
        
        # 获取ROS参数
        self.model_path = rospy.get_param('~model_path', 'models/r2d2_WASF_N16.pt')
        self.num_keypoints = rospy.get_param('~num_keypoints', 1000)
        self.reliability_thr = rospy.get_param('~reliability_thr', 0.9)
        self.repeatability_thr = rospy.get_param('~repeatability_thr', 0.7)
        self.map_range_x = rospy.get_param("~map_range_x", 100.0)
        self.map_range_y = rospy.get_param("~map_range_y", 100.0)
        self.map_range_z = rospy.get_param("~map_range_z", 50.0)
        self.voxel_size = rospy.get_param("~voxel_size", 0.1)
        self.merge_distance = rospy.get_param("~merge_distance", 0.2)
        self.publish_rate = rospy.get_param("~publish_rate", 1.0)
        self.world_frame = rospy.get_param("~world_frame", "world")
        # 线宽与颜色（可通过 ROS 参数覆盖）
        self.line_width = rospy.get_param('~visible_line_width', 0.02)
        self.line_color = rospy.get_param('~visible_line_color', [0.1, 0.8, 1.0, 0.9])  # RGBA
        self.rays_pub = rospy.Publisher('/r2d2/visible_rays', Marker, queue_size=1)


        # === NEW: 用你的相机内参作为默认值（可被参数覆盖）
        self.fx = rospy.get_param('~fx', 695.99511719)
        self.fy = rospy.get_param('~fy', 695.99511719)
        self.cx = rospy.get_param('~cx', 640)
        self.cy = rospy.get_param('~cy', 360)
        self.img_width  = rospy.get_param('~img_width', 1280)
        self.img_height = rospy.get_param('~img_height', 720)
        self.camera_frame = rospy.get_param('~camera_frame', 'D435i_camera_depth_frame')

        # 设置发布者和订阅者
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/D435i_camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/D435i_camera/depth/image_rect_raw", Image, self.depth_callback)
        self.odom_sub = rospy.Subscriber("/drone/odom", Odometry, self.odom_callback)
        self.vis_pub = rospy.Publisher('/r2d2/visualization', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('/r2d2/point_cloud', PointCloud2, queue_size=1)
        self.global_pub = rospy.Publisher("/r2d2/global_feature_map", PointCloud2, queue_size=1)

        # === NEW: 发布当前可见的 (u,v,intensity)
        self.visible_pub = rospy.Publisher('/r2d2/visible_features_uv', PointCloud2, queue_size=1)

        # === NEW: 订阅 OctoMap（用于获取分辨率等；可选再订阅体素中心点云以做 raycasting）
        self.octomap_resolution = rospy.get_param('~octomap_resolution', 0.2)  # 若消息未到，先用默认
        self.ray_step_factor = rospy.get_param('~ray_step_factor', 0.8)        # 步长 = res * factor
        self.octomap_sub = rospy.Subscriber('/octomap_binary', Octomap, self.octomap_callback)
        # 可选占据中心点云，若开启了 octomap_server 的 publish_point_cloud_centers:
        self.octo_centers_sub = rospy.Subscriber('/octomap_point_cloud_centers', PointCloud2,
                                                 self.octomap_centers_callback, queue_size=1)

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

        # === NEW: OctoMap 占据点（体素中心）KDTree（若收到则用于可见性测试）
        self.occ_points = None
        self.occ_kdtree = None
        
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

    # === NEW: 接收 OctoMap（仅用来获取分辨率等元信息；具体占据点靠 centers 点云）
    def octomap_callback(self, msg: Octomap):
        try:
            # 记录分辨率；若你需要更严格的 raycasting，可在 C++ 端做服务调用
            self.octomap_resolution = float(msg.resolution)
        except Exception as e:
            rospy.logwarn(f"Octomap msg parse failed: {e}")

    # === NEW: 接收体素中心点云并建立 KDTree，用于近似 raycasting
    def octomap_centers_callback(self, msg: PointCloud2):
        try:
            pts = []
            for p in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
                pts.append([p[0], p[1], p[2]])
            if len(pts) > 0:
                self.occ_points = np.asarray(pts, dtype=np.float32)
                self.occ_kdtree = cKDTree(self.occ_points)
        except Exception as e:
            rospy.logwarn(f"Failed to parse octomap centers point cloud: {e}")

    # === NEW: 世界点对当前相机位姿的可见性测试（含 FOV 与遮挡；遮挡需 occ_kdtree 可用）
    def is_visible_world(self, pw: np.ndarray, T_wc: np.ndarray) -> (bool, float, float):
        """
        输入: 世界点 pw[3], 相机位姿 T_wc (4x4), 内参 (fx, fy, cx, cy), 图像尺寸
        输出: (是否可见, u, v). 若不可见，u,v 无意义
        """
        # 相机->世界的逆用于把世界点变到相机坐标
        T_cw = np.linalg.inv(T_wc)
        pw_h = np.array([pw[0], pw[1], pw[2], 1.0], dtype=np.float64)
        pc = (T_cw @ pw_h)[0:3]

        # 前方可见
        if pc[0] <= 0.0:
            return False, 0.0, 0.0

        # 像素坐标
        u = -self.fx * (pc[1] / pc[0]) + self.cx
        v = -self.fy * (pc[2] / pc[0]) + self.cy

        # 视野范围
        if (u < 0) or (u >= self.img_width) or (v < 0) or (v >= self.img_height):
            return False, 0.0, 0.0

        # 遮挡检测（可选：仅当 KDTree 可用）
        if self.occ_kdtree is not None and self.octomap_resolution > 0.0:
            cam_o = T_wc[0:3, 3]              # 相机在世界中的位置
            dir_w = pw - cam_o
            dist = np.linalg.norm(dir_w)
            if dist < 1e-6:
                return True, u, v
            dir_w = dir_w / dist

            step = max(0.05, self.octomap_resolution * self.ray_step_factor)
            n_steps = int(dist / step)
            # 从靠近相机的一点开始，避免直接命中特征点本身
            for i in range(1, max(1, n_steps)):
                s = cam_o + dir_w * (i * step)
                # 查询最近占据体素中心距离
                d, _ = self.occ_kdtree.query(s, k=1)
                if d < (self.octomap_resolution * 0.5):
                    return False, 0.0, 0.0  # 被遮挡

        return True, float(u), float(v)

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
            print("selected_scores range: ", selected_scores.min(), selected_scores.max())
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
                            points.append([x_world, y_world, z_world, score, x, y])

            # Publish local point cloud
            if len(points) > 0:
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1),
                    PointField('u', 16, PointField.FLOAT32, 1),
                    PointField('v', 20, PointField.FLOAT32, 1)
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

        # === NEW: 同步发布“当前相机视角下可见的 (u, v, intensity)”
        try:
            T_wc = self.current_pose.copy()
            uv_intensity = []
            visible_world_points = []
            # 为了性能，可限制最多检查的点数（可通过参数控制，这里给个保护）
            max_check = rospy.get_param('~max_visible_check', 5000)
            pts_for_check = points if points.shape[0] <= max_check else points[np.random.choice(points.shape[0], max_check, replace=False)]

            for p in pts_for_check:
                pw = p[:3]
                inten = float(p[3])
                ok, u, v = self.is_visible_world(pw, T_wc)
                if ok:
                    uv_intensity.append([u, v, inten])
                    visible_world_points.append((pw, inten))

            if len(uv_intensity) > 0:
                uv_intensity = np.asarray(uv_intensity, dtype=np.float32)
                header_uv = rospy.Header()
                header_uv.stamp = rospy.Time.now()
                header_uv.frame_id = self.camera_frame

                uv_fields = [
                    PointField('u', 0, PointField.FLOAT32, 1),
                    PointField('v', 4, PointField.FLOAT32, 1),
                    PointField('intensity', 8, PointField.FLOAT32, 1)
                ]
                # pc2.create_cloud 支持自定义字段；传 Nx3 数组即可
                uv_msg = pc2.create_cloud(header_uv, uv_fields, uv_intensity.tolist())
                self.visible_pub.publish(uv_msg)

                # === NEW: 发布“当前位置到可见点”的连线（RViz）
                try:
                    if len(visible_world_points) > 0:
                        marker = Marker()
                        marker.header.frame_id = self.world_frame
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "r2d2_visible_rays"
                        marker.id = 0
                        marker.type = Marker.LINE_LIST
                        marker.action = Marker.ADD
                        marker.scale.x = float(self.line_width)  # 线宽（米）
                        # 统一颜色
                        marker.color.r = float(self.line_color[0])
                        marker.color.g = float(self.line_color[1])
                        marker.color.b = float(self.line_color[2])
                        marker.color.a = float(self.line_color[3])
                        # 建立点对：相机原点 -> 可见世界点
                        cam_o = T_wc[0:3, 3]
                        cam_pt = GeoPoint(x=float(cam_o[0]), y=float(cam_o[1]), z=float(cam_o[2]))
                        marker.points = []
                        # （可选）限制最多画多少条线，避免太密；默认沿用 max_visible_check
                        max_lines = rospy.get_param('~max_visible_lines', rospy.get_param('~max_visible_check', 5000))
                        count = 0
                        for (pw, inten) in visible_world_points:
                            if count >= max_lines:
                                break
                            world_pt = GeoPoint(x=float(pw[0]), y=float(pw[1]), z=float(pw[2]))
                            marker.points.append(cam_pt)
                            marker.points.append(world_pt)
                            count += 1
                        # 让它随定时器刷新，过期自动清除
                        marker.lifetime = rospy.Duration(1.0 / max(1e-3, self.publish_rate))
                        self.rays_pub.publish(marker)
                except Exception as e:
                    rospy.logwarn(f"Failed to publish visible rays marker: {e}")

        except Exception as e:
            rospy.logwarn(f"Failed to publish visible (u,v,intensity): {e}")

if __name__ == '__main__':
    try:
        node = R2D2GlobalFeatureNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
