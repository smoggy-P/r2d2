#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge

import torch
import numpy as np

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
from visualization_msgs.msg import Marker
from octomap_msgs.msg import Octomap
import tf
from geometry_msgs.msg import PoseStamped


class R2D2GlobalFeatureNode:

    # Camera to baselink translation (x, y, z) in meters
    CAMERA_TO_BASELINK_TRANSLATION = np.array([0.05, 0.325, 0.1475])

    def __init__(self):
        rospy.init_node('r2d2_global_feature_node_simple')

        self._load_parameters()
        self._init_ros_communication()
        self._init_network()
        self._init_state()

        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_global_map)

        rospy.loginfo("R2D2 Global Feature Node (manual sync) initialized")

    def _load_parameters(self):
        # 模型相关
        self.model_path = rospy.get_param('~model_path', 'models/r2d2_WASF_N16.pt')
        self.num_keypoints = rospy.get_param('~num_keypoints', 1000)
        self.reliability_thr = rospy.get_param('~reliability_thr', 0.9)
        self.repeatability_thr = rospy.get_param('~repeatability_thr', 0.9)
        self.score_threshold = rospy.get_param('~score_threshold', 0.998)

        # 地图与发布
        self.merge_distance = rospy.get_param('~merge_distance', 0.1)
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.camera_frame = rospy.get_param("~camera_frame", "zedm_camera_center")

        # 相机内参
        self.fx = rospy.get_param('~fx', 365.19)
        self.fy = rospy.get_param('~fy', 365.19)
        self.cx = rospy.get_param('~cx', 316.83)
        self.cy = rospy.get_param('~cy', 182.29)
        self.img_width = rospy.get_param('~img_width', 1280)
        self.img_height = rospy.get_param('~img_height', 720)

        # 视锥 / raycasting 参数
        self.max_view_distance = rospy.get_param('~max_view_distance', 10.0)
        self.octomap_resolution = rospy.get_param('~octomap_resolution', 0.1)
        self.ray_step_factor = rospy.get_param('~ray_step_factor', 0.8)

        # 同步参数
        self.sync_queue_size = rospy.get_param('~sync_queue_size', 5)
        self.sync_slop = rospy.get_param('~sync_slop', 0.05)  # seconds

        # 话题名（可用 rosparam 覆盖）
        self.image_topic = rospy.get_param('~image_topic', '/zedm/zed_node/rgb/image_rect_color')
        self.pose_topic = rospy.get_param('~pose_topic', '/vrpn_client_node/kingfisher/pose')
        # self.pose_topic = rospy.get_param('~pose_topic', '/zedm/zed_node/pose')
        self.occ_centers_topic = rospy.get_param(
            '~occ_centers_topic', '/sdf_map/occupancy_all'
        )

    # ==================== ROS 通信 ====================

    def _init_ros_communication(self):
        self.bridge = CvBridge()

        # 手动同步：分别订阅图像和位姿
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )
        self.pose_sub = rospy.Subscriber(
            self.pose_topic, PoseStamped, self.pose_callback, queue_size=self.sync_queue_size * 10
        )

        # 占据体素中心
        self.octo_centers_sub = rospy.Subscriber(
            self.occ_centers_topic, PointCloud2,
            self.octomap_centers_callback, queue_size=1
        )

        # 可视化图像 & 全局特征点云
        self.vis_pub = rospy.Publisher('/r2d2/visualization', Image, queue_size=1)
        self.global_pub = rospy.Publisher("/r2d2/global_feature_map", PointCloud2, queue_size=1)

    # ==================== 网络 ====================

    def _init_network(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")

        self.net = load_network(self.model_path)
        self.net = self.net.to(self.device)
        self.net.eval()

        self.detector = NonMaxSuppression(
            rel_thr=self.reliability_thr,
            rep_thr=self.repeatability_thr
        )

    # ==================== 状态 ====================

    def _init_state(self):
        self.lock = Lock()
        self.global_points = []          # list of [x, y, z, intensity] in world frame
        self.kdtree = None

        self.occ_points = None          # occupied voxel centers
        self.occ_kdtree = None

        # 同步相关：保存最近一段时间的位姿
        self.sync_lock = Lock()
        self.pose_buffer = deque(maxlen=self.sync_queue_size * 20)

    # ==================== Octomap occupied centers ====================

    def octomap_centers_callback(self, msg: PointCloud2):
        """接收占据体素中心点云，构建 KDTree 用于 raycasting."""
        try:
            pts = []
            for p in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                pts.append([p[0], p[1], p[2]])

            if pts:
                self.occ_points = np.asarray(pts, dtype=np.float32)
                self.occ_kdtree = cKDTree(self.occ_points)
        except Exception as e:
            rospy.logwarn(f"Failed to parse octomap centers point cloud: {e}")

    # ==================== 工具函数：位姿/射线 ====================

    def pose_to_T_world_camera(self, pose_msg: PoseStamped) -> np.ndarray:
        """
        将无人机基座位姿转换为相机在 world 下的 4x4 变换矩阵.
        """
        pose = pose_msg.pose
        q = pose.orientation
        t = pose.position

        T_world_baselink = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T_world_baselink[0, 3] = t.x
        T_world_baselink[1, 3] = t.y
        T_world_baselink[2, 3] = t.z

        T_baselink_camera = np.eye(4)
        T_baselink_camera[0:3, 3] = self.CAMERA_TO_BASELINK_TRANSLATION

        T_world_camera = T_world_baselink @ T_baselink_camera
        return T_world_camera

    def raycast_to_octomap(self, u, v, T_wc):
        """
        从像素 (u, v) 发射射线，找到与占据体素的第一个交点.
        返回 world 坐标，未命中则返回 None.
        """
        if self.occ_kdtree is None or self.occ_points is None:
            rospy.logwarn_throttle(10.0, "Octomap centers not available for raycasting")
            return None

        cam_origin = T_wc[0:3, 3]

        # 像素 -> 相机坐标系中的方向（以 x 为前方）
        x_cam = 1.0
        y_cam = -(u - self.cx) * x_cam / self.fx
        z_cam = -(v - self.cy) * x_cam / self.fy
        ray_dir_cam = np.array([x_cam, y_cam, z_cam, 0.0])

        # 转到 world
        ray_dir_world = (T_wc @ ray_dir_cam)[0:3]
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)

        step = max(0.05, self.octomap_resolution * self.ray_step_factor)
        max_dist = self.max_view_distance
        n_steps = int(max_dist / step)

        for i in range(1, n_steps):
            current_point = cam_origin + ray_dir_world * (i * step)
            try:
                dist, idx = self.occ_kdtree.query(current_point, k=1)
                if dist < (self.octomap_resolution * 0.5):
                    return self.occ_points[idx]
            except Exception as e:
                rospy.logwarn_throttle(10.0, f"Raycast query failed: {e}")
                return None

        return None

    # ==================== 手动同步相关 ====================

    def pose_callback(self, pose_msg: PoseStamped):
        """位姿回调：简单地把位姿存入 buffer。"""
        with self.sync_lock:
            self.pose_buffer.append(pose_msg)

    def _get_closest_pose(self, stamp):
        """
        在 pose_buffer 中找到时间戳最接近 stamp 的 pose，
        若时间差小于 self.sync_slop（秒）则返回，否则返回 None。
        """
        if not self.pose_buffer:
            return None

        t_img = stamp.to_sec()
        best_pose = None
        best_dt = None
        max_dt = self.sync_slop

        # 遍历 buffer 找最近的时间
        for pose in self.pose_buffer:
            t_pose = pose.header.stamp.to_sec()
            dt = abs(t_pose - t_img)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_pose = pose

        if best_dt is not None and best_dt <= max_dt:
            return best_pose
        else:
            return None

    def image_callback(self, image_msg: Image):
        """
        图像回调：为该图像找一帧时间戳最近的位姿，然后调用原来的处理函数。
        """
        try:
            with self.sync_lock:
                pose_msg = self._get_closest_pose(image_msg.header.stamp)

            if pose_msg is None:
                # 没有找到足够接近的位姿就直接丢弃这一帧图像
                rospy.logdebug("No suitable pose found for image timestamp, skipping frame")
                return

            # 有匹配的位姿，执行原始逻辑
            self.process_image_and_pose(image_msg, pose_msg)

        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    # ==================== 核心处理 ====================

    def process_image_and_pose(self, image_msg: Image, pose_msg: PoseStamped):
        print("start process image")
        # 转成 numpy 图像
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        pil_image = PILImage.fromarray(cv_image)

        img = norm_RGB(pil_image)[None].to(self.device)

        # R2D2 特征提取
        start_time = time.time()
        xys, desc, scores = extract_multiscale(
            self.net, img, self.detector,
            scale_f=2**0.25,
            min_scale=0.0,
            max_scale=1.0,
            min_size=256,
            max_size=1024
        )
        rospy.logdebug(f"Feature extraction time: {time.time() - start_time:.3f}s")

        xys = xys.cpu().numpy()     # (N, 3) -> x, y, scale
        scores = scores.cpu().numpy()

        # 选 top K
        if xys.shape[0] == 0:
            return

        idxs = scores.argsort()[-self.num_keypoints:]
        xys_sel = xys[idxs]
        scores_sel = scores[idxs]

        # 相机位姿
        T_wc = self.pose_to_T_world_camera(pose_msg)

        # raycasting 得到 world 下 3D 点
        world_points = []      # [x, y, z, intensity]
        vis_points_2d = []     # (u, v, intensity) 用于画图

        print(f"maximum score: {np.max(scores_sel)}")
        print(f"minimum score: {np.min(scores_sel)}")

        for (x, y, scale), s in zip(xys_sel, scores_sel):
            if s < self.score_threshold:
                continue

            if x < 0 or x >= self.img_width or y < 0 or y >= self.img_height:
                continue

            pw = self.raycast_to_octomap(x, y, T_wc)
            if pw is not None:
                world_points.append([pw[0], pw[1], pw[2], float(s)])
                vis_points_2d.append((int(x), int(y), float(s)))

        # 更新 global map
        if world_points:
            self._update_global_map(world_points)
            self._visualize_features(cv_image, vis_points_2d)

        # 发布可视化图像
        vis_msg = self.bridge.cv2_to_imgmsg(cv_image, "rgb8")
        self.vis_pub.publish(vis_msg)

    # ==================== 全局地图维护 ====================

    def _update_global_map(self, new_points_world):
        """
        new_points_world: list of [x, y, z, intensity] in world frame
        利用 KDTree 做近邻合并.
        """
        new_points = np.asarray(new_points_world, dtype=np.float32)

        # 仅保留在工作空间内的特征点
        if new_points.size == 0:
            return
        mask = (
            (new_points[:, 0] >= -5.0) & (new_points[:, 0] <= 5.0) &
            (new_points[:, 1] >= -5.0) & (new_points[:, 1] <= 5.0) &
            (new_points[:, 2] >= 0.3)  & (new_points[:, 2] <= 2.0)
        )
        new_points = new_points[mask]
        if new_points.size == 0:
            return

        with self.lock:
            if len(self.global_points) == 0:
                self.global_points = new_points.tolist()
                self.kdtree = cKDTree(new_points[:, :3])
                return

            existing = np.asarray(self.global_points, dtype=np.float32)
            self.kdtree = cKDTree(existing[:, :3])

            for pt in new_points:
                dist, idx = self.kdtree.query(pt[:3], k=1)
                if dist < self.merge_distance:
                    # 用更高的强度更新
                    if pt[3] > existing[idx, 3]:
                        existing[idx, 3] = pt[3]
                else:
                    existing = np.vstack([existing, pt])

            self.global_points = existing.tolist()
            self.kdtree = cKDTree(existing[:, :3])

    def _visualize_features(self, vis_img, vis_points_2d):
        """
        在图像上画出成功 raycast 的特征点.
        """
        for u, v, intensity in vis_points_2d:
            # 简单归一化到 [0, 255]
            norm_intensity = float(np.clip((intensity - 0.8) / 0.2, 0.0, 1.0))
            color_intensity = int(norm_intensity * 255)
            color = (0, 255, color_intensity)  # BGR

            cv2.circle(vis_img, (u, v), 3, color, -1)
            cv2.circle(vis_img, (u, v), 5, (0, 255, 0), 1)

    # ==================== 发布全局点云 ====================

    def publish_global_map(self, event):
        with self.lock:
            if not self.global_points:
                return
            points = np.asarray(self.global_points, dtype=np.float32)

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points)
        self.global_pub.publish(pc_msg)


def main():
    try:
        node = R2D2GlobalFeatureNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
