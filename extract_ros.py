#!/usr/bin/env python3

# 首先导入ROS相关包
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 然后导入conda环境中的包
import torch
import numpy as np
import cv2
from PIL import Image as PILImage
import torch.nn.functional as F

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
from extract import load_network, NonMaxSuppression, extract_multiscale
from sensor_msgs.msg import PointCloud2
class R2D2ExtractorNode:
    def __init__(self):
        rospy.init_node('r2d2_extractor_node')
        
        # 获取ROS参数
        self.model_path = rospy.get_param('~model_path', 'models/r2d2_WASF_N16.pt')
        self.num_keypoints = rospy.get_param('~num_keypoints', 1000)
        self.reliability_thr = rospy.get_param('~reliability_thr', 0.7)
        self.repeatability_thr = rospy.get_param('~repeatability_thr', 0.7)
        
        # 设置发布者和订阅者
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/D435i_camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/D435i_camera/depth/image_rect_raw", Image, self.depth_callback)
        self.vis_pub = rospy.Publisher('/r2d2/visualization', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('/r2d2/point_cloud', PointCloud2, queue_size=1)
        # Get the camera info
        self.fx = rospy.get_param('~fx', 695.99511719)
        self.fy = rospy.get_param('~fy', 695.99511719)
        self.cx = rospy.get_param('~cx', 640)
        self.cy = rospy.get_param('~cy', 360)
        self.depth_img = None
        
        # 加载网络
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = load_network(self.model_path)
        self.net = self.net.to(self.device)
        self.net.eval()
        
        # 创建非极大值抑制检测器
        self.detector = NonMaxSuppression(
            rel_thr=self.reliability_thr,
            rep_thr=self.repeatability_thr
        )
        
        rospy.loginfo("R2D2 extractor node initialized")
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
            xys, desc, scores = extract_multiscale(
                self.net, img, self.detector,
                scale_f=2**0.25,
                min_scale=0.0,
                max_scale=1.0,
                min_size=256,
                max_size=1024
            )
            
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
            norm_scores = (selected_scores - selected_scores.min()) / (selected_scores.max() - selected_scores.min())
            
            if self.depth_img is not None:
                points = []
                for kp, score in zip(selected_kpts, norm_scores):
                    x, y, scale = kp
                    # Project the keypoint to the depth image and publish point cloud
                    if x < self.depth_img.shape[1] and y < self.depth_img.shape[0]:
                        depth = self.depth_img[int(y), int(x)]
                        if depth > 0:
                            depth_meters = depth / 1000.0

                            x_world = depth_meters
                            y_world = -(x - self.cx) * depth_meters / self.fx
                            z_world = -(y - self.cy) * depth_meters / self.fy
                            points.append((x_world, y_world, z_world, score))
            if len(points) > 0:
                # 创建PointCloud2消息
                from sensor_msgs.msg import PointField
                import sensor_msgs.point_cloud2 as pc2
                
                # 定义点云字段
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1)
                ]
                
                # 创建点云消息
                header = msg.header
                header.frame_id = "D435i_camera_depth_frame"
                pc_msg = pc2.create_cloud(header, fields, points)
                
                # 发布点云
                self.pc_pub.publish(pc_msg)
            
            # 发布可视化结果
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "rgb8")
            self.vis_pub.publish(vis_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    try:
        node = R2D2ExtractorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
