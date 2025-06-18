#!/bin/bash
# 激活conda环境但只使用PyTorch相关的包
export PYTHONPATH=/home/smoggy/miniforge3/envs/r2d2_py38/lib/python3.8/site-packages:$PYTHONPATH
export LD_PRELOAD=/home/smoggy/miniforge3/envs/r2d2_py38/lib/libtiff.so.5
# 运行ROS节点
python3 extract_ros.py