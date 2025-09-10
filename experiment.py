#!/usr/bin/env python3
"""
Visual Odometry Simulation Experiment Script (Python Version)
Usage: python experiment.py <world_name> <total_experiments>
Example: python experiment.py test_world1 10
"""

import os
import sys
import time
import subprocess
import threading
import signal
import argparse
from datetime import datetime
from pathlib import Path
import re
import psutil
import logging

class ExperimentRunner:
    def __init__(self, world_name, total_experiments):
        self.world_name = world_name
        self.total_experiments = total_experiments
        self.experiment_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = os.path.join(self.experiment_dir, "experiment_log.txt")
        
        # Results tracking
        self.success_count = 0
        self.init_fail_count = 0
        self.init_drift_count = 0
        
        # Process tracking
        self.processes = {}
        self.window_ids = {}
        
        # Create experiment directory first
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logging after directory is created
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_message(self, message, color=None):
        """Log message with optional color formatting"""
        if color:
            # ANSI color codes
            colors = {
                'RED': '\033[0;31m',
                'GREEN': '\033[0;32m',
                'YELLOW': '\033[1;33m',
                'BLUE': '\033[0;34m',
                'NC': '\033[0m'  # No Color
            }
            message = f"{colors.get(color, '')}{message}{colors['NC']}"
        
        self.logger.info(message)
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.log_message("Received interrupt signal, cleaning up...", 'YELLOW')
        self.cleanup_terminals()
        sys.exit(0)
    
    def cleanup_terminals(self):
        """Clean up all running processes and terminal windows"""
        self.log_message("Cleaning up terminals...", 'YELLOW')
        
        # Kill all tracked processes
        for name, process in self.processes.items():
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
        
        # Close terminal windows using wmctrl
        for name, window_id in self.window_ids.items():
            if window_id:
                try:
                    subprocess.run(['wmctrl', '-ic', window_id], 
                                 capture_output=True, timeout=5)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
        
        # Kill any remaining ROS processes
        ros_processes = [
            "roslaunch.*subscribe.launch",
            "roslaunch.*simulation.launch", 
            "python.*extract_ros_global.py",
            "rosbag.*record"
        ]
        
        for pattern in ros_processes:
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['cmdline'] and any(pattern.replace('.*', '') in ' '.join(proc.info['cmdline']) for pattern in ros_processes):
                            proc.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception:
                pass
        
        time.sleep(2)
    
    def wait_for_message(self, log_file, pattern, timeout):
        """Wait for a specific pattern to appear in log file"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if pattern in content:
                        return True
            except FileNotFoundError:
                pass
            time.sleep(0.1)
        
        return False
    
    def monitor_distance(self, log_file, timeout):
        """Monitor distance for drift detection"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                # Extract distance value using regex
                distance_match = re.search(r'dist = ([0-9.]+)', content)
                if distance_match:
                    distance = float(distance_match.group(1))
                    
                    if distance > 1.0:
                        self.log_message(f"Distance drift detected: {distance} meters", 'RED')
                        return False
                    
                    self.log_message(f"Distance: {distance} meters")
                    
            except (FileNotFoundError, ValueError):
                pass
            
            time.sleep(0.5)
        
        return True
    
    def get_window_id(self, title):
        """Get window ID by title using wmctrl"""
        try:
            result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                if title in line:
                    return line.split()[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def run_single_experiment(self, exp_num):
        """Run a single experiment"""
        self.log_message(f"=== Starting Experiment {exp_num}/{self.total_experiments} ===", 'BLUE')
        
        # Create temporary log files
        vo_log = f"/tmp/vo_log_{exp_num}.txt"
        sim_log = f"/tmp/sim_log_{exp_num}.txt"
        r2d2_log = f"/tmp/r2d2_log_{exp_num}.txt"
        
        # Clean up any existing log files
        for log_file in [vo_log, sim_log, r2d2_log]:
            try:
                os.remove(log_file)
            except FileNotFoundError:
                pass
        
        try:
            # 1. Start Visual Odometry in new terminal
            self.log_message("Starting Visual Odometry...")
            vo_cmd = [
                'gnome-terminal', '--title', f'VO_Exp_{exp_num}',
                '--', 'bash', '-c',
                f"docker exec -it c1ad613f28a6 bash -c 'source /catkin_ws/devel/setup.bash && roslaunch ov_msckf subscribe.launch config:=agiros' 2>&1 | tee '{vo_log}'; read -p 'Press Enter to close...'"
            ]
            self.processes['vo'] = subprocess.Popen(vo_cmd)
            time.sleep(2)
            self.window_ids['vo'] = self.get_window_id(f'VO_Exp_{exp_num}')
            
            # 2. Start Simulation in new terminal
            self.log_message("Starting Simulation...")
            sim_cmd = [
                'gnome-terminal', '--title', f'Sim_Exp_{exp_num}',
                '--', 'bash', '-c',
                f"""cd ~/workspace_ros1/vo_safe_ws && 
                source ~/.bashrc &&
                eval "$(conda shell.bash hook)" &&
                conda deactivate &&
                source devel/setup.bash --extend &&
                roslaunch agiros simulation.launch world_name:="/home/smoggy/workspace_ros1/vo_safe_ws/src/vo_safe_exploration/vo_safe_frontier/experiment/worlds/test_features.world" 2>&1 | tee '{sim_log}'; read -p 'Press Enter to close...'"""
            ]
            self.processes['sim'] = subprocess.Popen(sim_cmd)
            time.sleep(2)
            self.window_ids['sim'] = self.get_window_id(f'Sim_Exp_{exp_num}')
            
            # Wait 5 seconds before takeoff commands
            self.log_message("Waiting 5 seconds before takeoff...")
            time.sleep(5)
            
            # 3. Send takeoff commands
            self.log_message("Sending takeoff commands...")
            subprocess.run(['rostopic', 'pub', '-1', '/kingfisher/agiros_pilot/enable', 
                          'std_msgs/Bool', 'data: true'], timeout=10)
            time.sleep(1)
            subprocess.run(['rostopic', 'pub', '-1', '/kingfisher/agiros_pilot/start', 
                          'std_msgs/Empty', '{}'], timeout=10)
            
            # 4. Monitor for successful initialization
            self.log_message("Monitoring for successful initialization...")
            if self.wait_for_message(vo_log, "successful initialization", 2):
                self.log_message("Visual odometry initialized successfully", 'GREEN')
            else:
                self.log_message("Initialization failed - timeout", 'RED')
                self.init_fail_count += 1
                return False
            
            # 5. Monitor distance for drift
            self.log_message("Monitoring distance for drift (10 seconds)...")
            if not self.monitor_distance(vo_log, 10):
                self.log_message("Initialization drift detected", 'RED')
                self.init_drift_count += 1
                return False
            
            # 6. Start R2D2 in new terminal
            self.log_message("Starting R2D2...")
            r2d2_cmd = [
                'gnome-terminal', '--title', f'R2D2_Exp_{exp_num}',
                '--', 'bash', '-c',
                f"""cd ~/workspace_ros1/r2d2 && 
                source ~/.bashrc &&
                eval "$(conda shell.bash hook)" &&
                conda activate r2d2-gpu && 
                python extract_ros_global.py 2>&1 | tee '{r2d2_log}'; read -p 'Press Enter to close...'"""
            ]
            self.processes['r2d2'] = subprocess.Popen(r2d2_cmd)
            time.sleep(2)
            self.window_ids['r2d2'] = self.get_window_id(f'R2D2_Exp_{exp_num}')
            
            # 7. Start rosbag recording in new terminal
            bag_name = f"{self.world_name}_exp_{exp_num}_{datetime.now().strftime('%H%M%S')}"
            self.log_message(f"Starting rosbag recording: {bag_name}")
            rosbag_cmd = [
                'gnome-terminal', '--title', f'Rosbag_Exp_{exp_num}',
                '--', 'bash', '-c',
                f"""cd ~/workspace_ros1/r2d2 && 
                rosbag record -O '{bag_name}' /ov_msckf/loop_pose /kingfisher/ground_truth/odometry /exploration_rate /ov_msckf/loop_feats /r2d2/point_cloud /sdf_map/occupancy_all /sdf_map/occupancy_local /r2d2/global_feature_map; read -p 'Press Enter to close...'"""
            ]
            self.processes['rosbag'] = subprocess.Popen(rosbag_cmd)
            time.sleep(2)
            self.window_ids['rosbag'] = self.get_window_id(f'Rosbag_Exp_{exp_num}')
            
            # 8. Monitor for "No frontiers found"
            self.log_message("Monitoring for 'No frontiers found'...")
            max_wait = 300  # Maximum wait time (5 minutes)
            if self.wait_for_message(sim_log, "No frontiers found", max_wait):
                self.log_message(f"Experiment {exp_num} completed successfully!", 'GREEN')
                self.success_count += 1
                
                # Copy rosbag to results directory
                time.sleep(2)  # Give rosbag time to finish writing
                bag_path = os.path.expanduser(f"~/workspace_ros1/r2d2/{bag_name}.bag")
                if os.path.exists(bag_path):
                    dest_path = os.path.join(self.experiment_dir, f"{bag_name}.bag")
                    subprocess.run(['cp', bag_path, dest_path], timeout=10)
                    self.log_message(f"Rosbag saved to {dest_path}")
                
                return True
            else:
                self.log_message(f"Experiment {exp_num} timed out waiting for completion", 'RED')
                return False
                
        except Exception as e:
            self.log_message(f"Error in experiment {exp_num}: {str(e)}", 'RED')
            return False
        finally:
            self.cleanup_terminals()
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        dependencies = ['wmctrl', 'gnome-terminal', 'rostopic', 'rosbag']
        missing = []
        
        for dep in dependencies:
            try:
                subprocess.run(['which', dep], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                missing.append(dep)
        
        if missing:
            self.log_message(f"Missing dependencies: {', '.join(missing)}", 'RED')
            if 'wmctrl' in missing:
                self.log_message("Please install wmctrl: sudo apt-get install wmctrl", 'RED')
            return False
        
        return True
    
    def run_experiments(self):
        """Run all experiments"""
        self.log_message(f"Starting experiment series: {self.total_experiments} experiments with world {self.world_name}", 'GREEN')
        self.log_message(f"Results will be saved to: {self.experiment_dir}")
        
        # Check dependencies
        if not self.check_dependencies():
            return
        
        # Run experiments
        for i in range(1, self.total_experiments + 1):
            self.run_single_experiment(i)
            
            if i < self.total_experiments:
                self.log_message("Waiting 5 seconds before next experiment...", 'YELLOW')
                time.sleep(5)
        
        # Final results
        self.log_message("=== EXPERIMENT SUMMARY ===", 'BLUE')
        self.log_message(f"Total experiments: {self.total_experiments}")
        self.log_message(f"Successful: {self.success_count}", 'GREEN')
        self.log_message(f"Initialization failures: {self.init_fail_count}", 'RED')
        self.log_message(f"Initialization drift: {self.init_drift_count}", 'RED')
        
        success_rate = (self.success_count * 100) / self.total_experiments
        self.log_message(f"Success rate: {success_rate:.2f}%")
        self.log_message(f"Results saved in: {self.experiment_dir}")
        
        print(f"\n{'='*50}")
        print(f"Experiment series completed!")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Results saved in: {self.experiment_dir}")
        print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description='Visual Odometry Simulation Experiment Script')
    parser.add_argument('world_name', help='Name of the world to use for experiments')
    parser.add_argument('total_experiments', type=int, help='Total number of experiments to run')
    
    args = parser.parse_args()
    
    if args.total_experiments <= 0:
        print("Error: total_experiments must be a positive integer")
        sys.exit(1)
    
    runner = ExperimentRunner(args.world_name, args.total_experiments)
    runner.run_experiments()

if __name__ == "__main__":
    main()
