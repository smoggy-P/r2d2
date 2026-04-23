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
    def __init__(self, world_name, total_experiments, max_exploration_time=300, method='vo_safe', record_rosbag=True):
        self.world_name = world_name
        self.total_experiments = total_experiments
        self.max_exploration_time = max_exploration_time  # Maximum time to wait for exploration completion (seconds)
        self.experiment_dir = "./experiments"
        self.log_file = os.path.join(self.experiment_dir, "experiment_log.txt")
        self.method = method
        self.record_rosbag = record_rosbag
        self.conda_profile = "/home/smoggy/anaconda3/etc/profile.d/conda.sh"
        # Subdirectory per method for bag files
        self.method_dir = os.path.join(self.experiment_dir, str(self.method))
        
        # Results tracking
        self.success_count = 0
        self.init_fail_count = 0
        self.init_drift_count = 0
        self.exploration_fail_count = 0  # New counter for exploration failures
        
        # Process tracking
        self.processes = {}
        self.window_ids = {}
        # Track containers started by this run so we can stop them later
        self.started_containers = set()
        
        # Create experiment directory first
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.method_dir, exist_ok=True)
        
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
        
        # Kill Docker container processes
        try:
            # Kill Visual Odometry processes in Docker container
            subprocess.run(['docker', 'exec', 'c1ad613f28a6', 'pkill', '-f', 'roslaunch.*subscribe.launch'], 
                         timeout=10, capture_output=True)
            subprocess.run(['docker', 'exec', 'c1ad613f28a6', 'pkill', '-f', 'ov_msckf'], 
                         timeout=10, capture_output=True)
            subprocess.run(['docker', 'exec', 'c1ad613f28a6', 'pkill', '-f', 'exploration_manager'], 
                         timeout=10, capture_output=True)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Stop containers that were started by this run
        for cid in list(getattr(self, 'started_containers', set())):
            try:
                subprocess.run(['docker', 'stop', cid], timeout=15, capture_output=True)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Kill any remaining ROS processes
        ros_processes = [
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
                matches = re.findall(r'dist\s*=\s*[^0-9-]*([0-9]+(?:\.[0-9]+)?)', content)
                if matches:
                    distance = float(matches[-1])
                    
                    if distance > 2.0:
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
        fuel_log = f"/tmp/fuel_log_{exp_num}.txt"
        la_planner_log = f"/tmp/la_planner_log_{exp_num}.txt"
        
        # Clean up any existing log files
        for log_file in [vo_log, sim_log, r2d2_log, fuel_log, la_planner_log]:
            try:
                os.remove(log_file)
            except FileNotFoundError:
                pass
        
        try:
            # Ensure required containers are started
            try:
                result_vo = subprocess.run(['docker', 'start', 'c1ad613f28a6'], capture_output=True)
                if result_vo.returncode == 0:
                    self.started_containers.add('c1ad613f28a6')
            except FileNotFoundError:
                pass
            
            if self.method in ['fuel', 'la_planner']:
                try:
                    result_aux = subprocess.run(['docker', 'start', '09c1ac37930c'], capture_output=True)
                    if result_aux.returncode == 0:
                        self.started_containers.add('09c1ac37930c')
                except FileNotFoundError:
                    pass
            
            if self.method == 'la_planner':
                self.log_message("Starting LA Planner...")
                la_planner_cmd = [
                    'gnome-terminal', '--title', f'LA_Planner_Exp_{exp_num}',
                    '--', 'bash', '-c',
                    f"docker exec -it 09c1ac37930c bash -c 'cd ~/la_ws && source devel/setup.bash && roslaunch exploration_manager run_map1.launch' 2>&1 | tee '{la_planner_log}'"
                ]
                self.processes['la_planner'] = subprocess.Popen(la_planner_cmd)
                time.sleep(2)
                self.window_ids['la_planner'] = self.get_window_id(f'LA_Planner_Exp_{exp_num}')
            
            # 1. Start Visual Odometry in new terminal
            self.log_message("Starting Visual Odometry...")
            vo_cmd = [
                'gnome-terminal', '--title', f'VO_Exp_{exp_num}',
                '--', 'bash', '-c',
                f"docker exec -it c1ad613f28a6 bash -c 'source /catkin_ws/devel/setup.bash && roslaunch ov_msckf subscribe.launch config:=agiros' 2>&1 | tee '{vo_log}'"
            ]
            self.processes['vo'] = subprocess.Popen(vo_cmd)
            time.sleep(2)
            self.window_ids['vo'] = self.get_window_id(f'VO_Exp_{exp_num}')
            
            # 2. Start Simulation in new terminal
            self.log_message("Starting Simulation...")
            if self.method == 'vo_safe':
                sim_cmd = [
                    'gnome-terminal', '--title', f'Sim_Exp_{exp_num}',
                    '--', 'bash', '-c',
                    f"""set -eo pipefail
                    cd /home/smoggy/workspace_ros1/vo_safe_ws
                    [ -f "{self.conda_profile}" ] && source "{self.conda_profile}"
                    conda deactivate >/dev/null 2>&1 || true
                    source devel/setup.bash --extend
                    roslaunch agiros simulation.launch world_name:="/home/smoggy/workspace_ros1/vo_safe_ws/src/vo_safe_exploration/vo_safe_frontier/experiment/worlds/{self.world_name}.world" 2>&1 | tee '{sim_log}'"""
                ]
            else:
                sim_cmd = [
                    'gnome-terminal', '--title', f'Sim_Exp_{exp_num}',
                    '--', 'bash', '-c',
                    f"""set -eo pipefail
                    cd /home/smoggy/workspace_ros1/vo_safe_ws
                    [ -f "{self.conda_profile}" ] && source "{self.conda_profile}"
                    conda deactivate >/dev/null 2>&1 || true
                    source devel/setup.bash --extend
                    roslaunch agiros simulation_fuel.launch world_name:="/home/smoggy/workspace_ros1/vo_safe_ws/src/vo_safe_exploration/vo_safe_frontier/experiment/worlds/{self.world_name}.world" 2>&1 | tee '{sim_log}'"""
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
                f"""set -eo pipefail
                cd /home/smoggy/workspace_ros1/r2d2
                source "{self.conda_profile}"
                conda activate r2d2-gpu
                source /home/smoggy/workspace_ros1/vo_safe_ws/devel/setup.bash
                python extract_ros_global.py 2>&1 | tee '{r2d2_log}'"""
            ]
            self.processes['r2d2'] = subprocess.Popen(r2d2_cmd)
            time.sleep(2)
            self.window_ids['r2d2'] = self.get_window_id(f'R2D2_Exp_{exp_num}')

            if self.method == 'fuel':
                self.log_message("Starting Fuel...")
                fuel_cmd = [
                    'gnome-terminal', '--title', f'Fuel_Exp_{exp_num}',
                    '--', 'bash', '-c',
                    f"docker exec -it 09c1ac37930c bash -c 'cd ~/fuel_ws && source devel/setup.bash && roslaunch exploration_manager exploration.launch' 2>&1 | tee '{fuel_log}'"
                ]
                self.processes['fuel'] = subprocess.Popen(fuel_cmd)
                time.sleep(2)
                self.window_ids['fuel'] = self.get_window_id(f'Fuel_Exp_{exp_num}')
                subprocess.run(['rostopic', 'pub', '-1', '/move_base_simple/goal',
                                'geometry_msgs/PoseStamped',
                                '{ header: { stamp: now, frame_id: "map" }, '
                                'pose: { position: {x: 1.0, y: 2.0, z: 0.0}, '
                                'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0} } }'
                            ], timeout=10)
            
            # 7. Start rosbag recording in new terminal
            if self.record_rosbag:
                bag_name = f"{self.world_name}_exp_{exp_num}_{datetime.now().strftime('%H%M%S')}"
                self.log_message(f"Starting rosbag recording: {bag_name}")
                bag_output_dir = os.path.abspath(os.path.join(self.experiment_dir, self.method, "_new"))
                os.makedirs(bag_output_dir, exist_ok=True)
                bag_output_base = os.path.join(bag_output_dir, bag_name)
                rosbag_cmd = [
                    'gnome-terminal', '--title', f'Rosbag_Exp_{exp_num}',
                    '--', 'bash', '-c',
                    f"""cd ~/workspace_ros1/r2d2 && 
                    rosbag record -O '{bag_output_base}' /ov_msckf/loop_pose /kingfisher/ground_truth/odometry /exploration_rate /ov_msckf/loop_feats /sdf_map/occupancy_all /sdf_map/occupancy_local /r2d2/visible_features_uv"""
                ]
                self.processes['rosbag'] = subprocess.Popen(rosbag_cmd)
                time.sleep(2)
                self.window_ids['rosbag'] = self.get_window_id(f'Rosbag_Exp_{exp_num}')
            else:
                self.log_message("Rosbag recording disabled by parameter", 'YELLOW')
            
            # 8. Monitor for "No frontiers found"
            if self.method == 'vo_safe':
                end_message = "No frontiers found"
                monitored_log = sim_log
            elif self.method == 'fuel':
                end_message = "No coverable frontier."
                monitored_log = fuel_log
            else:
                end_message = "No feature viewpoint, wait for target"
                monitored_log = la_planner_log
            self.log_message(f"Monitoring for '{end_message}' (max wait: {self.max_exploration_time} seconds)...")
            if self.wait_for_message(monitored_log, end_message, self.max_exploration_time):
                self.log_message(f"Experiment {exp_num} completed successfully!", 'GREEN')
                self.success_count += 1
                
                # Stop rosbag recording gracefully
                if self.record_rosbag:
                    self.log_message("Stopping rosbag recording...")
                    if 'rosbag' in self.processes and self.processes['rosbag']:
                        try:
                            # Send SIGINT to rosbag process to stop recording gracefully
                            self.processes['rosbag'].send_signal(signal.SIGINT)
                            self.processes['rosbag'].wait(timeout=10)
                            self.log_message("Rosbag stopped successfully")
                        except (subprocess.TimeoutExpired, ProcessLookupError):
                            try:
                                self.processes['rosbag'].kill()
                            except ProcessLookupError:
                                pass
                    
                    # Wait a bit more for file system to complete writing
                    time.sleep(3)
                    
                    # Finalize bag and cleanup residual files
                    bag_path = f"{bag_output_base}.bag"
                    active_bag_path = f"{bag_output_base}.bag.active"
                    orig_active_bag_path = f"{bag_output_base}.bag.orig.active"

                    if not os.path.exists(bag_path) and os.path.exists(active_bag_path):
                        self.log_message("Found .bag.active file, attempting to finalize...")
                        try:
                            # Use rosbag fix to convert .bag.active to .bag
                            subprocess.run(['rosbag', 'reindex', active_bag_path], 
                                        timeout=30, check=True)
                            subprocess.run(['rosbag', 'fix', active_bag_path, bag_path], 
                                        timeout=60, check=True)
                        except subprocess.CalledProcessError:
                            self.log_message("Failed to finalize rosbag from .active file", 'RED')
                    
                    # Cleanup residual .active files
                    for leftover in [active_bag_path, orig_active_bag_path]:
                        try:
                            if os.path.exists(leftover):
                                os.remove(leftover)
                        except Exception:
                            pass
                    
                    if os.path.exists(bag_path):
                        self.log_message(f"Rosbag saved to {bag_path}")
                    else:
                        self.log_message("No rosbag file found", 'RED')
                
                return True
            else:
                self.log_message(f"Experiment {exp_num} exploration timed out after {self.max_exploration_time} seconds", 'RED')
                self.exploration_fail_count += 1
                
                # Try to stop rosbag and cleanup files on timeout
                if self.record_rosbag:
                    self.log_message("Stopping rosbag recording after timeout...")
                    if 'rosbag' in self.processes and self.processes['rosbag']:
                        try:
                            self.processes['rosbag'].send_signal(signal.SIGINT)
                            self.processes['rosbag'].wait(timeout=10)
                            self.log_message("Rosbag stopped successfully")
                        except (subprocess.TimeoutExpired, ProcessLookupError):
                            try:
                                self.processes['rosbag'].kill()
                            except ProcessLookupError:
                                pass
                    
                    time.sleep(3)
                    
                    # Cleanup residual bag files
                    bag_path = f"{bag_output_base}.bag"
                    active_bag_path = f"{bag_output_base}.bag.active"
                    orig_active_bag_path = f"{bag_output_base}.bag.orig.active"

                    if not os.path.exists(bag_path) and os.path.exists(active_bag_path):
                        try:
                            subprocess.run(['rosbag', 'reindex', active_bag_path], timeout=30, check=True)
                            subprocess.run(['rosbag', 'fix', active_bag_path, bag_path], timeout=60, check=True)
                        except subprocess.CalledProcessError:
                            pass

                    for leftover in [active_bag_path, orig_active_bag_path]:
                        try:
                            if os.path.exists(leftover):
                                os.remove(leftover)
                        except Exception:
                            pass
                
                # Clean up all processes before starting next experiment
                self.cleanup_terminals()
                return False
                
        except Exception as e:
            self.log_message(f"Error in experiment {exp_num}: {str(e)}", 'RED')
            return False
        finally:
            self.cleanup_terminals()
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        dependencies = ['wmctrl', 'gnome-terminal', 'rostopic', 'docker']
        if getattr(self, 'record_rosbag', True):
            dependencies.append('rosbag')
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
        self.log_message(f"Exploration failures: {self.exploration_fail_count}", 'RED')
        
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
    parser.add_argument('--max-exploration-time', type=int, default=300, 
                       help='Maximum time to wait for exploration completion in seconds (default: 200)')
    parser.add_argument('--method', type=str, default='vo_safe',
                       help='Method to use for exploration (default: vo_safe)')
    parser.add_argument('--no-rosbag', action='store_true',
                       help='Disable rosbag recording')
    args = parser.parse_args()
    
    if args.total_experiments <= 0:
        print("Error: total_experiments must be a positive integer")
        sys.exit(1)
    
    runner = ExperimentRunner(args.world_name, args.total_experiments, args.max_exploration_time, args.method, record_rosbag=(not args.no_rosbag))
    runner.run_experiments()

if __name__ == "__main__":
    main()
