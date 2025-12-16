#!/usr/bin/env python3
# ===================================================================
# Bio ANNa - Data Collection Node
#
# Author: Qwen Assistant
# Date: Nov 24, 2025
#
# Description:
# This node collects all the necessary data for generating the supplemental
# materials (figures and tables) for the Bio ANNa project. It subscribes
# to all relevant topics and saves the data to CSV files for later analysis.
# ===================================================================

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
from datetime import datetime

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class DataCollectionNode(Node):
    """
    Node for collecting data from all relevant topics in the Bio ANNa system.
    """
    def __init__(self):
        super().__init__('data_collection_node')
        self.get_logger().info('Initializing Data Collection Node...')
        
        # Create data storage
        self.data = {
            'timestamp': [],
            'antbot_x': [], 'antbot_y': [], 'antbot_theta': [],
            'fused_x': [], 'fused_y': [], 'fused_theta': [],
            'gridcore_x': [], 'gridcore_y': [], 'gridcore_theta': [],
            'cmd_vel_linear': [], 'cmd_vel_angular': [],
            'imu_angular_z': [], 'imu_linear_x': []
        }
        
        # Setup subscribers
        self.antbot_odom_sub = self.create_subscription(
            Odometry, '/odom/antbot_snn', self.antbot_callback, 10)
            
        self.fused_odom_sub = self.create_subscription(
            Odometry, '/odom/fused', self.fused_callback, 10)
            
        self.gridcore_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/pose_correction/gridcore', self.gridcore_callback, 10)
            
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
            
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # Timer to save data periodically
        self.timer = self.create_timer(5.0, self.save_data)
        self.get_logger().info('Data Collection Node initialized.')
        
    def extract_yaw(self, orientation):
        """Extract yaw from quaternion."""
        # Simplified extraction (in a real implementation, you'd use tf2 or similar)
        return 2 * np.arctan2(orientation.z, orientation.w)
    
    def antbot_callback(self, msg):
        """Callback for AntBot odometry data."""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.data['timestamp'].append(timestamp)
        self.data['antbot_x'].append(msg.pose.pose.position.x)
        self.data['antbot_y'].append(msg.pose.pose.position.y)
        yaw = self.extract_yaw(msg.pose.pose.orientation)
        self.data['antbot_theta'].append(yaw)
        
    def fused_callback(self, msg):
        """Callback for fused odometry data."""
        # We don't append timestamp here since it's handled by other callbacks
        self.data['fused_x'].append(msg.pose.pose.position.x)
        self.data['fused_y'].append(msg.pose.pose.position.y)
        yaw = self.extract_yaw(msg.pose.pose.orientation)
        self.data['fused_theta'].append(yaw)
        
    def gridcore_callback(self, msg):
        """Callback for GridCore pose correction data."""
        # We don't append timestamp here since it's handled by other callbacks
        self.data['gridcore_x'].append(msg.pose.pose.position.x)
        self.data['gridcore_y'].append(msg.pose.pose.position.y)
        yaw = self.extract_yaw(msg.pose.pose.orientation)
        self.data['gridcore_theta'].append(yaw)
        
    def cmd_vel_callback(self, msg):
        """Callback for command velocity data."""
        # We don't append timestamp here since it's handled by other callbacks
        self.data['cmd_vel_linear'].append(msg.linear.x)
        self.data['cmd_vel_angular'].append(msg.angular.z)
        
    def imu_callback(self, msg):
        """Callback for IMU data."""
        # We don't append timestamp here since it's handled by other callbacks
        self.data['imu_angular_z'].append(msg.angular_velocity.z)
        self.data['imu_linear_x'].append(msg.linear_acceleration.x)
    
    def save_data(self):
        """Save collected data to CSV files."""
        # Convert to DataFrame
        df = pd.DataFrame(self.data)
        
        # Create output directory if it doesn't exist
        output_dir = 'datasets/experiment_data'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{output_dir}/experiment_data_{timestamp}.csv'
        df.to_csv(filename, index=False)
        self.get_logger().info(f'Data saved to {filename}')
        
        # Also save a summary statistics file
        summary_filename = f'{output_dir}/summary_statistics_{timestamp}.csv'
        summary_stats = df.describe()
        summary_stats.to_csv(summary_filename)
        self.get_logger().info(f'Summary statistics saved to {summary_filename}')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save any remaining data before shutdown
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
