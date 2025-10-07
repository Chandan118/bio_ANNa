# ===================================================================
# Bio ANNa - Hardware Interface Node
#
# Author: chandan sheikder
# Date: o7 oct 2025
#
# Description:
# This node serves as the direct interface to the robot's hardware base
# and its core sensors (IMU, wheel encoders). It abstracts the low-level
# communication protocol (e.g., serial, CAN bus) from the rest of the
# ROS 2 system.
#
# Key Responsibilities:
# 1. Subscribes to `/cmd_vel` to receive velocity commands from the
#    navigation stack.
# 2. Sends these commands to the robot's motor controllers.
# 3. Reads sensor data (IMU, wheel velocities) from the hardware.
# 4. Publishes this data onto standard ROS 2 topics (`/imu/data`,
#    `/wheel_odometry/twist`).
# 5. Includes a `use_simulation` mode for development and testing
#    without a physical robot.
# ===================================================================

import rclpy
from rclpy.node import Node
import numpy as np
import time

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

# Placeholder for a hardware communication library
# In a real scenario, you might import something like:
# import serial

class HardwareInterfaceNode(Node):
    """
    ROS 2 Node for interfacing with the robot's physical hardware.
    """
    def __init__(self):
        super().__init__('hardware_interface_node')
        self.get_logger().info('Initializing Hardware Interface Node...')

        # --- 1. Declare and Get ROS 2 Parameters ---
        self.declare_parameter('use_simulation', True)
        self.declare_parameter('update_frequency', 50.0)
        self.declare_parameter('serial_port', '/dev/ttyACM0') # Example for real hardware
        self.declare_parameter('baud_rate', 115200)       # Example for real hardware

        self.use_simulation = self.get_parameter('use_simulation').value
        self.update_freq = self.get_parameter('update_frequency').value
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value

        # --- 2. Initialize Hardware Connection (or Simulation) ---
        self.robot_connection = None
        if self.use_simulation:
            self.get_logger().info('Running in SIMULATION mode.')
            # Simulation state variables
            self.sim_linear_vel = 0.0
            self.sim_angular_vel = 0.0
        else:
            self.get_logger().info(f'Attempting to connect to real hardware on {self.serial_port}...')
            try:
                # self.robot_connection = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
                # self.get_logger().info('Successfully connected to hardware.')
                # The line above is what you would uncomment and use with a real serial device.
                self.get_logger().warn('Real hardware connection is commented out. Running with zero-output.')
                pass # This is a placeholder
            except Exception as e:
                self.get_logger().error(f'Failed to connect to hardware: {e}. Shutting down.')
                # In a real application, you might want to retry or handle this more gracefully.
                rclpy.shutdown()
                return

        # --- 3. Setup ROS 2 Subscribers and Publishers ---
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',  # Standard topic for velocity commands
            self.cmd_vel_callback,
            10)

        self.imu_publisher = self.create_publisher(Imu, '/imu/data', 10)
        self.twist_publisher = self.create_publisher(TwistWithCovarianceStamped, '/wheel_odometry/twist', 10)

        # --- 4. Main Processing Loop ---
        self.timer = self.create_timer(1.0 / self.update_freq, self.update_loop)
        self.get_logger().info(f'Node initialized. Update loop running at {self.update_freq} Hz.')

    def cmd_vel_callback(self, msg: Twist):
        """Receives velocity commands and sends them to hardware or simulation."""
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        if self.use_simulation:
            self.sim_linear_vel = linear_x
            self.sim_angular_vel = angular_z
        else:
            self._send_to_robot_hardware(linear_x, angular_z)

    def update_loop(self):
        """Main loop for reading from hardware and publishing sensor data."""
        if self.use_simulation:
            # Generate fake data based on the last command
            linear_vel, angular_vel, imu_data = self._simulate_robot_sensors()
        else:
            # Read real data from the hardware
            linear_vel, angular_vel, imu_data = self._read_from_robot_hardware()

        current_time = self.get_clock().now().to_msg()

        # Publish IMU data
        if imu_data:
            imu_msg = Imu()
            imu_msg.header.stamp = current_time
            imu_msg.header.frame_id = 'imu_link' # A standard frame name
            imu_msg.angular_velocity.z = imu_data['angular_velocity_z']
            imu_msg.linear_acceleration.x = imu_data['linear_acceleration_x']
            # ... populate other IMU fields (orientation, covariances)
            self.imu_publisher.publish(imu_msg)

        # Publish wheel odometry twist
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = current_time
        twist_msg.header.frame_id = 'base_link'
        twist_msg.twist.twist.linear.x = linear_vel
        twist_msg.twist.twist.angular.z = angular_vel
        # ... populate covariances
        self.twist_publisher.publish(twist_msg)

    def _simulate_robot_sensors(self):
        """Generates realistic-looking sensor data for simulation."""
        # Add some noise to make it more realistic
        noise_factor_linear = 0.05
        noise_factor_angular = 0.02
        
        # Simulated velocity is the command + noise
        sim_lv_noisy = self.sim_linear_vel + np.random.normal(0, noise_factor_linear)
        sim_av_noisy = self.sim_angular_vel + np.random.normal(0, noise_factor_angular)

        # Simulated IMU data reflects the commanded motion
        imu_data = {
            'angular_velocity_z': sim_av_noisy,
            'linear_acceleration_x': self.sim_linear_vel * self.update_freq # A very rough approximation of acceleration
        }
        
        return sim_lv_noisy, sim_av_noisy, imu_data

    def _read_from_robot_hardware(self):
        """
        [IMPLEMENTATION REQUIRED]
        Placeholder for reading data from the actual robot hardware.
        """
        if self.robot_connection:
            try:
                # For a serial device, you might do something like this:
                # line = self.robot_connection.readline().decode('utf-8').strip()
                # if line:
                #     parts = line.split(',')
                #     linear_vel = float(parts[0])
                #     angular_vel = float(parts[1])
                #     ... parse other sensor data ...
                #     return linear_vel, angular_vel, imu_data
                pass
            except Exception as e:
                self.get_logger().error(f"Error reading from hardware: {e}")
        
        # Return zeros if no data is available
        return 0.0, 0.0, None

    def _send_to_robot_hardware(self, linear_x, angular_z):
        """
        [IMPLEMENTATION REQUIRED]
        Placeholder for sending commands to the actual robot hardware.
        """
        if self.robot_connection:
            try:
                # For a serial device, you might format a command string:
                # command = f"v,{linear_x},{angular_z}\n"
                # self.robot_connection.write(command.encode('utf-8'))
                pass
            except Exception as e:
                self.get_logger().error(f"Error writing to hardware: {e}")

    def on_shutdown(self):
        """Called when the node is shutting down."""
        self.get_logger().info("Hardware interface shutting down.")
        if self.robot_connection:
            # self.robot_connection.close()
            pass


def main(args=None):
    rclpy.init(args=args)
    node = HardwareInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
