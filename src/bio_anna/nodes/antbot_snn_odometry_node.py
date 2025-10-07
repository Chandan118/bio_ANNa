# ===================================================================
# bio ANNa - AntBot SNN Odometry Node
#
# Author: chandan sheikder
# Date: 7 oct 2025
#
# Description:
# This ROS 2 node implements the desert ant-inspired path integration
# algorithm using a Spiking Neural Network (SNN) simulated to be
# running on an Intel Loihi 2 chip. It subscribes to inertial (IMU)
# and heading (Polarization Compass) sensor data, encodes this data
# into spikes, processes it through the neuromorphic interface, decodes
# the output spikes into motion, and publishes the resulting odometry
# as both a nav_msgs/Odometry message and a tf2 transform.
# ===================================================================

import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros

from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

# Import the custom utility for interfacing with the (simulated) Loihi chip
from bio_anna.utils.loihi_interface import Loihi2Interface
from bio_anna.utils.math_utils import quaternion_from_yaw

class AntBotSNNNode(Node):
    """
    ROS 2 Node for AntBot-inspired Path Integration.
    """
    def __init__(self):
        super().__init__('antbot_snn_odometry_node')
        self.get_logger().info('Initializing AntBot SNN Odometry Node...')

        # --- 1. Declare and Get ROS 2 Parameters ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('odom_frame_id', 'odom'),
                ('base_frame_id', 'base_link'),
                ('update_frequency', 20.0),
                ('snn.num_neurons', 128),
                ('snn.encoding_gain', 15.0),
                ('snn.decoding_scaler_linear', 0.05),
                ('snn.decoding_scaler_angular', 0.02)
            ])
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.base_frame = self.get_parameter('base_frame_id').value
        self.update_freq = self.get_parameter('update_frequency').value
        self.num_neurons = self.get_parameter('snn.num_neurons').value
        self.encoding_gain = self.get_parameter('snn.encoding_gain').value
        self.linear_scaler = self.get_parameter('snn.decoding_scaler_linear').value
        self.angular_scaler = self.get_parameter('snn.decoding_scaler_angular').value

        # --- 2. Initialize the Neuromorphic Interface ---
        # In a real system, this path would point to a pre-compiled SNN configuration file.
        loihi_config_path = "loihi_configs/antbot_path_integration_net.net"
        self.loihi_antbot_net = Loihi2Interface(
            network_config_path=loihi_config_path,
            num_neurons=self.num_neurons
        )
        self.get_logger().info(f'Loihi interface initialized for network: {loihi_config_path}')

        # --- 3. Initialize State Variables ---
        self.current_pose = np.zeros(3)  # [x, y, theta]
        self.current_velocity = np.zeros(3) # [vx, vy, v_theta]
        self.last_update_time = self.get_clock().now()
        self.latest_imu_wz = 0.0
        self.latest_linear_velocity_x = 0.0 # Assuming forward velocity from wheel encoders or other source

        # --- 4. Setup ROS 2 Subscribers, Publishers, and TF Broadcaster ---
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)

        # In a real robot, this would come from wheel odometry.
        # Here we subscribe to a simple topic for demonstration.
        self.velocity_subscriber = self.create_subscription(
            TwistWithCovarianceStamped,
            '/wheel_odometry/twist', # A more realistic topic
            self.velocity_callback,
            10)

        self.odom_publisher = self.create_publisher(Odometry, '/odom/antbot_snn', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- 5. Main Processing Loop ---
        self.timer = self.create_timer(1.0 / self.update_freq, self.update_loop)
        self.get_logger().info(f'Node initialized. Update loop running at {self.update_freq} Hz.')

    def imu_callback(self, msg: Imu):
        """Stores the latest angular velocity around the z-axis from the IMU."""
        self.latest_imu_wz = msg.angular_velocity.z

    def velocity_callback(self, msg: TwistWithCovarianceStamped):
        """Stores the latest linear velocity from wheel odometry."""
        self.latest_linear_velocity_x = msg.twist.twist.linear.x
        
    def encode_inputs_to_spikes(self, linear_vel_x, angular_vel_z):
        """
        Encodes continuous sensor values into a vector of spike rates.
        This uses Gaussian Receptive Fields (also called basis functions), a common
        and powerful encoding method. Each neuron fires most for a specific input value.
        """
        # Define the preferred values for each neuron
        neuron_indices = np.arange(self.num_neurons)
        # First half of neurons encode linear velocity, second half angular
        preferred_linear_vels = np.linspace(-1.0, 1.0, self.num_neurons // 2)
        preferred_angular_vels = np.linspace(-np.pi/2, np.pi/2, self.num_neurons // 2)
        
        # Calculate spike rates for linear velocity neurons
        linear_distances = np.abs(preferred_linear_vels - linear_vel_x)
        linear_spikes = np.exp(-self.encoding_gain * linear_distances**2)
        
        # Calculate spike rates for angular velocity neurons
        angular_distances = np.abs(preferred_angular_vels - angular_vel_z)
        angular_spikes = np.exp(-self.encoding_gain * angular_distances**2)
        
        return np.concatenate([linear_spikes, angular_spikes])

    def decode_spikes_to_motion(self, output_spikes):
        """
        Decodes the output spike vector from the SNN into motion commands.
        This assumes the SNN has learned to map input spikes to output populations
        representing linear and angular change. We use a Center of Mass decoder.
        """
        # Assume first half of output neurons represent linear motion, second half angular
        num_output_neurons = len(output_spikes)
        linear_pop_spikes = output_spikes[:num_output_neurons//2]
        angular_pop_spikes = output_spikes[num_output_neurons//2:]

        # Prevent division by zero if there's no activity
        if np.sum(linear_pop_spikes) > 1e-6:
            # Center of mass calculation
            indices = np.arange(len(linear_pop_spikes))
            center_of_mass_linear = np.sum(indices * linear_pop_spikes) / np.sum(linear_pop_spikes)
            # Map center of mass back to a physical value
            delta_linear = (center_of_mass_linear - len(linear_pop_spikes)/2) * self.linear_scaler
        else:
            delta_linear = 0.0

        if np.sum(angular_pop_spikes) > 1e-6:
            indices = np.arange(len(angular_pop_spikes))
            center_of_mass_angular = np.sum(indices * angular_pop_spikes) / np.sum(angular_pop_spikes)
            delta_angular = (center_of_mass_angular - len(angular_pop_spikes)/2) * self.angular_scaler
        else:
            delta_angular = 0.0
            
        return delta_linear, delta_angular # Returns delta_x and delta_theta

    def update_loop(self):
        """The main computational loop, executed at a fixed frequency."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        if dt == 0:
            return

        # 1. Encode sensor data into spikes
        input_spikes = self.encode_inputs_to_spikes(self.latest_linear_velocity_x, self.latest_imu_wz)
        
        # 2. Process spikes through the neuromorphic hardware interface
        output_spikes = self.loihi_antbot_net.run_step(input_spikes)

        # 3. Decode output spikes into motion changes (dx, dtheta)
        delta_linear, delta_angular = self.decode_spikes_to_motion(output_spikes)

        # 4. Integrate pose (update the robot's estimated position and orientation)
        # Calculate change in x and y in the robot's frame, then rotate to the odom frame
        dx = delta_linear * dt * np.cos(self.current_pose[2])
        dy = delta_linear * dt * np.sin(self.current_pose[2])
        dtheta = delta_angular * dt

        self.current_pose[0] += dx
        self.current_pose[1] += dy
        self.current_pose[2] += dtheta
        
        # Normalize angle to be within -pi to pi
        self.current_pose[2] = np.arctan2(np.sin(self.current_pose[2]), np.cos(self.current_pose[2]))

        # Store velocity for the odometry message
        self.current_velocity = [delta_linear, 0.0, delta_angular] # [vx, vy, v_theta]

        # 5. Publish Odometry message and TF transform
        self.publish_odometry(current_time)
        self.publish_tf(current_time)

        self.last_update_time = current_time

    def publish_odometry(self, timestamp):
        """Creates and publishes a nav_msgs/Odometry message."""
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        # Set the position
        odom_msg.pose.pose.position.x = self.current_pose[0]
        odom_msg.pose.pose.position.y = self.current_pose[1]
        
        # Convert yaw angle to quaternion
        qx, qy, qz, qw = quaternion_from_yaw(self.current_pose[2])
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        # Set the velocity
        odom_msg.twist.twist.linear.x = self.current_velocity[0]
        odom_msg.twist.twist.angular.z = self.current_velocity[2]

        # Note: Covariances are not populated in this example but would be
        # derived from the SNN's output variance in a full implementation.

        self.odom_publisher.publish(odom_msg)

    def publish_tf(self, timestamp):
        """Creates and publishes the transform from odom_frame to base_frame."""
        t = TransformStamped()
        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = self.current_pose[0]
        t.transform.translation.y = self.current_pose[1]
        t.transform.translation.z = 0.0 # Assuming 2D motion

        qx, qy, qz, qw = quaternion_from_yaw(self.current_pose[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AntBotSNNNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
