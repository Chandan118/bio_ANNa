# ===================================================================
# Bio ANNa - Navigation Control Node
#
# Author: chndan sheikder
# Date: 07 oct 2025
#
# Description:
# This is the central decision-making node for the ANNa system. It acts
# as the "brain," performing three critical functions:
# 1. State Estimation: It fuses the continuous odometry from the AntBot SNN
#    with the absolute pose corrections from the GridCore SNN using a
#    Bayesian Fusion Engine (Kalman Filter).
# 2. Path Control: It subscribes to a goal pose and implements a simple
#    proportional controller to navigate the robot towards that goal.
# 3. State Publishing: It publishes the final, fused odometry and the
#    authoritative tf transform for the rest of the ROS 2 system to use.
# ===================================================================

import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
from bio_anna.utils.math_utils import quaternion_from_yaw, yaw_from_quaternion

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry

from bio_anna.utils.bayesian_fusion_engine import BayesianFusionEngine

class NavigationControlNode(Node):
    """
    The central control and state estimation node for the robot.
    """
    def __init__(self):
        super().__init__('navigation_control_node')
        self.get_logger().info('Initializing Navigation Control Node...')

        # --- 1. Declare and Get ROS 2 Parameters ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('odom_frame_id', 'odom'),
                ('base_frame_id', 'base_link'),
                ('control_frequency', 20.0),
                ('goal_tolerance_dist', 0.2), # Meters
                ('goal_tolerance_angle', 0.1), # Radians
                ('max_linear_velocity', 0.5), # m/s
                ('max_angular_velocity', 1.0), # rad/s
                ('p_gain_linear', 0.8), # Proportional gain for linear velocity
                ('p_gain_angular', 1.5) # Proportional gain for angular velocity
            ])

        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.base_frame = self.get_parameter('base_frame_id').value
        self.control_freq = self.get_parameter('control_frequency').value
        self.goal_dist_tol = self.get_parameter('goal_tolerance_dist').value
        self.goal_angle_tol = self.get_parameter('goal_tolerance_angle').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.p_gain_linear = self.get_parameter('p_gain_linear').value
        self.p_gain_angular = self.get_parameter('p_gain_angular').value

        # --- 2. Initialize Core Components ---
        self.fusion_engine = BayesianFusionEngine()
        self.get_logger().info('Bayesian Fusion Engine initialized.')

        # --- 3. Initialize State Variables ---
        self.latest_antbot_odom = None
        self.latest_gridcore_correction = None
        self.current_goal = None
        self.goal_reached = True
        self._last_control_time = self.get_clock().now()

        # --- 4. Setup ROS 2 Subscribers, Publishers, and TF Broadcaster ---
        self.antbot_odom_sub = self.create_subscription(
            Odometry, '/odom/antbot_snn', self._antbot_odom_callback, 10)

        self.gridcore_correction_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/pose_correction/gridcore', self._gridcore_correction_callback, 10)

        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.fused_odom_pub = self.create_publisher(Odometry, '/odom/fused', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- 5. Main Control Loop ---
        self.timer = self.create_timer(1.0 / self.control_freq, self._control_loop)
        self.get_logger().info(f'Node initialized. Control loop running at {self.control_freq} Hz.')

    def _antbot_odom_callback(self, msg: Odometry):
        self.latest_antbot_odom = msg

    def _gridcore_correction_callback(self, msg: PoseWithCovarianceStamped):
        self.latest_gridcore_correction = msg

    def _goal_callback(self, msg: PoseStamped):
        """Receives a new navigation goal."""
        if msg.header.frame_id != self.odom_frame:
            self.get_logger().error(f"Goal must be in '{self.odom_frame}' frame, but received in '{msg.header.frame_id}'")
            return
        
        self.current_goal = msg
        self.goal_reached = False
        self.get_logger().info(f"New goal received: X={msg.pose.position.x:.2f}, Y={msg.pose.position.y:.2f}")

    def _control_loop(self):
        """The main computational loop for state estimation and control."""
        current_time = self.get_clock().now()
        dt = (current_time - self._last_control_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 1.0 / self.control_freq
        self._last_control_time = current_time

        # --- State Estimation Step ---
        # Predict forward based on motion model
        self.fusion_engine.predict(dt)

        # Update with latest sensor data if available
        if self.latest_antbot_odom is not None:
            velocity_measurement, velocity_cov = self._extract_antbot_measurement(self.latest_antbot_odom)
            self.fusion_engine.update_with_antbot_odometry(velocity_measurement, velocity_cov)
            self.latest_antbot_odom = None

        if self.latest_gridcore_correction is not None:
            pose_measurement, pose_cov = self._extract_gridcore_measurement(self.latest_gridcore_correction)
            self.fusion_engine.update_with_gridcore_correction(pose_measurement, pose_cov)
            self.latest_gridcore_correction = None
        
        # Get the fused pose
        fused_pose = self.fusion_engine.get_current_pose() # [x, y, theta]
        # Get fused velocity for publishing
        # fused_velocity = self.fusion_engine.get_current_velocity()
        
        # Publish the fused odometry and TF
        self._publish_fused_state(fused_pose)

        # --- Control Step ---
        if self.current_goal is None or self.goal_reached:
            # If no goal or goal is reached, send a zero velocity command
            self._publish_cmd_vel(0.0, 0.0)
            return

        # Calculate error to goal
        error_x = self.current_goal.pose.position.x - fused_pose[0]
        error_y = self.current_goal.pose.position.y - fused_pose[1]
        distance_to_goal = np.sqrt(error_x**2 + error_y**2)

        if distance_to_goal < self.goal_dist_tol:
            self.get_logger().info("Goal reached!")
            self.goal_reached = True
            self.current_goal = None
            self._publish_cmd_vel(0.0, 0.0)
            return

        # Calculate angle to goal
        angle_to_goal = np.arctan2(error_y, error_x)
        angle_error = angle_to_goal - fused_pose[2]
        # Normalize angle error to be within -pi to pi
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        # --- Proportional Controller Logic ---
        # If the angle error is too large, prioritize turning
        if abs(angle_error) > self.goal_angle_tol:
            linear_vel = 0.0
            angular_vel = self.p_gain_angular * angle_error
        else:
            # Otherwise, drive forward while correcting heading
            linear_vel = self.p_gain_linear * distance_to_goal
            angular_vel = self.p_gain_angular * angle_error

        # --- Velocity Saturation (Clamping) ---
        # Ensure commands do not exceed robot's physical limits
        linear_vel = np.clip(linear_vel, -self.max_linear_vel, self.max_linear_vel)
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        # Publish the final command
        self._publish_cmd_vel(linear_vel, angular_vel)

    def _extract_antbot_measurement(self, odom_msg: Odometry):
        """Extracts velocity measurement and covariance from AntBot odometry."""
        v = float(odom_msg.twist.twist.linear.x)
        w = float(odom_msg.twist.twist.angular.z)

        measurement_cov = None
        if len(odom_msg.twist.covariance) == 36:
            cov = np.array(odom_msg.twist.covariance, dtype=float).reshape(6, 6)
            relevant_indices = np.ix_([0, 5], [0, 5])
            measurement_cov = cov[relevant_indices]
            if np.allclose(measurement_cov, 0.0):
                measurement_cov = None

        return np.array([v, w], dtype=float), measurement_cov

    def _extract_gridcore_measurement(self, correction_msg: PoseWithCovarianceStamped):
        """Extracts pose correction measurement and covariance."""
        position = correction_msg.pose.pose.position
        orientation = correction_msg.pose.pose.orientation
        yaw = yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)

        measurement_cov = None
        if len(correction_msg.pose.covariance) == 36:
            cov = np.array(correction_msg.pose.covariance, dtype=float).reshape(6, 6)
            relevant_indices = np.ix_([0, 1, 5], [0, 1, 5])
            measurement_cov = cov[relevant_indices]
            if np.allclose(measurement_cov, 0.0):
                measurement_cov = None

        return np.array([position.x, position.y, yaw], dtype=float), measurement_cov

    def _publish_fused_state(self, pose):
        """Publishes the fused odometry and its corresponding TF transform."""
        timestamp = self.get_clock().now().to_msg()

        # Publish Odometry Message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose.position.x = pose[0]
        odom_msg.pose.pose.position.y = pose[1]
        qx, qy, qz, qw = quaternion_from_yaw(pose[2])
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        # Populate covariance from fusion engine
        # odom_msg.pose.covariance = self.fusion_engine.get_covariance_matrix()
        self.fused_odom_pub.publish(odom_msg)

        # Publish TF Transform
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)
        
    def _publish_cmd_vel(self, linear, angular):
        """Creates and publishes a Twist message."""
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear
        cmd_vel_msg.angular.z = angular
        self.cmd_vel_pub.publish(cmd_vel_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = NavigationControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
