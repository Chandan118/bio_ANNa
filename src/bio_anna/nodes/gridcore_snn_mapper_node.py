# ===================================================================
# Bio ANNa - GridCore SNN Mapper Node
#
# Author: chandan sheikder
# Date: 7 oct 2025
#
# Description:
# This ROS 2 node implements the rodent-inspired cognitive mapping and
# re-localization system. It simulates the function of the hippocampus
# (place cells) and entorhinal cortex (grid cells) using a "GridCore" SNN.
#
# The node performs the following steps:
# 1. Subscribes to 2D Laser Scan (LiDAR) data.
# 2. Extracts salient landmarks (corners, edges) from the scan.
# 3. Encodes landmark information into spikes.
# 4. Processes spikes through the GridCore SNN (via the Loihi interface)
#    to get a unique, stable neural representation of the landmark.
# 5. Compares this representation against an internal "cognitive map".
# 6. If a landmark is recognized, it calculates a pose correction and
#    publishes it with high confidence to aid the main navigation filter.
# 7. If a landmark is new, it adds it to the cognitive map.
# ===================================================================

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from bio_anna.utils.loihi_interface import Loihi2Interface
from bio_anna.utils.math_utils import quaternion_from_yaw, yaw_from_quaternion

class GridCoreSNNNode(Node):
    """
    ROS 2 Node for rodent-inspired cognitive mapping.
    """
    def __init__(self):
        super().__init__('gridcore_snn_mapper_node')
        self.get_logger().info('Initializing GridCore SNN Mapper Node...')

        # --- 1. Declare and Get ROS 2 Parameters ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('odom_frame_id', 'odom'),
                ('base_frame_id', 'base_link'),
                ('update_frequency', 2.0), # Mapping can be slower than odometry
                ('landmark_detection_threshold', 0.5), # Meters, for corner detection
                ('relocalization_confidence', 0.1), # Low covariance for high confidence
                ('new_landmark_confidence', 99.0), # High covariance for low confidence
                ('map_storage_resolution', 0.25), # Meters, for discretizing landmark signatures
                ('snn.num_neurons', 128)
            ])
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.base_frame = self.get_parameter('base_frame_id').value
        self.update_freq = self.get_parameter('update_frequency').value
        self.landmark_thresh = self.get_parameter('landmark_detection_threshold').value
        self.reloc_cov = self.get_parameter('relocalization_confidence').value
        self.new_lm_cov = self.get_parameter('new_landmark_confidence').value
        self.map_resolution = self.get_parameter('map_storage_resolution').value
        self.num_neurons = self.get_parameter('snn.num_neurons').value

        # --- 2. Initialize the Neuromorphic Interface ---
        loihi_config_path = "loihi_configs/gridcore_cognitive_map_net.net"
        self.loihi_gridcore_net = Loihi2Interface(
            network_config_path=loihi_config_path,
            num_neurons=self.num_neurons
        )
        self.get_logger().info(f'Loihi interface initialized for network: {loihi_config_path}')

        # --- 3. Initialize State Variables ---
        self.latest_scan = None
        self.current_odom = None
        # The Cognitive Map: { landmark_signature (tuple): global_pose (np.array) }
        self.cognitive_map = {}

        # --- 4. Setup ROS 2 Subscribers and Publishers ---
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # Subscribes to the fused odometry to know the robot's current best guess
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom/fused', # Assuming a fused odometry topic from the main navigator
            self.odom_callback,
            10)

        self.correction_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/pose_correction/gridcore',
            10)

        # --- 5. Main Processing Loop ---
        self.timer = self.create_timer(1.0 / self.update_freq, self.update_loop)
        self.get_logger().info(f'Node initialized. Update loop running at {self.update_freq} Hz.')

    def scan_callback(self, msg: LaserScan):
        """Stores the latest laser scan data."""
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        """Stores the latest odometry data."""
        self.current_odom = msg

    def extract_landmarks_from_scan(self, scan: LaserScan):
        """
        A simple but effective landmark detector. It finds corners by looking for
        large, sudden changes in range between consecutive laser points.
        """
        landmarks = []
        ranges = np.array(scan.ranges)
        # Invalidate infinite ranges
        ranges[np.isinf(ranges)] = scan.range_max

        # We look at triplets of points (p_prev, p_curr, p_next)
        for i in range(1, len(ranges) - 1):
            # Check for sudden jumps in range, indicating an edge
            is_discontinuous = abs(ranges[i-1] - ranges[i]) > self.landmark_thresh or \
                               abs(ranges[i+1] - ranges[i]) > self.landmark_thresh

            if is_discontinuous and ranges[i] < scan.range_max:
                angle = scan.angle_min + i * scan.angle_increment
                x = ranges[i] * np.cos(angle)
                y = ranges[i] * np.sin(angle)
                landmarks.append(np.array([x, y]))

        return landmarks

    def generate_landmark_signature(self, landmark_local_pos):
        """
        Creates a unique, discrete signature for a landmark based on its position.
        This simulates the invariant representation a place cell might provide.
        """
        # Discretize the position to make it robust to small noises
        x_discrete = int(landmark_local_pos[0] / self.map_resolution)
        y_discrete = int(landmark_local_pos[1] / self.map_resolution)
        
        # NOTE: A more complex signature would involve SNN processing.
        # Here we simulate the SNN's output as a discrete identifier.
        # Spikes would be encoded from landmark_local_pos, fed to Loihi,
        # and the output would be a stable activity pattern that we can hash.
        # For demonstration, we use this direct hashing.
        return (x_discrete, y_discrete)

    def update_loop(self):
        """The main computational loop for mapping and re-localization."""
        if self.latest_scan is None or self.current_odom is None:
            self.get_logger().warn('Waiting for sensor and odometry data...', throttle_duration_sec=5)
            return

        landmarks = self.extract_landmarks_from_scan(self.latest_scan)
        if not landmarks:
            return

        # Get current robot pose from odometry
        current_pos = self.current_odom.pose.pose.position
        current_orient = self.current_odom.pose.pose.orientation
        current_yaw = yaw_from_quaternion(current_orient.x, current_orient.y, current_orient.z, current_orient.w)
        current_robot_pose = np.array([current_pos.x, current_pos.y, current_yaw])

        num_recalcs = 0
        for lm_local in landmarks:
            # Generate a signature for the landmark we just saw
            signature = self.generate_landmark_signature(lm_local)

            if signature in self.cognitive_map:
                # --- RE-LOCALIZATION ---
                # We have seen this landmark before!
                num_recalcs += 1

                # Get the global pose of the landmark from our map
                lm_global_pose = self.cognitive_map[signature]

                # From the landmark's global pose and its current local pose,
                # we can calculate a corrected global pose for the robot.
                # This is a coordinate frame transformation.
                cos_th, sin_th = np.cos(lm_global_pose[2]), np.sin(lm_global_pose[2])
                
                # Transform local landmark position to global frame based on landmark's known orientation
                # This is complex, so we simplify: assume orientation is also stored or derived.
                # Here, we calculate where the robot *must be* to see the landmark at `lm_local`.
                corrected_robot_x = lm_global_pose[0] - (lm_local[0] * np.cos(current_yaw) - lm_local[1] * np.sin(current_yaw))
                corrected_robot_y = lm_global_pose[1] - (lm_local[0] * np.sin(current_yaw) + lm_local[1] * np.cos(current_yaw))

                # For now, we trust the odom's yaw, but a more advanced system
                # would correct yaw based on seeing multiple landmarks.
                corrected_pose = np.array([corrected_robot_x, corrected_robot_y, current_yaw])
                self.publish_correction(corrected_pose, self.reloc_cov)
            else:
                # --- NEW LANDMARK DISCOVERY ---
                # Transform the landmark's local position to the global frame using current odometry
                lm_global_x = current_robot_pose[0] + (lm_local[0] * np.cos(current_yaw) - lm_local[1] * np.sin(current_yaw))
                lm_global_y = current_robot_pose[1] + (lm_local[0] * np.sin(current_yaw) + lm_local[1] * np.cos(current_yaw))
                
                # Store the new landmark in our map
                # The stored "pose" of the landmark can just be its position for now.
                self.cognitive_map[signature] = np.array([lm_global_x, lm_global_y, current_yaw])
        
        if num_recalcs > 0:
            self.get_logger().info(f'Re-localized based on {num_recalcs} recognized landmarks.')
        else:
             self.get_logger().info(f'Explored new area. Added {len(landmarks)} landmarks to cognitive map (total: {len(self.cognitive_map)}).')


    def publish_correction(self, pose, covariance):
        """Publishes a pose correction with a given confidence."""
        correction_msg = PoseWithCovarianceStamped()
        correction_msg.header.stamp = self.get_clock().now().to_msg()
        correction_msg.header.frame_id = self.odom_frame
        
        correction_msg.pose.pose.position.x = pose[0]
        correction_msg.pose.pose.position.y = pose[1]
        
        qx, qy, qz, qw = quaternion_from_yaw(pose[2])
        correction_msg.pose.pose.orientation.x = qx
        correction_msg.pose.pose.orientation.y = qy
        correction_msg.pose.pose.orientation.z = qz
        correction_msg.pose.pose.orientation.w = qw

        # A 6x6 covariance matrix. Low values on the diagonal mean high confidence.
        correction_msg.pose.covariance[0] = covariance  # x variance
        correction_msg.pose.covariance[7] = covariance  # y variance
        correction_msg.pose.covariance[35] = covariance * 2 # Slightly less confidence in yaw

        self.correction_publisher.publish(correction_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GridCoreSNNNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
