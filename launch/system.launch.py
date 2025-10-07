# ===================================================================
# bio ANNa - Main System Launch File
#
# This file orchestrates the launch of all essential nodes for the
# Autonomous Neuromorphic Navigation Architecture. It handles the
# loading of parameters, remapping of topics, and starting of each
# component in the correct configuration.
#
# To run:
#   ros2 launch bio_anna system.launch.py
#
# To run with a custom parameter file:
#   ros2 launch bio_anna system.launch.py params_file:=/path/to/your/params.yaml
# ===================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    Generates the launch description for the entire ANNa system.
    This function is the entry point for the ROS 2 launch system.
    """

    # --- 1. Get Project and Parameter File Paths ---
    # Find the share directory of the 'project_anna' package
    bio_anna_pkg_dir = get_package_share_directory('bio_anna')

    # Define the default path to the parameters file
    default_params_file = os.path.join(bio_anna_pkg_dir, 'config', 'navigation_params.yaml')

    # --- 2. Declare Launch Arguments ---
    # These are variables that can be set from the command line when launching the system.
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS2 parameters file to use.'
    )

    # --- 3. Define Node Configurations ---
    # Each 'Node' object describes how to run a single executable (a ROS 2 node).

    # Hardware Interface Node: Connects to real sensors (IMU, Pol-Compass, LiDAR) and robot base.
    hardware_interface_node = Node(
        package='bio_anna',
        executable='hardware_interface_node',
        name='hardware_interface',
        output='screen',
        emulate_tty=True  # Helps with colorized logging
    )

    # AntBot SNN Odometry Node: Processes motion sensor data via the Loihi SNN.
    antbot_snn_odometry_node = Node(
        package='bio_anna',
        executable='antbot_snn_odometry_node',
        name='antbot_snn_odometry',
        output='screen',
        emulate_tty=True,
        remappings=[
            # Remap the default output to a more specific topic name
            ('/odom/neuromorphic', '/odom/antbot_snn')
        ]
    )

    # GridCore SNN Mapper Node: Processes LiDAR data via the Loihi SNN for landmark detection.
    gridcore_snn_mapper_node = Node(
        package='bio_anna',
        executable='gridcore_snn_mapper_node',
        name='gridcore_snn_mapper',
        output='screen',
        emulate_tty=True
    )

    # Navigation Control Node: The "brain" of the robot.
    # It runs the Bayesian Fusion Engine, path planning, and motor control.
    # It is configured using the external YAML parameter file.
    navigation_control_node = Node(
        package='bio_anna',
        executable='navigation_control_node',
        name='navigation_control',
        output='screen',
        emulate_tty=True,
        parameters=[
            # Pass the path to the parameter file
            LaunchConfiguration('params_file'),
            # Override a specific parameter, in this case, use_sim_time
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    # --- 4. Assemble and Return the Launch Description ---
    # The LaunchDescription object is a container for all the actions to be executed.
    return LaunchDescription([
        # Add the declared arguments to the launch description
        declare_use_sim_time_arg,
        declare_params_file_arg,

        # Add the nodes to be launched
        hardware_interface_node,
        antbot_snn_odometry_node,
        gridcore_snn_mapper_node,
        navigation_control_node
    ])
