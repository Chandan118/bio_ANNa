# ===================================================================
# Stage 1: Base Image and Core Dependencies
# ===================================================================
# Use the official ROS 2 Galactic base image. This includes all the core ROS libraries
# and a minimal Ubuntu 20.04 (Focal Fossa) environment.
FROM osrf/ros:galactic-desktop

# Set shell to bash for all subsequent RUN commands
SHELL ["/bin/bash", "-c"]

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ===================================================================
# Stage 2: System and Python Dependencies Installation
# ===================================================================
# Update package lists and install system-level dependencies required for the project.
# - python3-pip: For installing Python packages.
# - python3-colcon-common-extensions: Standard tool for building ROS 2 workspaces.
# - git: For version control.
# - other tools: Common utilities for development.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-scipy \
    python3-pandas \
    python3-matplotlib \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python package dependencies using pip.
# This is done in a separate step to leverage Docker's layer caching.
# If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ===================================================================
# Stage 3: ROS 2 Workspace Setup and Project Build
# ===================================================================
# Create a ROS 2 workspace directory.
RUN mkdir -p /root/ros2_ws/src
WORKDIR /root/ros2_ws

# Copy the entire project source code into the workspace's 'src' directory.
# The `.` in the `COPY` command refers to the Docker build context (the project's root folder).
COPY . /root/ros2_ws/src/bio_anna

# Source the ROS 2 environment and build the workspace using colcon.
# This command compiles all the Python and C++ nodes in your project.
RUN /bin/bash -lc "source /opt/ros/galactic/setup.bash && colcon build --symlink-install"

# ===================================================================
# Stage 4: Final Configuration and Entrypoint
# ===================================================================
# Set up a custom entrypoint script. This script will be executed every time
# the container starts. It automatically sources the ROS 2 installation and
# the local workspace, making the environment ready to use immediately.
RUN printf '#!/bin/bash\nset -e\nsource /opt/ros/galactic/setup.bash\nsource /root/ros2_ws/install/setup.bash\nexec "$@"\n' > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

# Set the entrypoint for the container.
ENTRYPOINT ["/ros_entrypoint.sh"]

# The default command to execute when the container starts.
# This will drop you into an interactive bash shell inside the container.
CMD ["bash"]
