# Contributing to Bio ANNa

Thank you for your interest in contributing to Bio ANNa (Autonomous Neuromorphic Navigation Architecture)! We welcome contributions that improve our neuromorphic navigation stack, documentation, or experimental utilities.

## Getting Started

1.  **Fork the Repository**: Create your own copy of the repository on GitHub.
2.  **Clone the Fork**: Clone your fork to your local machine.
3.  **Set Up the Environment**:
    - Ensure you have [ROS 2 Galactic](https://docs.ros.org/en/galactic/Installation.html) installed.
    - Install Python dependencies: `pip install -r requirements.txt`.
    - Build the workspace: `colcon build --symlink-install`.

## Development Guidelines

- **Branching Strategy**: Use descriptive branch names (e.g., `feature/snn-optimization` or `fix/imu-driver`).
- **Coding Standards**:
    - Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
    - Use clear and descriptive variable names.
    - Document new functions and classes with docstrings.
- **ROS 2 Integration**:
    - Ensure all new nodes are added to `setup.py` entry points.
    - Update `package.xml` if you add new ROS dependencies.

## Data Collection Requirements

If you are working on the `data_collection_node.py` or adding new sensor data:
- Ensure data is saved in a structured format (CSV/HDF5) within the `datasets/` directory.
- Verify that the node handles high-frequency sensor streams without dropping messages.

## Submitting Changes

1.  **Commit Your Changes**: Write clear, concise commit messages.
2.  **Push to Your Fork**: Push your branch to your GitHub repository.
3.  **Submit a Pull Request (PR)**:
    - Target the `main` branch of the original repository.
    - Provide a detailed description of your changes and why they are necessary.
    - Link any relevant issues.

## Testing

Before submitting a PR, please ensure:
- Your code passes syntax checks: `python3 -m compileall src/bio_anna`.
- The package builds successfully: `colcon build`.
- You have tested your changes in a simulated environment or with provided datasets.

---
By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
