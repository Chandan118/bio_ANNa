from glob import glob
from setuptools import find_packages, setup

package_name = 'bio_anna'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Chandan Sheikder',
    maintainer_email='chandansheikder@example.com',
    description='Autonomous Neuromorphic Navigation Architecture Python nodes and utilities.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hardware_interface_node = bio_anna.nodes.hardware_interface_node:main',
            'antbot_snn_odometry_node = bio_anna.nodes.antbot_snn_odometry_node:main',
            'gridcore_snn_mapper_node = bio_anna.nodes.gridcore_snn_mapper_node:main',
            'navigation_control_node = bio_anna.nodes.navigation_control_node:main',
            'data_collection_node = bio_anna.nodes.data_collection_node:main',
            'run_full_experiment = bio_anna.scripts.run_full_experiment:main',
        ],
    },
)
