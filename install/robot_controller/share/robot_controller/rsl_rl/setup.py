from setuptools import setup
import os
from glob import glob


package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rsl_rl'), glob('*.py'))
    ],
    install_requires=['setuptools', 'torch', 'numpy', 'rclpy', 'sensor_msgs', 
                      'trajectory_msgs'],
    zip_safe=True,
    maintainer='kjh',
    maintainer_email='jh12351002@gmail.com',
    description='ROS2 node for controlling a quadruped robot using a pre-trained reinforcement learning policy',
    license='TODO: License declaration',  # Replace TODO with your desired license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_wtw_policy = robot_controller.load_wtw_policy:main',
            'run_dreamwaq_policy = robot_controller.load_dreamwaq_policy:main',
        ],
    },
)
