from setuptools import setup, find_packages
import os
from glob import glob


package_name = 'robot_UI'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name + 'submodule/feather'),glob('icons/*.svg')),
        
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kjh',
    maintainer_email='jh12351002@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_ui = robot_UI.UI_controller:main',
        ],
    },
)
