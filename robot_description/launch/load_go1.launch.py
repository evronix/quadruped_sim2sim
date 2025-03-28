
"""
Launches rviz with the go1 urdf file.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, Shutdown, SetLaunchConfiguration, \
     IncludeLaunchDescription
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, \
     TextSubstitution
from launch.conditions import LaunchConfigurationEquals, IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
     return LaunchDescription([
          DeclareLaunchArgument(name='use_jsp', default_value='gui',
                                choices=['gui', 'jsp', 'none'],
                                description='Choose if joint_state_publisher is launched'),
          DeclareLaunchArgument(name='use_rviz', default_value='true',
                                choices=['true', 'false'],
                                description='Choose if rviz is launched'),
          DeclareLaunchArgument(name='use_nav2_links', default_value='false',
                                choices=['true', 'false'],
                                description='Use Nav2 frames in URDF'),
          DeclareLaunchArgument(name='fixed_frame', default_value='base',
                                description='Fixed frame for RVIZ'),
          DeclareLaunchArgument(name='namespace', default_value='',
                                description=
                                   'Choose a namespace for the launched topics.'),

          SetLaunchConfiguration(name='config_file',
                                 value='config.rviz'),
          SetLaunchConfiguration(name='model',
                                 value=PathJoinSubstitution([FindPackageShare('robot_description'),
                                                             'urdf','go1',
                                                             'go1.urdf.xacro'])),
          
          
          SetLaunchConfiguration(name='rvizconfig',
                                 value=PathJoinSubstitution([FindPackageShare('robot_description'),
                                                             'config',
                                                             'config.rviz'])),

          Node(package='joint_state_publisher',
               executable='joint_state_publisher',
               namespace=LaunchConfiguration('namespace'),
               condition=LaunchConfigurationEquals('use_jsp', 'jsp')),

          Node(package='joint_state_publisher_gui',
               executable='joint_state_publisher_gui',
               namespace=LaunchConfiguration('namespace'),
               condition=LaunchConfigurationEquals('use_jsp', 'gui')),

          Node(package='robot_state_publisher',
               executable='robot_state_publisher',
               parameters=[{
                    'robot_description':
                         ParameterValue(
                              Command([
                                   'xacro ',
                                   LaunchConfiguration('model'),
                                   TextSubstitution(text=' use_nav2_links:='),
                                   LaunchConfiguration('use_nav2_links'),
                              ]),
                              value_type=str
                         ),
                    'frame_prefix': [LaunchConfiguration('namespace'), '/']
               }],
               namespace=LaunchConfiguration('namespace')
          ),

          Node(package='rviz2',
               executable='rviz2',
               name='rviz2',
               arguments=[
                    '-d', LaunchConfiguration('rvizconfig'),
                    '-f', LaunchConfiguration('fixed_frame')
               ],
               condition=IfCondition(LaunchConfiguration('use_rviz')))
    ])