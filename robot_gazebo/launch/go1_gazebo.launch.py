from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_path

pkg_path = get_package_share_path('robot_description')
default_rviz_config_path = pkg_path / 'rviz/urdf.rviz'

world_path = PathJoinSubstitution(
        [FindPackageShare("robot_gazebo"), "worlds", "go1_world.world"]
    )

config_path = PathJoinSubstitution(
        [FindPackageShare("robot_gazebo"), "config", "go1_ros_control.yaml"]
    )

gazebo_path = PathJoinSubstitution(
        [FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"]
    )

robot_description = ParameterValue(Command(['xacro ', PathJoinSubstitution([str(pkg_path), 'urdf/go1/go1.urdf.xacro'])]))

def generate_launch_description():
  

    return LaunchDescription([



        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),



        Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_effort_controller",
            "--controller-manager",
            "/controller_manager"
            ]
        ),

        Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_states_controller",
            "--controller-manager",
            "/controller_manager"
            ]
        ),


        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_path),
            launch_arguments={
                'use_sim_time': str("true"),
                'world': world_path,
            }.items()
        ),

         
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'go1', '-z', '0.4'],
            output='screen'
        ),

    ])
