from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('vision_data_collector'),
        'config', 'params.yaml'
    )

    mission_node = Node(
        package='vision_data_collector',
        executable='mission_node',
        name='mission_node',
        output='screen',
    )

    camera_node = Node(
        package='vision_data_collector',
        executable='camera_node',
        name='camera_node',
        parameters=[config],
        output='screen',
    )

    return LaunchDescription([mission_node, camera_node])
