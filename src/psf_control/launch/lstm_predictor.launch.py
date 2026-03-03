from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch LSTM predictor node"""

    lstm_predictor_node = Node(
        package="psf_control",
        executable="lstm_predictor",
        name="lstm_predictor",
        output="screen",
    )

    return LaunchDescription([lstm_predictor_node])
