import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from builtin_interfaces.msg import Duration as DurationMsg

from tf2_ros import *

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')

        self.__marker_array_sub = self.create_subscription(
            MarkerArray,
            "/natnet_client_node/unlabeled",
            self._marker_array_callback,
            qos_profile=qos_profile_system_default
        )

        self._ids = []

        self.get_logger().info("TestNode has been initialized.")


    def _marker_array_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            marker: Marker

            if marker.id not in self._ids:
                self._ids.append(marker.id)
                self.get_logger().info(f"New marker ID: {marker.id}")



def main(args=None):
    rclpy.init(args=args)

    test_node = TestNode()

    rclpy.spin(test_node)

    test_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()