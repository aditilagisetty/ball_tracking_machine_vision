"""ROS 2 node that ties detection, triangulation, and filtering together."""

import rclpy
from rclpy.node import Node


class TrackerNode(Node):

    def __init__(self):
        super().__init__('tracker_node')

    def image_callback(self, msg):
        """Handle a new stereo frame."""
        pass

    def publish_estimate(self, state):
        """Publish the filtered ball state."""
        pass

    def publish_landing(self, landing_point):
        """Publish the predicted landing point."""
        pass


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
