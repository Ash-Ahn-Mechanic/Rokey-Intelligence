#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from vision_interfaces.srv import CaptureImage


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('image_topic', '/robot4/oakd/rgb/preview/image_raw')
        self.declare_parameter('jpg_quality', 95)

        image_topic = self.get_parameter('image_topic').value
        self._bridge = CvBridge()
        self._latest_image = None

        self._image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, 10)
        self._capture_srv = self.create_service(
            CaptureImage, '/vision/capture_image', self._capture_callback)

        self.get_logger().info(f'CameraNode started. Subscribing: {image_topic}')

    def _image_callback(self, msg: Image):
        self._latest_image = msg

    def _capture_callback(self, request, response):
        if self._latest_image is None:
            response.success = False
            response.message = 'No image received yet'
            self.get_logger().warn('Capture requested but no image available')
            return response

        save_path = request.save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            cv_image = self._bridge.imgmsg_to_cv2(self._latest_image, 'bgr8')
            quality = self.get_parameter('jpg_quality').value
            cv2.imwrite(save_path, cv_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            self.get_logger().info(f'Saved: {save_path}')
            response.success = True
            response.message = save_path
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f'Save failed: {e}')

        return response


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
