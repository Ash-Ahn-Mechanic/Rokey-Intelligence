#!/usr/bin/env python3
"""usb_cam 대체 노드: OpenCV VideoCapture로 직접 캡처 후 /ext_cam/image_raw 발행."""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image


class CvCamPublisher(Node):
    def __init__(self):
        super().__init__('cv_cam_publisher')

        self.declare_parameter('video_device', '/dev/video2')
        self.declare_parameter('image_width',  640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('framerate',    25.0)
        self.declare_parameter('frame_id',     'ext_cam_frame')

        device  = self.get_parameter('video_device').value
        width   = self.get_parameter('image_width').value
        height  = self.get_parameter('image_height').value
        fps     = self.get_parameter('framerate').value
        self._frame_id = self.get_parameter('frame_id').value

        # /dev/videoN → index N  (cam_open.py 와 동일한 방식)
        dev_idx = int(device.replace('/dev/video', '')) \
            if isinstance(device, str) else int(device)

        self._cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        if not self._cap.isOpened():
            self.get_logger().error(f'카메라를 열 수 없음: {device}')
            return

        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self._cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(f'카메라 열림: {device}  {aw}x{ah} @ {af:.0f}fps')

        qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub = self.create_publisher(Image, '/ext_cam/image_raw', qos)
        self.create_timer(1.0 / fps, self._publish)

    def _publish(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn('프레임 캡처 실패', throttle_duration_sec=2.0)
            return

        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height   = frame.shape[0]
        msg.width    = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.step     = frame.shape[1] * 3
        msg.data     = frame.tobytes()
        self._pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, '_cap'):
            self._cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = CvCamPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
