#!/usr/bin/env python3
"""
YOLO 감지 노드
- /ext_cam/image_raw 구독
- 'car' 클래스 감지 시 /car_detected (Bool) 발행
- 바운딩 박스 그린 영상을 /ext_cam/image_yolo 로 발행
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from ultralytics import YOLO


def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(msg.encoding, 3)
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return arr.reshape(msg.height, msg.width, channels) if channels > 1 else arr.reshape(msg.height, msg.width)


def numpy_to_imgmsg(arr: np.ndarray, encoding: str, header) -> Image:
    msg = Image()
    msg.header   = header
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.step     = arr.shape[1] * (arr.shape[2] if arr.ndim == 3 else 1)
    msg.data     = arr.tobytes()
    return msg


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        model_path    = '/home/rokey/rokey_ws/src/webcam_detect/resource/best_webcam.pt'
        self._target  = 'car'
        self._conf    = 0.5

        self._model = YOLO(model_path)
        self.get_logger().info(f'YOLO 모델 로드: {model_path}')

        cam_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._det_pub = self.create_publisher(Bool,  '/car_detected',        10)
        self._img_pub = self.create_publisher(Image, '/ext_cam/image_yolo',  10)

        self.create_subscription(Image, '/ext_cam/image_raw', self._on_image, cam_qos)
        self.get_logger().info(f'감지 클래스: {self._target}  conf>={self._conf}')

    def _on_image(self, msg: Image):
        frame = imgmsg_to_numpy(msg).copy()  # frombuffer 반환값은 readonly → 쓰기 가능 복사본
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = self._model(frame, conf=self._conf, verbose=False)

        detected = False
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)].lower() == self._target:
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{self._target} {conf:.2f}',
                                (x1, max(y1 - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self._det_pub.publish(Bool(data=detected))
        self._img_pub.publish(numpy_to_imgmsg(frame, 'bgr8', msg.header))

        if detected:
            self.get_logger().info('[YOLO] car 감지!', throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
