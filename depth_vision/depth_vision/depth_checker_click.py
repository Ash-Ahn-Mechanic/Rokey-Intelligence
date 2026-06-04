import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo
import numpy as np
import cv2

# ================================
# 설정 상수
# ================================
DEPTH_TOPIC = '/robot4/oakd/stereo/image_raw/compressedDepth'
CAMERA_INFO_TOPIC = '/robot4/oakd/stereo/camera_info'
MAX_DEPTH_METERS = 5.0
NORMALIZE_DEPTH_RANGE = 3.0
WINDOW_NAME = 'Depth Image (Click to get distance)'
# ================================

class DepthChecker(Node):
    def __init__(self):
        super().__init__('depth_checker')
        self.K = None
        self.should_exit = False
        self.depth_mm = None
        self.depth_colored = None

        self.subscription = self.create_subscription(
            CompressedImage,
            DEPTH_TOPIC,
            self.depth_callback,
            10)

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
            )

    def depth_callback(self, msg):
        if self.should_exit:
            return

        if self.K is None:
            self.get_logger().warn('Waiting for CameraInfo...')
            return

        # compressedDepth: 12바이트 ConfigHeader 제거 후 PNG 디코딩
        raw = bytes(msg.data)
        np_arr = np.frombuffer(raw[12:], dtype=np.uint8)
        self.depth_mm = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)

        if self.depth_mm is None:
            self.get_logger().warn('imdecode failed')
            return

        height, width = self.depth_mm.shape

        depth_vis = np.nan_to_num(self.depth_mm, nan=0.0)
        depth_vis = np.clip(depth_vis, 0, NORMALIZE_DEPTH_RANGE * 1000)
        depth_vis = (depth_vis / (NORMALIZE_DEPTH_RANGE * 1000) * 255).astype(np.uint8)
        self.depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cx = int(self.K[0, 2])
        cy = int(self.K[1, 2])
        cv2.circle(self.depth_colored, (cx, cy), 5, (0, 0, 0), -1)
        cv2.line(self.depth_colored, (0, cy), (width, cy), (0, 0, 0), 1)
        cv2.line(self.depth_colored, (cx, 0), (cx, height), (0, 0, 0), 1)

        cv2.imshow(WINDOW_NAME, self.depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.should_exit = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.depth_mm is not None:
            distance_mm = self.depth_mm[y, x]
            distance_m = distance_mm / 1000.0
            self.get_logger().info(f"Clicked at (u={x}, v={y}) → Distance = {distance_m:.2f} meters")

def main():
    rclpy.init()
    node = DepthChecker()

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
