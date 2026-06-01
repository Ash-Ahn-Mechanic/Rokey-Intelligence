import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
from cv_bridge import CvBridge

#---------------------
# (1) config constant
#---------------------
Depth_Topic = '/robot4/oakd/stereo/image_raw'
Camera_Info_Topic = '/robot4/oakd/stereo/camera_info'
Max_Depth_Meters = 5.0 # 측정되는 최대 depth
Normalize_Depth_Range = 3.0 # Depth 정규화 범위 (over 3m, 255 clamp)

#---------------------
# (2) DepthChecker Node
#---------------------
class DepthChecker(Node):
    # - initial settings
    def __init__(self):
        super().__init__('depth_checker')
        self.bridge = CvBridge()
        self.K = None
        self.should_exit = False
        
        # - depth img subscription
        self.depth_sub = self.create_subscription(
            Image,
            Depth_Topic,
            self.depth_callback,
            10
        )

        # - depth camera info subscription
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            Camera_Info_Topic,
            self.camera_info_callback,
            10
        )

    # - callback_1: camera info
    def camera_info_callback(self, msg):
        if self.K is None:
            # - camera 내부 parameter 행렬(K)를 (3, 3) form으로
            # - K = [fx,  0, cx,
            # -      0, fy, cy,
            # -      0,  0,  1]
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    # - callback_2: depth img
    def depth_callback(self, msg):
        
        if self.should_exit:
            return
        
        if self.K is None:
            self.get_logger().warn('Waiting for CameraInfo...')
            return
        
        # - img 메세지를 OpenCV 배열로 뱐환(이때 원본 type(uint16 or float32) 그대로 반환)
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        height, width = depth_mm.shape

        cx = self.K[0, 2]
        cy = self.K[1, 2]
        # camera 보정 행렬에서의 중심점 좌표 추출
        u, v = int(cx), int(cy)

        # depth img의 중앙값 읽기
        distance_mm = depth_mm[v, u]
        distance_m = distance_mm / 1000.0

        self.get_logger().info(f"Image size: {width}x{height}, Distance at (u={u}, v={v}) = {distance_m:.2f} meters")

        # depth img를 시각화 전처리
        # - Nan값을 0으로 치환
        # - 3000mm(3m)이상은 0으로 clamp
        # - 해당 이미지 값을 0~255로 정규화
        depth_vis = np.nan_to_num(depth_mm, nan=0.0)
        depth_vis = np.clip(depth_vis, 0, Normalize_Depth_Range*1000)
        depth_vis = (depth_vis / (Normalize_Depth_Range*1000)*255).astype(np.uint8)

        # jet colormap 적용(0=blue, 255=red)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # colormap 중앙 maker 생성
        cv2.circle(depth_colored, (u,v), 5, (0,0,0), -1)
        cv2.line(depth_colored, (0,v), (width, v), (0,0,0), 1)
        cv2.line(depth_colored, (u,0), (u, height), (0,0,0), 1)

        # Cv window 출력: 'q' make shut down
        cv2.imshow('Depth Image with Center Mark', depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.should_exit = True

#---------------------
# (3) Main Sequence
#---------------------
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