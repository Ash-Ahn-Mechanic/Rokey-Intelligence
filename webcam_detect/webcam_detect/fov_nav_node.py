#!/usr/bin/env python3
"""
FOV 마커 + Goal 요청 노드
- /car_detected (Bool) 구독
- car 감지 시 FOV 삼각형 중점 좌표를 /nav_goal_request (PoseStamped) 로 발행
- turtlebot_state 가 IDLE 상태일 때만 실제 이동 처리
- RViz2 에 FOV 삼각형 마커 항상 표시
"""

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, ColorRGBA, String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped


class FovNavNode(Node):
    def __init__(self):
        super().__init__('fov_nav_node')

        self.declare_parameter('cam_x',         -3.0)
        self.declare_parameter('cam_y',          2.5)
        self.declare_parameter('cam_yaw',        0.0)
        self.declare_parameter('fov_angle_deg', 40.0)
        self.declare_parameter('fov_distance',   1.5)
        self.declare_parameter('map_frame',     'map')

        p = self.get_parameter
        self.cam_x     = p('cam_x').value
        self.cam_y     = p('cam_y').value
        self.cam_yaw   = p('cam_yaw').value
        self.fov_angle = math.radians(p('fov_angle_deg').value)
        self.fov_dist  = p('fov_distance').value
        self.map_frame = p('map_frame').value

        self._detected = False
        self._goal_reached = False         # 한 번 도달한 목표는 재발행 차단
        self._was_driving_webcam = False   # DRIVING_WEBCAM → IDLE 전환 감지용

        # FOV 중점 좌표 (한 번만 계산)
        self._goal_x = self.cam_x + (self.fov_dist * 0.5) * math.cos(self.cam_yaw)
        self._goal_y = self.cam_y + (self.fov_dist * 0.5) * math.sin(self.cam_yaw)

        self._goal_pub   = self.create_publisher(PoseStamped, '/nav_goal_request_webcam', 10)
        self._marker_pub = self.create_publisher(Marker, '/cam_fov_marker', 10)

        self.create_subscription(Bool, '/car_detected', self._on_detected, 10)
        self.create_subscription(String, '/robot_state', self._on_robot_state, 10)
        self.create_timer(0.5, self._publish_fov)

        self.get_logger().info(
            f'cam=({self.cam_x},{self.cam_y})  yaw={math.degrees(self.cam_yaw):.1f}°  '
            f'goal=({self._goal_x:.2f},{self._goal_y:.2f})'
        )

    def _on_robot_state(self, msg: String):
        state = msg.data
        if state == 'DRIVING_WEBCAM':
            self._was_driving_webcam = True
        elif state == 'IDLE' and self._was_driving_webcam:
            self._goal_reached = True
            self._was_driving_webcam = False
            self.get_logger().info('[FOV] 목표 도달 확인 — 동일 좌표 재발행 차단')

    def _on_detected(self, msg: Bool):
        if msg.data == self._detected:
            return
        self._detected = msg.data
        if self._detected and not self._goal_reached:
            self._publish_goal_request()
        self._publish_fov()

    def _publish_goal_request(self):
        pose = PoseStamped()
        pose.header.frame_id    = self.map_frame
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = self._goal_x
        pose.pose.position.y    = self._goal_y
        pose.pose.orientation.z = math.sin(self.cam_yaw / 2.0)
        pose.pose.orientation.w = math.cos(self.cam_yaw / 2.0)
        self._goal_pub.publish(pose)
        self.get_logger().warn(
            f'[FOV] goal 요청 발행 → ({self._goal_x:.2f}, {self._goal_y:.2f})')

    def _publish_fov(self):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = 'cam_fov'
        m.id     = 0
        m.type   = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = ColorRGBA(r=1.0, g=0.1, b=0.0, a=0.5) if self._detected \
             else ColorRGBA(r=0.0, g=0.85, b=0.2, a=0.25)

        half = self.fov_angle / 2.0
        d    = self.fov_dist
        cx, cy, yaw = self.cam_x, self.cam_y, self.cam_yaw
        m.points = [
            Point(x=cx, y=cy, z=0.02),
            Point(x=cx + d * math.cos(yaw + half), y=cy + d * math.sin(yaw + half), z=0.02),
            Point(x=cx + d * math.cos(yaw - half), y=cy + d * math.sin(yaw - half), z=0.02),
        ]
        self._marker_pub.publish(m)


def main():
    rclpy.init()
    node = FovNavNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
