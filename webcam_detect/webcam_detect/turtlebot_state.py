#!/usr/bin/env python3
"""
TurtleBot4 State Manager
- 시작 시 도킹 상태면 자동 언도킹 후 IDLE
- IDLE 일 때 webcam/turtle goal 수신 가능
- DRIVING_TURTLE 은 DRIVING_WEBCAM 을 즉시 선점(preempt)
- 두 카메라 모두 car 미감지 시 SEARCHING (회전 탐색)
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from enum import Enum

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose

try:
    from irobot_create_msgs.msg import DockStatus
    from irobot_create_msgs.action import Undock
    _HAS_DOCK = True
except ImportError:
    _HAS_DOCK = False


class RobotState(Enum):
    UNDOCKING      = "UNDOCKING"
    IDLE           = "IDLE"
    SEARCHING      = "SEARCHING"
    DRIVING_WEBCAM = "DRIVING_WEBCAM"
    DRIVING_TURTLE = "DRIVING_TURTLE"
    EMERGENCY      = "EMERGENCY"


class TurtlebotStateManager(Node):
    def __init__(self):
        super().__init__('turtlebot_state_manager')

        self.declare_parameter('nav_namespace',      '/robot4')
        self.declare_parameter('goal_cooldown_sec',   10.0)
        self.declare_parameter('lost_search_delay',    1.0)   # 미감지 후 탐색 시작 딜레이(초)
        self.declare_parameter('search_angular_speed', 0.25)  # 회전 속도 (rad/s)
        self.declare_parameter('search_rotation_angle', 2.0 * 3.141592)  # 총 회전각 (rad)

        nav_ns               = self.get_parameter('nav_namespace').value.rstrip('/')
        self._cooldown       = self.get_parameter('goal_cooldown_sec').value
        self._lost_delay     = self.get_parameter('lost_search_delay').value
        self._search_speed   = abs(self.get_parameter('search_angular_speed').value)
        search_angle         = abs(self.get_parameter('search_rotation_angle').value)
        self._search_dur     = search_angle / self._search_speed if self._search_speed > 0 else 0.0

        # ── 상태 변수 ──────────────────────────────────────────────────
        self._state          = RobotState.IDLE
        self._pending_goal   = None
        self._pending_state  = RobotState.IDLE
        self._last_goal_t    = 0.0
        self._goal_handle    = None
        self._dock_initialized = False

        # ── 감지 상태 ─────────────────────────────────────────────────
        self._webcam_detected = False
        self._turtle_detected = False
        self._lost_start_t    = 0.0   # 둘 다 미감지 시작 시각 (0=미시작)

        # ── 탐색 회전 상태 ────────────────────────────────────────────
        self._search_start_t  = 0.0
        self._search_done     = False

        # ── ROS 인터페이스 ────────────────────────────────────────────
        self._nav_client    = ActionClient(self, NavigateToPose, f'{nav_ns}/navigate_to_pose')
        self._undock_client = ActionClient(self, Undock, f'{nav_ns}/undock') if _HAS_DOCK else None
        self._state_pub    = self.create_publisher(String, '/robot_state', 10)
        self._cmd_vel_pub  = self.create_publisher(Twist,  f'{nav_ns}/cmd_vel', 10)

        self.create_subscription(PoseStamped, '/nav_goal_request_webcam', self._on_webcam_goal, 10)
        self.create_subscription(PoseStamped, '/nav_goal_request_turtle', self._on_turtle_goal, 10)
        self.create_subscription(Bool, '/car_detected',        self._on_webcam_det, 10)
        self.create_subscription(Bool, '/turtle_car_detected', self._on_turtle_det, 10)

        self._is_docked = False
        if _HAS_DOCK:
            self.create_subscription(DockStatus, f'{nav_ns}/dock_status', self._dock_cb, 10)
        else:
            self.get_logger().warn('irobot_create_msgs 없음 — dock 감지 비활성화')
            self._dock_initialized = True  # dock 확인 불가 → 즉시 허용

        self.create_timer(0.1, self._search_loop)   # 탐색 제어 (10 Hz)
        self.create_timer(0.5, self._publish_state)
        self.get_logger().info(f'StateManager 시작  nav_ns={nav_ns}')

    # ── 감지 콜백 ──────────────────────────────────────────────────────
    def _on_webcam_det(self, msg: Bool):
        self._webcam_detected = msg.data
        if msg.data:
            self._on_any_detected()

    def _on_turtle_det(self, msg: Bool):
        self._turtle_detected = msg.data
        if msg.data:
            self._on_any_detected()

    def _on_any_detected(self):
        """어느 카메라라도 car 감지 → 탐색 중이면 중단, IDLE 복귀."""
        self._lost_start_t  = 0.0
        self._search_done   = False
        if self._state == RobotState.SEARCHING:
            self._stop_rotation()
            self._state = RobotState.IDLE
            self.get_logger().info('[STATE] car 감지 → SEARCHING 종료 → IDLE')

    # ── 탐색 루프 (10 Hz timer) ────────────────────────────────────────
    def _search_loop(self):
        both_lost = (not self._webcam_detected) and (not self._turtle_detected)

        if self._state == RobotState.IDLE:
            if not self._dock_initialized:
                return  # dock 상태 미확인 — 대기
            if self._is_docked:
                return  # 도킹 상태 — undock 서비스 실패해도 SEARCHING 진입 차단
            if both_lost:
                now = time.monotonic()
                if self._lost_start_t == 0.0:
                    self._lost_start_t = now
                elif (not self._search_done and
                      now - self._lost_start_t >= self._lost_delay):
                    self._start_searching()
            else:
                self._lost_start_t = 0.0

        elif self._state == RobotState.SEARCHING:
            if not both_lost:
                return  # _on_any_detected 가 처리
            now = time.monotonic()
            if now - self._search_start_t >= self._search_dur:
                self._stop_rotation()
                self._search_done = True
                self._state = RobotState.IDLE
                self.get_logger().info('[SEARCH] 회전 완료 → IDLE')
            else:
                self._publish_rotation()

    def _start_searching(self):
        self._state         = RobotState.SEARCHING
        self._search_start_t = time.monotonic()
        self.get_logger().warn('[SEARCH] SEARCHING 시작 — 회전 탐색')

    def _publish_rotation(self):
        twist = Twist()
        twist.angular.z = self._search_speed
        self._cmd_vel_pub.publish(twist)

    def _stop_rotation(self):
        self._cmd_vel_pub.publish(Twist())

    # ── 도킹 콜백 ──────────────────────────────────────────────────────
    def _dock_cb(self, msg):
        self._is_docked = msg.is_docked
        if not self._dock_initialized:
            self._dock_initialized = True
            if self._is_docked:
                self.get_logger().info('[DOCK] 도킹 감지 → 언도킹 시작')
                self._start_undocking()
            else:
                self.get_logger().info('[DOCK] 언도킹 상태 → IDLE')
        elif self._state == RobotState.UNDOCKING and not self._is_docked:
            self.get_logger().info('[DOCK] 언도킹 완료 → IDLE')
            self._state = RobotState.IDLE

    def _start_undocking(self):
        self._state = RobotState.UNDOCKING
        if not self._undock_client.server_is_ready():
            self.get_logger().error('[UNDOCK] 액션 서버 없음 → IDLE 강제 전환')
            self._state = RobotState.IDLE
            return
        future = self._undock_client.send_goal_async(Undock.Goal())
        future.add_done_callback(self._undock_response_cb)

    def _undock_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('[UNDOCK] goal 거부 → IDLE 강제 전환')
            self._state = RobotState.IDLE
            return
        self.get_logger().info('[UNDOCK] 언도킹 중...')
        handle.get_result_async().add_done_callback(
            lambda _: self.get_logger().info('[UNDOCK] 완료'))

    # ── Webcam Goal ────────────────────────────────────────────────────
    def _on_webcam_goal(self, msg: PoseStamped):
        self._request_goal(msg, RobotState.DRIVING_WEBCAM)

    # ── Turtle Goal (DRIVING_WEBCAM 선점 가능) ─────────────────────────
    def _on_turtle_goal(self, msg: PoseStamped):
        if self._state == RobotState.DRIVING_WEBCAM and self._goal_handle is not None:
            self.get_logger().warn('[STATE] DRIVING_WEBCAM 취소 → DRIVING_TURTLE 선점')
            self._pending_goal  = msg
            self._pending_state = RobotState.DRIVING_TURTLE
            cancel_f = self._goal_handle.cancel_goal_async()
            cancel_f.add_done_callback(self._on_webcam_cancelled)
        elif self._state == RobotState.SEARCHING:
            # 탐색 중 turtle goal 수신 → 탐색 중단 후 이동
            self._stop_rotation()
            self._state = RobotState.IDLE
            self._request_goal(msg, RobotState.DRIVING_TURTLE)
        else:
            self._request_goal(msg, RobotState.DRIVING_TURTLE)

    def _on_webcam_cancelled(self, _):
        self._goal_handle = None
        self._state       = RobotState.IDLE
        self.get_logger().warn('[STATE] 취소 완료 → DRIVING_TURTLE 시작')
        self._dispatch_goal()

    # ── Goal 요청 공통 처리 ────────────────────────────────────────────
    def _request_goal(self, msg: PoseStamped, target_state: RobotState):
        now = self.get_clock().now().nanoseconds / 1e9

        if self._state != RobotState.IDLE:
            self.get_logger().info(
                f'[STATE] {self._state.value} — {target_state.value} 요청 무시')
            return

        if now - self._last_goal_t < self._cooldown:
            remaining = self._cooldown - (now - self._last_goal_t)
            self.get_logger().info(f'[STATE] 쿨다운 중 (남은: {remaining:.1f}s)')
            return

        self._pending_goal  = msg
        self._pending_state = target_state
        self._dispatch_goal()

    # ── NavigateToPose 전송 ────────────────────────────────────────────
    def _dispatch_goal(self):
        if not self._nav_client.server_is_ready():
            self.get_logger().error('[STATE] Nav2 서버 미준비')
            self._state = RobotState.IDLE
            return

        pose = self._pending_goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self._state        = self._pending_state
        self._last_goal_t  = self.get_clock().now().nanoseconds / 1e9
        self._pending_goal = None

        future = self._nav_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_cb)
        future.add_done_callback(self._goal_response_cb)

        self.get_logger().warn(
            f'[NAV] {self._state.value} goal → '
            f'({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')

    def _feedback_cb(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        self.get_logger().info(f'[NAV] 남은 거리: {dist:.2f}m', throttle_duration_sec=3.0)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('[NAV] goal 거부 → IDLE')
            self._state = RobotState.IDLE
            return
        self._goal_handle = handle
        self.get_logger().info('[NAV] goal 수락 — 이동 중')
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._goal_handle = None
        self._state       = RobotState.IDLE
        self.get_logger().warn('[NAV] 목표 도달 완료 → IDLE')

    # ── 상태 퍼블리시 ──────────────────────────────────────────────────
    def _publish_state(self):
        msg = String()
        msg.data = self._state.value
        self._state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotStateManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
