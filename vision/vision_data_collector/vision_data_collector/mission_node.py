#!/usr/bin/env python3
import os
import yaml
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler

from vision_interfaces.srv import StartMission, CaptureImage
from vision_data_collector.orbit_calculator import calculate_orbit_positions, filter_by_costmap


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')
        self._costmap: OccupancyGrid = None
        self._mission_active = False

        self._costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/robot4/global_costmap/costmap',
            self._costmap_callback,
            10,
        )
        self._status_pub = self.create_publisher(String, '/vision/mission_status', 10)
        self._nav_client = ActionClient(self, NavigateToPose, '/robot4/navigate_to_pose')
        self._capture_client = self.create_client(CaptureImage, '/vision/capture_image')
        self._mission_srv = self.create_service(
            StartMission, '/vision/start_mission', self._start_mission_callback)

        self.get_logger().info('MissionNode started')

    def _costmap_callback(self, msg: OccupancyGrid):
        self._costmap = msg

    def _publish_status(self, msg: str):
        self.get_logger().info(msg)
        self._status_pub.publish(String(data=msg))

    def _start_mission_callback(self, request, response):
        if self._mission_active:
            response.success = False
            response.message = 'Mission already running'
            return response
        if self._costmap is None:
            response.success = False
            response.message = 'Costmap not received yet'
            return response

        self._mission_active = True
        save_dir = os.path.expanduser(request.save_dir)

        try:
            metadata = []
            for wp_idx, waypoint in enumerate(request.waypoints):
                self._publish_status(f'Waypoint {wp_idx}: calculating orbit')

                positions = calculate_orbit_positions(
                    waypoint.x, waypoint.y,
                    request.radius,
                    request.angle_step_deg,
                )
                info = self._costmap.info
                valid_positions = filter_by_costmap(
                    positions,
                    costmap_data=list(self._costmap.data),
                    width=info.width,
                    height=info.height,
                    resolution=info.resolution,
                    origin_x=info.origin.position.x,
                    origin_y=info.origin.position.y,
                )
                self._publish_status(
                    f'Waypoint {wp_idx}: {len(valid_positions)}/{len(positions)} positions valid'
                )

                wp_dir = os.path.join(save_dir, f'waypoint_{wp_idx}')
                os.makedirs(wp_dir, exist_ok=True)

                for pos_idx, (px, py, yaw) in enumerate(valid_positions):
                    angle_deg = int(pos_idx * request.angle_step_deg)
                    self._publish_status(f'Waypoint {wp_idx} angle {angle_deg}deg: navigating')

                    success = self._navigate_to(px, py, yaw)
                    if not success:
                        self._publish_status(
                            f'Waypoint {wp_idx} angle {angle_deg}deg: navigation failed, skipping'
                        )
                        continue

                    save_path = os.path.join(wp_dir, f'angle_{angle_deg:03d}.jpg')
                    cap_success = self._capture(save_path)

                    metadata.append({
                        'waypoint': wp_idx,
                        'angle_deg': angle_deg,
                        'robot_x': px,
                        'robot_y': py,
                        'robot_yaw': yaw,
                        'obstacle_x': waypoint.x,
                        'obstacle_y': waypoint.y,
                        'saved': save_path if cap_success else None,
                        'timestamp': datetime.now().isoformat(),
                    })

            meta_path = os.path.join(save_dir, 'metadata.yaml')
            with open(meta_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

            self._publish_status('Mission complete')
            response.success = True
            response.message = f'Saved to {save_dir}'
        except Exception as e:
            self.get_logger().error(f'Mission error: {e}')
            response.success = False
            response.message = str(e)
        finally:
            self._mission_active = False

        return response

    def _navigate_to(self, x: float, y: float, yaw: float) -> bool:
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y

        q = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.pose.orientation.x = q[0]
        goal.pose.pose.orientation.y = q[1]
        goal.pose.pose.orientation.z = q[2]
        goal.pose.pose.orientation.w = q[3]

        future = self._nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        return result_future.result() is not None

    def _capture(self, save_path: str) -> bool:
        if not self._capture_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('CaptureImage service not available')
            return False

        req = CaptureImage.Request()
        req.save_path = save_path
        future = self._capture_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            return False
        return future.result().success


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
