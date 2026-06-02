#!/usr/bin/env python3
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Bool
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image as ROSImage
import tf2_geometry_msgs  # Registers PointStamped transforms with tf2.
from tf2_ros import Buffer, TransformException, TransformListener
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions
from ultralytics import YOLO


CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640


class YOLOCarNavigator(Node):
    def __init__(self, model):
        super().__init__("yolo_car_navigator")

        self.model = model
        self.bridge = CvBridge()
        self.class_names = model.names if hasattr(model, "names") else {}
        self.lock = threading.Lock()

        ns = self.get_namespace().rstrip("/")
        topic_prefix = ns if ns else ""
        self.declare_parameter("rgb_topic", f"{topic_prefix}/oakd/rgb/image_raw/compressed")
        self.declare_parameter("depth_topic", f"{topic_prefix}/oakd/stereo/image_raw")
        self.declare_parameter("camera_info_topic", f"{topic_prefix}/oakd/rgb/camera_info")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("target_class", "car")
        self.declare_parameter("stop_distance", 0.4)
        self.declare_parameter("min_depth", 0.2)
        self.declare_parameter("max_depth", 5.0)
        self.declare_parameter("goal_cooldown", 5.0)
        self.declare_parameter("sync_slop", 1.0)
        self.declare_parameter("send_nav_goal", True)
        self.declare_parameter("show_window", True)
        self.declare_parameter("init_turtlebot4_nav", False)
        self.declare_parameter("lost_search_rotation", False)  # turtlebot_state 가 담당
        self.declare_parameter("lost_search_delay", 1.0)
        self.declare_parameter("search_angular_speed", 0.25)
        self.declare_parameter("search_rotation_angle", 2.0 * math.pi)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.info_topic = self.get_parameter("camera_info_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.target_class = self.get_parameter("target_class").value
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.goal_cooldown = float(self.get_parameter("goal_cooldown").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)
        self.send_nav_goal = bool(self.get_parameter("send_nav_goal").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.init_turtlebot4_nav = bool(self.get_parameter("init_turtlebot4_nav").value)
        self.lost_search_rotation = bool(self.get_parameter("lost_search_rotation").value)
        self.lost_search_delay = float(self.get_parameter("lost_search_delay").value)
        self.search_angular_speed = abs(float(self.get_parameter("search_angular_speed").value))
        self.search_rotation_angle = abs(float(self.get_parameter("search_rotation_angle").value))

        self.K = None
        self.rgb_image = None
        self.depth_image = None
        self.depth_frame = None
        self.display_image = None
        self.shutdown_requested = False
        self.last_goal_time = 0.0
        self.last_warn_time = 0.0
        self.goal_running = False
        self.nav_ready = not self.send_nav_goal
        self.last_sync_time = 0.0
        self.last_target_time = 0.0
        self.lost_target_start_time = 0.0
        self.search_rotation_done = False
        self.search_rotation_active = False
        self.search_rotation_start = 0.0
        self.search_rotation_duration = (
            self.search_rotation_angle / self.search_angular_speed
            if self.search_angular_speed > 0.0 else 0.0
        )
        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        self.target_class_ids = self.resolve_target_class_ids()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = TurtleBot4Navigator(namespace=topic_prefix)
        if self.init_turtlebot4_nav:
            if not self.navigator.getDockedStatus():
                self.get_logger().info("Docking before initializing pose")
                self.navigator.dock()
            initial_pose = self.navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
            self.navigator.setInitialPose(initial_pose)
            self.navigator.waitUntilNav2Active()
            self.nav_ready = True
            self.navigator.undock()

        self.target_pub   = self.create_publisher(PointStamped, "detected_car_point", 1)
        self.goal_pub     = self.create_publisher(PoseStamped, "detected_car_goal", 1)
        self.goal_req_pub = self.create_publisher(PoseStamped, "/nav_goal_request_turtle", 1)
        self.det_pub      = self.create_publisher(Bool, "/turtle_car_detected", 1)
        self.rgb_pub      = self.create_publisher(ROSImage, f"{topic_prefix}/yolo_car_rgb", 1)
        self.depth_pub    = self.create_publisher(ROSImage, f"{topic_prefix}/yolo_car_depth", 1)
        self.cmd_vel_pub  = self.create_publisher(Twist, f"{topic_prefix}/cmd_vel", 1)

        self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.rgb_sub = Subscriber(
            self,
            CompressedImage,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = Subscriber(
            self,
            ROSImage,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=self.sync_slop,
        )
        self.ts.registerCallback(self.synced_image_callback)

        self.gui_thread_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        self.gui_thread.start()
        self.nav_wait_thread = None
        if self.send_nav_goal and not self.nav_ready:
            self.nav_wait_thread = threading.Thread(target=self.wait_for_nav2_active, daemon=True)
            self.nav_wait_thread.start()

        self.get_logger().info(f"RGB: {self.rgb_topic}")
        self.get_logger().info(f"Depth: {self.depth_topic}")
        self.get_logger().info(f"CameraInfo: {self.info_topic}")
        self.get_logger().info(f"RGB/depth sync slop: {self.sync_slop:.2f} sec")
        self.get_logger().info("TF Tree stabilization starting. Will begin YOLO processing in 5 sec.")
        self.start_timer = self.create_timer(5.0, self.start_processing)

    def wait_for_nav2_active(self):
        self.get_logger().info("Waiting for Nav2. Camera display will start independently.")
        try:
            self.navigator.waitUntilNav2Active()
        except Exception as exc:
            self.get_logger().error(f"Nav2 wait failed: {exc}")
            return
        self.nav_ready = True
        self.get_logger().info("Nav2 is active. Navigation goals enabled.")

    def start_processing(self):
        self.get_logger().info("TF Tree stabilized. Starting YOLO car navigation.")
        self.timer = self.create_timer(0.2, self.process_and_publish)
        self.start_timer.cancel()

    def resolve_target_class_ids(self):
        if not self.target_class:
            return None

        names = self.class_names
        if isinstance(names, dict):
            matches = [
                idx for idx, name in names.items()
                if str(name).lower() == self.target_class.lower()
            ]
        else:
            matches = [
                idx for idx, name in enumerate(names)
                if str(name).lower() == self.target_class.lower()
            ]

        if matches:
            return set(matches)

        self.get_logger().warn(
            f"Class '{self.target_class}' not found in model names. "
            "All detected classes will be considered targets."
        )
        return None

    def camera_info_callback(self, msg):
        with self.lock:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.info_width = msg.width
            self.info_height = msg.height
            if not self.logged_intrinsics:
                self.get_logger().info(
                    f"Camera intrinsics: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                    f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
                )
                self.logged_intrinsics = True

    def synced_image_callback(self, rgb_msg, depth_msg):
        try:
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Synced image conversion failed: {exc}")
            return

        if rgb is None or rgb.size == 0 or depth is None or depth.size == 0:
            self.throttled_warn("Waiting for valid RGB/depth images.")
            return

        with self.lock:
            if not self.logged_rgb_shape:
                self.get_logger().info(f"RGB image shape: {rgb.shape}")
                self.logged_rgb_shape = True
            if not self.logged_depth_shape:
                self.get_logger().info(f"Depth image shape: {depth.shape}")
                self.logged_depth_shape = True

            self.rgb_image = rgb
            self.depth_image = depth
            self.depth_frame = depth_msg.header.frame_id
            self.last_sync_time = time.monotonic()

    def process_and_publish(self):
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
            frame_id = self.depth_frame
            K = self.K.copy() if self.K is not None else None
            info_width = getattr(self, "info_width", 0)
            info_height = getattr(self, "info_height", 0)

        if rgb is None or depth is None or frame_id is None:
            return
        if K is None:
            self.throttled_warn("Waiting for camera intrinsics.")
            return

        rgb_display = rgb.copy()
        depth_display = depth.copy()
        depth_colored = self.colorize_depth(depth_display)

        start_time = time.perf_counter()
        detections = self.detect_targets(rgb)
        process_ms = (time.perf_counter() - start_time) * 1000.0
        best_target = None

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            depth_m, depth_u, depth_v = self.depth_at_rgb_pixel(
                depth_display, u, v, rgb_display.shape[1], rgb_display.shape[0]
            )

            label = f"{self.class_name(cls)} {conf:.2f}"
            if depth_m is not None:
                label += f" {depth_m:.2f}m"
                if best_target is None or conf > best_target["conf"]:
                    best_target = {
                        "u": u,
                        "v": v,
                        "depth_m": depth_m,
                        "detection": detection,
                        "conf": conf,
                    }

                cv2.circle(depth_colored, (depth_u, depth_v), 4, (255, 255, 255), -1)
                cv2.putText(
                    depth_colored,
                    f"{depth_m:.2f} m",
                    (depth_u + 8, max(20, depth_v - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(rgb_display, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(
                rgb_display,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        detected = best_target is not None
        self.det_pub.publish(Bool(data=detected))

        if detected:
            self.handle_target_seen()
            self.handle_target(best_target, rgb_display, K, info_width, info_height, frame_id)
        else:
            self.update_lost_search_rotation()

        self.draw_overlay(rgb_display, len(detections), process_ms)
        combined = np.hstack((rgb_display, depth_colored))

        with self.lock:
            self.display_image = combined.copy()

        self.publish_images(rgb_display, depth_colored)

    def detect_targets(self, image):
        try:
            results = self.model.track(
                source=image,
                stream=False,
                persist=True,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO tracking failed: {exc}")
            return []

        detections = []
        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0]) if box.cls is not None else 0
                if self.target_class_ids is not None and cls not in self.target_class_ids:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                detections.append((x1, y1, x2, y2, conf, cls))

        return self.nms(detections)

    def nms(self, detections, iou_threshold=0.5):
        detections = sorted(detections, key=lambda d: d[4], reverse=True)
        kept = []
        for det in detections:
            x1, y1, x2, y2, _, _ = det
            suppressed = False
            for kx1, ky1, kx2, ky2, _, _ in kept:
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
                if union > 0 and inter / union > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(det)
        return kept

    def depth_at_rgb_pixel(self, depth, u, v, rgb_width, rgb_height):
        depth_height, depth_width = depth.shape[:2]
        depth_u = int(round(u * depth_width / max(1, rgb_width)))
        depth_v = int(round(v * depth_height / max(1, rgb_height)))
        depth_u = int(np.clip(depth_u, 0, depth_width - 1))
        depth_v = int(np.clip(depth_v, 0, depth_height - 1))

        radius = 4
        patch = depth[
            max(0, depth_v - radius): min(depth_height, depth_v + radius + 1),
            max(0, depth_u - radius): min(depth_width, depth_u + radius + 1),
        ].astype(np.float32)
        if patch.size == 0:
            return None, depth_u, depth_v

        if depth.dtype == np.uint16:
            patch *= 0.001

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid >= self.min_depth) & (valid <= self.max_depth)]
        if valid.size == 0:
            return None, depth_u, depth_v

        return float(np.median(valid)), depth_u, depth_v

    def handle_target(self, target, rgb_display, K, info_width, info_height, camera_frame):
        u = target["u"]
        v = target["v"]
        depth_m = target["depth_m"]
        height, width = rgb_display.shape[:2]

        point_camera = self.pixel_to_camera_point(
            u, v, depth_m, K, info_width, info_height, width, height, camera_frame
        )
        if point_camera is None:
            return

        try:
            point_map = self.tf_buffer.transform(
                point_camera,
                self.target_frame,
                timeout=Duration(seconds=1.0),
            )
        except TransformException as exc:
            self.throttled_warn(f"TF transform failed: {exc}")
            return

        self.target_pub.publish(point_map)
        goal_pose = self.make_standoff_goal(point_map)
        self.goal_pub.publish(goal_pose)
        self.goal_req_pub.publish(goal_pose)   # turtlebot_state → DRIVING_TURTLE
        self.maybe_send_goal(goal_pose)         # 기존 navigator 경로 (send_nav_goal=True 시)

        x1, y1, _, _, _, _ = target["detection"]
        label = f"{self.target_frame}({point_map.point.x:.2f}, {point_map.point.y:.2f})"
        cv2.putText(
            rgb_display,
            label,
            (x1, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    def pixel_to_camera_point(self, u, v, depth_m, K, info_width, info_height, rgb_width, rgb_height, frame_id):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        if fx == 0.0 or fy == 0.0:
            self.throttled_warn("Invalid camera intrinsics.")
            return None

        info_width = info_width or rgb_width
        info_height = info_height or rgb_height
        u_info = u * info_width / max(1, rgb_width)
        v_info = v * info_height / max(1, rgb_height)

        point = PointStamped()
        point.header.stamp = Time().to_msg()
        point.header.frame_id = frame_id
        point.point.x = (u_info - cx) * depth_m / fx
        point.point.y = (v_info - cy) * depth_m / fy
        point.point.z = depth_m
        return point

    def make_standoff_goal(self, point_map):
        robot_x, robot_y = self.lookup_robot_xy()
        target_x = point_map.point.x
        target_y = point_map.point.y

        if robot_x is None:
            goal_x = target_x
            goal_y = target_y
        else:
            dx = target_x - robot_x
            dy = target_y - robot_y
            distance = math.hypot(dx, dy)
            if distance > self.stop_distance:
                goal_x = target_x - dx / distance * self.stop_distance
                goal_y = target_y - dy / distance * self.stop_distance
            else:
                goal_x = robot_x
                goal_y = robot_y

        yaw = math.atan2(target_y - goal_y, target_x - goal_x)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.target_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        return goal_pose

    def lookup_robot_xy(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
            return transform.transform.translation.x, transform.transform.translation.y
        except TransformException as exc:
            self.throttled_warn(f"Robot pose TF failed: {exc}")
            return None, None

    def maybe_send_goal(self, goal_pose):
        if not self.send_nav_goal:
            return
        if not self.nav_ready:
            self.throttled_warn("Nav2 is not active yet. Publishing detected point only.")
            return
        if self.goal_running:
            if not self.navigator.isTaskComplete():
                return
            self.goal_running = False
        if time.monotonic() - self.last_goal_time < self.goal_cooldown:
            return

        self.navigator.goToPose(goal_pose)
        self.goal_running = True
        self.last_goal_time = time.monotonic()
        self.get_logger().info(
            f"Sent goal: x={goal_pose.pose.position.x:.2f}, y={goal_pose.pose.position.y:.2f}"
        )

    def handle_target_seen(self):
        self.last_target_time = time.monotonic()
        self.lost_target_start_time = 0.0
        self.search_rotation_done = False
        if self.search_rotation_active:
            self.stop_search_rotation("Target found. Stopping search rotation.")

    def update_lost_search_rotation(self):
        if not self.lost_search_rotation or not self.send_nav_goal:
            return
        if self.goal_running:
            if not self.navigator.isTaskComplete():
                return
            self.goal_running = False
        if self.search_angular_speed <= 0.0 or self.search_rotation_duration <= 0.0:
            return

        now = time.monotonic()
        if self.lost_target_start_time == 0.0:
            self.lost_target_start_time = now

        if self.search_rotation_active:
            if now - self.search_rotation_start >= self.search_rotation_duration:
                self.stop_search_rotation("Search rotation complete.")
                self.search_rotation_done = True
                return
            self.publish_search_rotation()
            return

        if self.search_rotation_done:
            return
        if now - self.lost_target_start_time < self.lost_search_delay:
            return

        self.search_rotation_active = True
        self.search_rotation_start = now
        self.get_logger().info(
            f"No usable {self.target_class} target. Rotating once to search."
        )
        self.publish_search_rotation()

    def publish_search_rotation(self):
        twist = Twist()
        twist.angular.z = self.search_angular_speed
        self.cmd_vel_pub.publish(twist)

    def stop_search_rotation(self, message=None):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.search_rotation_active = False
        if message:
            self.get_logger().info(message)

    def colorize_depth(self, depth):
        depth_display = depth.astype(np.float32)
        if depth.dtype == np.uint16:
            depth_display *= 0.001
        depth_display = np.nan_to_num(depth_display, nan=0.0, posinf=0.0, neginf=0.0)
        depth_display = np.clip(depth_display, 0.0, self.max_depth)
        depth_normalized = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    def publish_images(self, rgb_display, depth_colored):
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_display, encoding="bgr8")
        rgb_msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_pub.publish(rgb_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth_colored, encoding="bgr8")
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_pub.publish(depth_msg)

    def draw_overlay(self, image, detection_count, process_ms):
        lines = [
            f"target: {self.target_class}",
            f"detections: {detection_count} conf>={CONF_THRESHOLD}",
            f"infer: {process_ms:.1f} ms",
            f"nav goal: {'ready' if self.nav_ready else 'waiting' if self.send_nav_goal else 'off'}",
            f"search rotate: {'active' if self.search_rotation_active else 'ready' if not self.search_rotation_done else 'done'}",
        ]
        y = 24
        for line in lines:
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y += 24

    def gui_loop(self):
        if self.show_window:
            cv2.namedWindow("YOLO Car Navigator RGB | Depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO Car Navigator RGB | Depth", 1280, 480)

        while not self.gui_thread_stop.is_set():
            with self.lock:
                image = self.display_image.copy() if self.display_image is not None else None

            if self.show_window and image is not None:
                cv2.imshow("YOLO Car Navigator RGB | Depth", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.get_logger().info("Shutdown requested by user.")
                    self.shutdown_requested = True
                    self.gui_thread_stop.set()
                    rclpy.shutdown()
            elif self.show_window:
                placeholder = self.make_placeholder_image()
                cv2.imshow("YOLO Car Navigator RGB | Depth", placeholder)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    self.get_logger().info("Shutdown requested by user.")
                    self.shutdown_requested = True
                    self.gui_thread_stop.set()
                    rclpy.shutdown()
            else:
                cv2.waitKey(10)

    def make_placeholder_image(self):
        image = np.full((480, 1280, 3), 245, dtype=np.uint8)
        lines = [
            "Waiting for synchronized RGB + depth frames",
            f"RGB: {self.rgb_topic}",
            f"Depth: {self.depth_topic}",
            f"CameraInfo: {self.info_topic}",
            f"sync_slop: {self.sync_slop:.2f} sec",
        ]
        if self.K is None:
            lines.append("CameraInfo not received yet")
        if self.last_sync_time > 0.0:
            lines.append(f"last sync: {time.monotonic() - self.last_sync_time:.1f}s ago")

        y = 50
        for line in lines:
            cv2.putText(image, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
            y += 36
        return image

    def class_name(self, cls):
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls, str(cls))
        if 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return str(cls)

    def throttled_warn(self, message):
        now = time.monotonic()
        if now - self.last_warn_time >= 2.0:
            self.get_logger().warn(message)
            self.last_warn_time = now

    def destroy_node(self):
        if self.search_rotation_active:
            self.stop_search_rotation()
        self.gui_thread_stop.set()
        super().destroy_node()


def find_model_path():
    from ament_index_python.packages import get_package_share_directory
    try:
        share = Path(get_package_share_directory('webcam_detect'))
        candidate = share / 'resource' / 'best_turtlebot.pt'
        if candidate.exists():
            return candidate
    except Exception:
        pass

    # 소스 트리에서 실행할 때 fallback
    fallbacks = [
        Path(__file__).resolve().parent / 'resource' / 'best_turtlebot.pt',
        Path(__file__).resolve().parent.parent / 'resource' / 'best_turtlebot.pt',
    ]
    for path in fallbacks:
        if path.exists():
            return path
    return fallbacks[0]


def main():
    model_path = find_model_path()
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    print("Model loaded. Starting YOLO car navigator...")

    rclpy.init()
    node = YOLOCarNavigator(model)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.gui_thread_stop.set()
        node.gui_thread.join(timeout=2.0)
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
