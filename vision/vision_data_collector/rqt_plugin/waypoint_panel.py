import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from geometry_msgs.msg import Point

from python_qt_binding.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QDoubleSpinBox, QListWidget, QSplitter, QGroupBox,
)
from python_qt_binding.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from python_qt_binding.QtCore import Qt, QPoint, pyqtSignal
import numpy as np

from rqt_gui_py.plugin import Plugin
from vision_interfaces.srv import StartMission


class MapWidget(QLabel):
    waypoint_added = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._map_msg = None
        self._waypoints = []
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setText('맵 대기 중...')

    def update_map(self, msg):
        self._map_msg = msg
        self._render()

    def clear_waypoints(self):
        self._waypoints.clear()
        self._render()

    def mousePressEvent(self, event):
        if self._map_msg is None:
            return
        map_x, map_y = self._pixel_to_map(event.x(), event.y())
        if map_x is not None:
            self._waypoints.append((map_x, map_y))
            self._render()
            self.waypoint_added.emit(map_x, map_y)

    def _pixel_to_map(self, px, py):
        if self._map_msg is None:
            return None, None
        info = self._map_msg.info
        w, h = info.width, info.height
        lw, lh = self.width(), self.height()
        scale = min(lw / w, lh / h)
        ox = (lw - w * scale) / 2
        oy = (lh - h * scale) / 2
        cell_x = int((px - ox) / scale)
        cell_y = int((py - oy) / scale)
        cell_y_map = h - 1 - cell_y
        if not (0 <= cell_x < w and 0 <= cell_y_map < h):
            return None, None
        map_x = info.origin.position.x + cell_x * info.resolution
        map_y = info.origin.position.y + cell_y_map * info.resolution
        return map_x, map_y

    def _map_to_pixel(self, map_x, map_y):
        if self._map_msg is None:
            return None, None
        info = self._map_msg.info
        w, h = info.width, info.height
        lw, lh = self.width(), self.height()
        scale = min(lw / w, lh / h)
        ox = (lw - w * scale) / 2
        oy = (lh - h * scale) / 2
        cell_x = (map_x - info.origin.position.x) / info.resolution
        cell_y_map = (map_y - info.origin.position.y) / info.resolution
        cell_y = h - 1 - cell_y_map
        return int(cell_x * scale + ox), int(cell_y * scale + oy)

    def _render(self):
        if self._map_msg is None:
            return
        info = self._map_msg.info
        w, h = info.width, info.height
        data = np.array(self._map_msg.data, dtype=np.int8).reshape(h, w)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data == 0] = [255, 255, 255]
        img[data == 100] = [0, 0, 0]
        img[data == -1] = [128, 128, 128]

        img_flipped = np.flipud(img)
        qimg = QImage(img_flipped.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(6)
        painter.setPen(pen)
        for (mx, my) in self._waypoints:
            px, py = self._map_to_pixel(mx, my)
            if px is not None:
                painter.drawEllipse(QPoint(px, py), 5, 5)
        painter.end()
        self.setPixmap(pixmap)


class WaypointPanel(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('WaypointPanel')

        if not rclpy.ok():
            rclpy.init()
        self._node = Node('waypoint_panel_node')

        self._widget = QWidget()
        self._build_ui()
        context.add_widget(self._widget)

        self._node.create_subscription(OccupancyGrid, '/robot4/map', self._map_callback, 10)
        self._node.create_subscription(String, '/vision/mission_status', self._status_callback, 10)
        self._start_client = self._node.create_client(StartMission, '/vision/start_mission')

        from threading import Thread
        self._spin_thread = Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

    def _build_ui(self):
        layout = QHBoxLayout(self._widget)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self._map_widget = MapWidget()
        self._map_widget.waypoint_added.connect(self._on_waypoint_added)
        splitter.addWidget(self._map_widget)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        param_group = QGroupBox('파라미터')
        param_layout = QVBoxLayout(param_group)

        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.3, 5.0)
        self._radius_spin.setValue(1.0)
        self._radius_spin.setSingleStep(0.1)
        self._radius_spin.setSuffix(' m')
        param_layout.addWidget(QLabel('반지름:'))
        param_layout.addWidget(self._radius_spin)

        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(10.0, 180.0)
        self._angle_spin.setValue(45.0)
        self._angle_spin.setSingleStep(5.0)
        self._angle_spin.setSuffix(' °')
        param_layout.addWidget(QLabel('각도 간격:'))
        param_layout.addWidget(self._angle_spin)

        self._save_dir_edit = QLineEdit()
        self._save_dir_edit.setText(os.path.expanduser('~/dataset'))
        param_layout.addWidget(QLabel('저장 경로:'))
        param_layout.addWidget(self._save_dir_edit)
        right_layout.addWidget(param_group)

        wp_group = QGroupBox('웨이포인트 목록')
        wp_layout = QVBoxLayout(wp_group)
        self._waypoint_list = QListWidget()
        wp_layout.addWidget(self._waypoint_list)
        clear_btn = QPushButton('전체 삭제')
        clear_btn.clicked.connect(self._clear_waypoints)
        wp_layout.addWidget(clear_btn)
        right_layout.addWidget(wp_group)

        self._start_btn = QPushButton('미션 시작')
        self._start_btn.clicked.connect(self._start_mission)
        right_layout.addWidget(self._start_btn)

        self._status_label = QLabel('대기 중')
        self._status_label.setWordWrap(True)
        right_layout.addWidget(self._status_label)

        splitter.addWidget(right)
        splitter.setSizes([600, 300])

    def _map_callback(self, msg):
        self._map_widget.update_map(msg)

    def _on_waypoint_added(self, map_x, map_y):
        idx = self._waypoint_list.count()
        self._waypoint_list.addItem(f'WP{idx}: ({map_x:.2f}, {map_y:.2f})')

    def _clear_waypoints(self):
        self._waypoint_list.clear()
        self._map_widget.clear_waypoints()

    def _status_callback(self, msg):
        self._status_label.setText(msg.data)

    def _start_mission(self):
        if not self._start_client.service_is_ready():
            self._status_label.setText('mission_node 연결 안됨')
            return

        waypoints = []
        for i in range(self._waypoint_list.count()):
            text = self._waypoint_list.item(i).text()
            coords = text.split('(')[1].rstrip(')').split(', ')
            p = Point()
            p.x = float(coords[0])
            p.y = float(coords[1])
            p.z = 0.0
            waypoints.append(p)

        if not waypoints:
            self._status_label.setText('웨이포인트 없음')
            return

        req = StartMission.Request()
        req.waypoints = waypoints
        req.radius = self._radius_spin.value()
        req.angle_step_deg = self._angle_spin.value()
        req.save_dir = self._save_dir_edit.text()

        self._start_btn.setEnabled(False)
        self._status_label.setText('미션 전송 중...')
        future = self._start_client.call_async(req)
        future.add_done_callback(self._on_mission_response)

    def _on_mission_response(self, future):
        result = future.result()
        if result and result.success:
            self._status_label.setText(f'완료: {result.message}')
        else:
            msg = result.message if result else 'No response'
            self._status_label.setText(f'실패: {msg}')
        self._start_btn.setEnabled(True)

    def shutdown_plugin(self):
        self._node.destroy_node()
