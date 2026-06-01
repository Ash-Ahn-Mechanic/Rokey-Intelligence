import math
from typing import List, Tuple


def calculate_orbit_positions(
    obstacle_x: float,
    obstacle_y: float,
    radius: float,
    angle_step_deg: float,
) -> List[Tuple[float, float, float]]:
    """Return orbit positions around obstacle. Each item: (x, y, yaw)."""
    n = int(round(360.0 / angle_step_deg))
    positions = []
    for i in range(n):
        theta = math.radians(i * angle_step_deg)
        px = obstacle_x + radius * math.cos(theta)
        py = obstacle_y + radius * math.sin(theta)
        yaw = math.atan2(obstacle_y - py, obstacle_x - px)
        positions.append((px, py, yaw))
    return positions


def filter_by_costmap(
    positions: List[Tuple[float, float, float]],
    costmap_data: List[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> List[Tuple[float, float, float]]:
    """Filter positions to only FREE cells (cost==0) in costmap."""
    valid = []
    for (x, y, yaw) in positions:
        cell_x = int((x - origin_x) / resolution)
        cell_y = int((y - origin_y) / resolution)
        if 0 <= cell_x < width and 0 <= cell_y < height:
            idx = cell_y * width + cell_x
            if costmap_data[idx] == 0:
                valid.append((x, y, yaw))
    return valid
