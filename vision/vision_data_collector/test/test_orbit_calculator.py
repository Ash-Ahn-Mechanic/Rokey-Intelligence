import math
import pytest
from vision_data_collector.orbit_calculator import calculate_orbit_positions, filter_by_costmap


def test_orbit_count_360_45deg():
    positions = calculate_orbit_positions(0.0, 0.0, 1.0, 45.0)
    assert len(positions) == 8


def test_orbit_count_360_90deg():
    positions = calculate_orbit_positions(0.0, 0.0, 1.0, 90.0)
    assert len(positions) == 4


def test_orbit_radius():
    positions = calculate_orbit_positions(0.0, 0.0, 2.0, 90.0)
    for (x, y, yaw) in positions:
        dist = math.sqrt(x**2 + y**2)
        assert abs(dist - 2.0) < 1e-6


def test_orbit_yaw_faces_obstacle():
    # obstacle at (1,1), radius 1, angle 0 → robot at (2,1), yaw faces left (π)
    positions = calculate_orbit_positions(1.0, 1.0, 1.0, 360.0)
    assert len(positions) == 1
    x, y, yaw = positions[0]
    assert abs(x - 2.0) < 1e-6
    assert abs(y - 1.0) < 1e-6
    expected_yaw = math.atan2(1.0 - 1.0, 1.0 - 2.0)  # π
    assert abs(yaw - expected_yaw) < 1e-6


def test_filter_free_cells_only():
    positions = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)]
    # costmap: 2x1, cell(0,0)=0 (free), cell(1,0)=100 (obstacle)
    costmap_data = [0, 100]
    result = filter_by_costmap(
        positions,
        costmap_data=costmap_data,
        width=2, height=1,
        resolution=1.0,
        origin_x=0.0, origin_y=0.0
    )
    assert len(result) == 1
    assert abs(result[0][0] - 0.5) < 1e-6


def test_filter_out_of_bounds():
    positions = [(-1.0, -1.0, 0.0)]
    costmap_data = [0]
    result = filter_by_costmap(
        positions,
        costmap_data=costmap_data,
        width=1, height=1,
        resolution=1.0,
        origin_x=0.0, origin_y=0.0
    )
    assert len(result) == 0
