cmake_minimum_required(VERSION 3.8)
project(vision_data_collector)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/StartMission.srv"
  "srv/CaptureImage.srv"
  DEPENDENCIES geometry_msgs
)

# Note: ament_python_install_package conflicts with rosidl_generate_interfaces
# The Python package is still installed via rosidl's Python generation
# ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  vision_data_collector/mission_node.py
  vision_data_collector/camera_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config maps rqt_plugin
  DESTINATION share/${PROJECT_NAME}
)

install(FILES rqt_plugin/plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY resource
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
