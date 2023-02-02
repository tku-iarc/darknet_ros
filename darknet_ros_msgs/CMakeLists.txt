cmake_minimum_required(VERSION 3.8)

project(darknet_ros_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  std_msgs
  sensor_msgs
  geometry_msgs
  actionlib_msgs
  rosidl_default_generators
)

find_package(ament_cmake REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/BoundingBoxes.msg"
  "msg/ObjectCount.msg"
  "action/CheckForObjects.action"
DEPENDENCIES
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()