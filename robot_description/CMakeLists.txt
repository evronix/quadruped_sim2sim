cmake_minimum_required(VERSION 3.5)
project(robot_description)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch meshes rviz urdf 
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
