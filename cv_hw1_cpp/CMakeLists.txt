cmake_minimum_required(VERSION 3.6) # 最低版本
project(program1)    # 示例
project(program2)

find_package(OpenCV REQUIRED)

include_directories("${OpenCV_INCLUDE_DIRS}")

add_executable(program1 p1_object_attributes.cpp)
add_executable(program2 p2_hough_circles.cpp)

target_link_libraries(program1 "${OpenCV_LIBRARIES}")
target_link_libraries(program2 "${OpenCV_LIBRARIES}")

set_property(TARGET program1 PROPERTY CXX_STANDARD 17)
set_property(TARGET program2 PROPERTY CXX_STANDARD 17)