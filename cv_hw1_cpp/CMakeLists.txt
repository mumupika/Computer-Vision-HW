cmake_minimum_required(VERSION 3.6) # 最低版本
project(program1)    # 示例

find_package(OpenCV REQUIRED)

include_directories("${OpenCV_INCLUDE_DIRS}")

add_executable(program1 p1_object_attributes.cpp)
target_link_libraries(program1 "${OpenCV_LIBRARIES}")
set_property(TARGET program1 PROPERTY CXX_STANDARD 17)