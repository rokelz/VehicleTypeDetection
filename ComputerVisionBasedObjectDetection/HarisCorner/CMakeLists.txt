cmake_minimum_required(VERSION 2.8.12)

project(HarisCorner)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HarisCorner demo.cpp)
target_link_libraries(HarisCorner ${OpenCV_LIBS})
