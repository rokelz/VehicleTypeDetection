cmake_minimum_required(VERSION 2.8.12)

project(ORB)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ORB demo.cpp)
target_link_libraries(ORB ${OpenCV_LIBS})
