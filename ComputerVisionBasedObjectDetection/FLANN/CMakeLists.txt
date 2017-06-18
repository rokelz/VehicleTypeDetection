cmake_minimum_required(VERSION 2.8.12)

project(FLANN)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FLANN demo.cpp)
target_link_libraries(FLANN ${OpenCV_LIBS})
