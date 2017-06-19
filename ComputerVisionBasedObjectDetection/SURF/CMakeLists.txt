cmake_minimum_required(VERSION 2.8.12)

project(SURF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SURF demo.cpp)
target_link_libraries(SURF ${OpenCV_LIBS})
