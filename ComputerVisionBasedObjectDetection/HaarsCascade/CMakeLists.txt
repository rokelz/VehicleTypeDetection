cmake_minimum_required(VERSION 2.8.12)

project(HaarsCascade)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HaarsCascade demo.cpp)
target_link_libraries(HaarsCascade ${OpenCV_LIBS})
