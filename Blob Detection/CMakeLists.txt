cmake_minimum_required(VERSION 2.8.12)

project(src)





find_package(OpenCV REQUIRED)


file(GLOB src kalmanTest2.cpp)


file(GLOB src background.cpp)
file(GLOB src foreground.cpp)
file(GLOB src KalmanProcess.cpp)
file(GLOB src NewKalman.cpp)
file(GLOB src struct.h)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(src kalmanTest2.cpp)
target_link_libraries(src ${OpenCV_LIBS})
