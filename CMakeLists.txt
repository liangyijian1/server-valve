cmake_minimum_required(VERSION 3.28)
project(server)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:/Program Files/PCL 1.14.0/3rdParty/opencv4.9/build")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# hk
include_directories("C:/Program Files/PCL 1.14.0/3rdParty/hk/Include")

add_executable(server
        main.cpp
        _utils.cpp
        _utils.h
        cmdline.h)

target_link_libraries(server ${OpenCV_LIBS})
target_link_libraries(server "C:/Program Files/PCL 1.14.0/3rdParty/hk/Libs/win64/MvCameraControl.lib")