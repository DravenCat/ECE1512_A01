cmake_minimum_required(VERSION 4.0)
project(A01)

set(CMAKE_CXX_STANDARD 20)
set(OpenCV_DIR "C:\\personal\\abroad\\master\\fall\\ECE1512\\opencv\\opencv\\mingw_build\\install")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(A01 main.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(A01 ${OpenCV_LIBS})