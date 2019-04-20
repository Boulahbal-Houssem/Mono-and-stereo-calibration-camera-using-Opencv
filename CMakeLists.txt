cmake_minimum_required(VERSION 2.8)

project( Calib )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( calib calib.cpp )
add_executable( stereo stereo.cpp )

target_link_libraries( calib ${OpenCV_LIBS} )
target_link_libraries( stereo ${OpenCV_LIBS} )
