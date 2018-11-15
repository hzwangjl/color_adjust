cmake_minimum_required(VERSION 2.8)
project(ADJUST)

SET(OPENCV_PATH "/home/hzwangjialei/commlibs/opencv-2.4.13")

SET(OPENCV_INC "${OPENCV_PATH}/include" "${OPENCV_PATH}/include/opencv" "${OPENCV_PATH}/include/opencv2" )
SET(OPENCV_LIB "${OPENCV_PATH}/lib" )

SET( LIB_INCS "inc")
file( GLOB CPP_SRCS "src/*.cc")
file( GLOB CPP_INC "inc/*.h")
file( GLOB TEST_SRCS ${CPP_SRCS} "test/test.cc" )


INCLUDE_DIRECTORIES(${OPENCV_INC} ${LIB_INCS})
LINK_DIRECTORIES(${OPENCV_LIB})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -D__STDC_CONSTANT_MACROS" )

add_executable(color_test ${TEST_SRCS})
target_link_libraries(color_test -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d) 

