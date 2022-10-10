cmake_minimum_required(VERSION 3.0.2)
project(gp_kalmann)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "-fPIC")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS 
  roscpp
)
find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} )

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
INCLUDE_DIRS include  ${EIGEN3_INCLUDE_DIR}
LIBRARIES  ${PROJECT_NAME}
CATKIN_DEPENDS roscpp 
)


add_executable(gp_kalmann 
	src/GP_kalmann.cpp 
	src/demo.cpp
)

target_link_libraries(gp_kalmann
   ${catkin_LIBRARIES}  
)

add_dependencies(gp_kalmann  ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS} )
