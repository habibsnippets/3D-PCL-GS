cmake_minimum_required(VERSION 3.0.0)
project(ground_segmentation)

# Find PCL
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters segmentation visualization features kdtree)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# Set C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(center center.cpp)

# Link libraries
target_link_libraries(center ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})

# Install targets
install(TARGETS center DESTINATION bin)
