cmake_minimum_required(VERSION 3.10)
project(ground_segmentation)

# -------------------------------
# 1. Compiler and optimization setup
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3 -march=native -ffast-math -funroll-loops)

# -------------------------------
# 2. Find and include dependencies
# -------------------------------
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters segmentation visualization features kdtree)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# -------------------------------
# 3. OpenMP (for parallelism)
# -------------------------------
find_package(OpenMP REQUIRED)

# -------------------------------
# 4. Build the executable
# -------------------------------
add_executable(center center.cpp)

# -------------------------------
# 5. Link all required libraries
# -------------------------------
target_link_libraries(center 
    ${PCL_LIBRARIES} 
    ${YAML_CPP_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# -------------------------------
# 6. Install target (optional)
# -------------------------------
install(TARGETS center DESTINATION bin)
