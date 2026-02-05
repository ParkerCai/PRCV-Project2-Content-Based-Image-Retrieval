cmake_minimum_required(VERSION 3.24)
project(cbir)  # Content-based Image Retrieval

# Set C++ standard to C++23 for modern features like println
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)      # Your headers
include_directories(${CMAKE_SOURCE_DIR}/csv_util)        # csv_util

# Output executables to bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/../bin)

# --- Core source files ---
# TODO: Add source files here as you implement them
set(SOURCES
    features.cpp
    distance.cpp
    csv_util/csv_util.cpp    # csv_util
)

# --- Executables ---
# TODO: Add new executables and/or modify existing as you build your programs

# CBIR main program
add_executable(cbir cbir.cpp ${SOURCES})
target_link_libraries(cbir ${OpenCV_LIBS})

# Example: Build feature database
# add_executable(build_db build_db.cpp ${SOURCES})
# target_link_libraries(build_db ${OpenCV_LIBS})
