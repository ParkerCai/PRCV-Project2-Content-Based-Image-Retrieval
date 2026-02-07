cmake_minimum_required(VERSION 3.10)
project(CBIR_Project2)

# Use C++23 for std::println support
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV - it will auto-detect from your system
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/features.cpp
    src/distance.cpp
    src/csv_util/csv_util.cpp
)

# Main CBIR executable
add_executable(cbir 
    src/cbir.cpp
    ${SOURCES}
)

target_link_libraries(cbir ${OpenCV_LIBS})

# Disable PDB to avoid linker limit on large projects
if(MSVC)
    target_link_options(cbir PRIVATE /DEBUG:NONE)
endif()

# Output to bin folder
set_target_properties(cbir PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin
)