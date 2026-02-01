# Project 2: Content-based Image Retrieval

CS5330 Pattern Recognition & Computer Vision  

## Overview

Build a content-based image retrieval (CBIR) system that finds images in a database with similar content to a target image.

## Directory Structure

```
Project_2_Content_based_Image_Retrieval/
├── bin/                    # Executables (auto-generated)
├── build/                  # CMake build files (auto-generated)
├── data/                   
│   └── olympus/            # Image database (1107 images)
├── features/               # Pre-computed feature CSV files
├── include/                # Header files
│   ├── features.h          # Feature extraction declarations (prototypes)
│   └── distance.h          # Distance metric declarations (prototypes)
├── src/                    # Source files
│   ├── CMakeLists.txt      # Build configuration
│   ├── CMakePresets.json   # CMake presets for VS Code
│   ├── features.cpp        # TODO: Implement feature extraction
│   ├── distance.cpp        # TODO: Implement distance metrics
│   ├── csv_util/           # CSV utilities
│   │   ├── csv_util.h
│   │   └── csv_util.cpp
│   └── readfiles.cpp       # Instructor's directory reading example
├── report/                 # Report documents (gdocs and pdf)
└── README.md
```

## Build Instructions

### Option 1: VS Code with CMake Tools

1. Open the `src/` folder in VS Code
2. CMake Tools will auto-detect `CMakePresets.json`
3. Click **Build** button in the status bar at the bottom

### Option 2: Command Line

```bash
cd src
cmake --preset default
cmake --build --preset release
```

### Option 3: Manual (Most reliable, just run the ./build.bat file at root)

```bash
cd Assignment1/src
mkdir build
cd build
cmake -Wno-dev -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:\opencv_build\build\install" ../src
cmake --build . --config Release
```

## Assignment Tasks

### Task 1: Baseline Matching
- **Feature**: 7x7 center pixel values

### Task 2: Histogram Matching
- **Feature**: 2D color histogram
