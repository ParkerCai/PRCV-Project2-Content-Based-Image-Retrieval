# Project 2: Content-based Image Retrieval

**CS5330 Pattern Recognition & Computer Vision**


## Team

- GitHub: Parker Cai - [@parkercai (https://github.com/ParkerCai)] 

- GitHub: Jenny Nguyen - [@jennyncodes (https://github.com/jennyncodes)]

## Overview

Build a content-based image retrieval (CBIR) system that finds images in a database with similar content to a target image.

## Project Description

Tasks (from Prof. Maxwell's project description)

For this task the inputs to the system will be a target image T, an image database B, a method of computing features for an image F, a distance metric for comparing the image features from two images D(Ft,Fi), and the desired number of output images N. The output will be an ordered list of the N images in B that are most similar to T, according to F and D, in ascending order of distance. Remember, the smaller a distance, the more similar two images will be.

The process can be generally described as the following four steps.

1. Compute the features Ft on the target image T.
2. Compute the features {Fi} on all of the images in B.
3. Compute the distance of T from all of the images in B using the distance metric D(Ft,Fi).
4. Sort the images in B according to their distance from T and return the best N matches.

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
|   ├── cbir.cpp            # Main Program
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
- **Distance Metric**: Sum of Squared Difference (SSD)
- **Testing**: .\bin\cbir.exe data\olympus\pic.1016.jpg data\olympus baseline

### Task 2: Histogram Matching

- **Feature**: 2D color histogram
- **Distance Metric**: Histogram Intersection
- **Testing**: .\bin\cbir.exe data\olympus\pic.0164.jpg data\olympus histogram

### Task 3: Mutli-histogram Matching

- **Feature**: Two RGB histograms (8x8x8 bins) - top half and bottom half
- **Distance Metric**: Histogram Intersection with equal weighting (50% top, 50% bottom)
- **Testing**: .\bin\cbir.exe data\olympus\pic.0274.jpg data\olympus multihistogram

