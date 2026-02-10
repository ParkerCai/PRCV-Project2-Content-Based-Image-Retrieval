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
├── release/                # Distribution packages (auto-generated)
├── data/   
│   ├── olympus/            # Image database (1107 images)
│   └── ResNet18_olym.csv   # DNN embeddings for olympus images
├── features/               # Pre-computed feature CSV files
├── include/                # Header files
│   ├── features.h          # Feature extraction declarations
│   └── distance.h          # Distance metric declarations
├── src/                    # Source files
│   ├── CMakeLists.txt      # Build configuration
│   ├── cbir.cpp            # CLI program
│   ├── features.cpp        # Feature extraction implementation
│   ├── distance.cpp        # Distance metrics implementation
│   ├── csv_util/           # CSV utilities
│   │   ├── csv_util.h
│   │   └── csv_util.cpp
│   └── gui/                # Dear ImGui GUI application
│       ├── cbir_gui.cpp    # GUI main source
│       ├── app_icon.ico    # Windows executable icon
│       └── app_icon.rc     # Windows resource file
├── report/                 # Report documents
├── build.bat               # Build script
├── package_release.bat     # Release packaging script
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

### Task 4: Texture and Color

- **Feature**: Sobel gradient magnitude histogram (16 bins) + RGB color histogram (8x8x8 bins)
- **Distance Metric**: Histogram Intersection with equal weighting (50% texture, 50% color)
- **Testing**: .\bin\cbir.exe data\olympus\pic.0535.jpg data\olympus textureandcolor

### Task 5: DNN Embedding Matching

- **Feature**: DNN embedding (512-dimensional vector)
- **Distance Metric**: Cosine distance: d(vA, vB) = 1 - cos(theta) 
- **Testing1(fire hydrant)**: .\bin\cbir.exe data\olympus\pic.0893.jpg data\olympus dnnembedding data\ResNet18_olym.csv
- **Testing2**: .\bin\cbir.exe data\olympus\pic.0164.jpg data\olympus dnnembedding data\ResNet18_olym.csv


### Task 7: Custom Distance for Portrait Matching

- **Feature**: Combined feature vector (529 dimensions)
  - DNN embedding (512 values)
  - Skin tone histogram (16 bins)
  - Brightness value (1 value)
- **Distance Metric**: Weighted combination
  - 70% DNN cosine distance
  - 20% Skin tone histogram intersection
  - 10% Brightness absolute difference
- **Testing**: 
  - Portrait: .\bin\cbir.exe data\olympus\pic.0607.jpg data\olympus custom
  - Basketball: .\bin\cbir.exe data\olympus\pic.0280.jpg data\olympus custom

### Extension: Orientation Gradient Histogram

- **Feature**: 8-bin histogram of edge directions (8 dimensions)
- **Distance Metric**: Histogram Intersection
- **Testing**: .\bin\cbir.exe data\olympus\pic.0462.jpg data\olympus gradient

### Extension: GUI

- **Framework**: Dear ImGui with GLFW + OpenGL2 backend
- **Features**:
  - Drag-and-drop image loading for query input
  - Browse and switch image database directories on the fly
  - Dropdown selection for all feature types and distance metrics
  - Side-by-side visual comparison of query image and ranked results
  - Resizable split panel layout
  - Keyboard shortcuts (Enter to search, Q to quit)
- **Run**: `.\bin\cbir_gui.exe`
