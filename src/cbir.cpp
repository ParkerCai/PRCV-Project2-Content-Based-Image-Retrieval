/*
  Parker Cai
  February 3, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  CBIR main program - finds similar images using baseline features
*/


#include <iostream>
#include <vector>
#include <string>
#include <print>  // for modern C++ printing (C++23)
#include <filesystem>  // for directory traversal (cross-platform)
#include <opencv2/opencv.hpp>
#include "features.h"
#include "distance.h"
// TODO: include csv_util for reading directory

enum CBIRExitCode {
  Success = 0,
  MissingArg = 1,
  ImageLoadFailed = 2
};

// Helper function to check if a file is an image based on extension
bool isImageFile(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  return ext == ".jpg" || ext == ".png" || ext == ".ppm" || ext == ".tif";
}


/*
  Standard main function with command line arguments for
  Content-based Image Retrieval using baseline features.
  Usage:
  ./cbir.exe ./query_image.jpg ./image_database_directory/
*/
int main(int argc, char* argv[]) {
  // 1. parse command line arguments
  // Error handling for missing arguments
  if (argc < 3) {
    std::println("Usage: {} <query_image> <image_database_directory>", argv[0]);
    exit(MissingArg);  // exit with error code
  }

  cv::Mat src;  // read image from file specified in command line argument
  std::string imageDir = argv[2];  // image directory path

  // read and load the query image
  src = cv::imread(argv[1]);

  // 2. Read directory
  std::vector<std::string> imageFiles;
  for (const auto& entry : std::filesystem::directory_iterator(imageDir)) {
    if (entry.is_regular_file() && isImageFile(entry.path())) {
      imageFiles.push_back(entry.path().string());
    }
  }

  // 3. Extract features from query image
  std::vector<float> queryFeatures;
  int status = extractBaselineFeatures(src, queryFeatures);

  // Error handling for feature extraction failure
  if (status != 0) {
    std::println(stderr, "Error: Failed to extract features from query image");
    exit(ImageLoadFailed);
  }

  // 4. Sort images by distance
  std::vector<std::pair<float, std::string>> distances;

  // iterate through all images in the directory
  for (const auto& imageFile : imageFiles) {
    cv::Mat image = cv::imread(imageFile);

    // Error handling for image loading failure
    if (image.empty()) {
      std::println(stderr, "Error: Failed to load image {}", imageFile);
      continue;
    }

    std::vector<float> features;
    int status = extractBaselineFeatures(image, features);

    // Error handling for feature extraction failure
    if (status != 0) {
      std::println(stderr, "Error: Failed to extract features from image {}", imageFile);
      continue;
    }

    // compute distance and store it with the filename
    float distance = sumOfSquaredDifference(queryFeatures, features);
    distances.push_back(std::make_pair(distance, imageFile)); // {distance, filename} pair
  }
  std::sort(distances.begin(), distances.end());

  // 5. Display top 4 results
  std::println("Top 4 similar images:");
  for (int i = 0; i < 4 && i < (int)distances.size(); i++) {
    std::filesystem::path p(distances[i].second); // get filename from path
    std::println("{}: {} (distance: {})", i + 1, p.filename().string(), distances[i].first);
  }

  return Success;
}
