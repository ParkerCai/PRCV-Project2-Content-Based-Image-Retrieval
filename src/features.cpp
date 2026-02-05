/*
  Parker Cai
  February 2, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  Implementation of feature extraction functions for CBIR.
*/

#include "features.h"
#include <iostream>
#include <print>  // for modern C++ printing (C++23)


/*
  Extract Baseline Features from the image

  Use the 7x7 square in the middle of the image as a feature vector.

  Input:
    src - input image (cv::Mat), using const to prevent modifying src img (safety)
    features - output feature vector (std::vector<float>) (list of 7*7*3 = 147 values)
*/
int extractBaselineFeatures(const cv::Mat& src, std::vector<float>& features) {
  // Edge case: image too small
  if (src.rows < 7 || src.cols < 7) {
    std::println(stderr, "Error: Image too small for baseline feature extraction");
    return -1;
  }

  // find the center of the image
  int centerX = src.cols / 2;
  int centerY = src.rows / 2;

  // Clear the features vector before writing
  features.clear();

  // extract 7x7 square around the center
  for (int i = -3; i <= 3; i++) {
    for (int j = -3; j <= 3; j++) {
      // get pixel at (centerY + i, centerX + j)
      cv::Vec3b pixel = src.at<cv::Vec3b>(centerY + i, centerX + j);
      // store B, G, R values as floats to the feature vector
      // casting uchar to float w/ C++ static_cast (explicit)
      features.push_back(static_cast<float>(pixel[0])); // B
      features.push_back(static_cast<float>(pixel[1])); // G
      features.push_back(static_cast<float>(pixel[2])); // R
    }
  }

  return 0;
}
