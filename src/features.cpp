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


/*
  Extract 2D RG Chromaticity Histogram from the image

  Chromaticity removes intensity information, keeping only color ratios:
    r = R / (R + G + B)
    g = G / (R + G + B)
  This makes the feature more robust to lighting changes and intensity variations.

  Input:
    src - input image (cv::Mat), using const to prevent modifying src img (safety)
    features - output histogram as flattened vector (bins * bins values)
    bins - number of bins for each dimension (default 16 bins)
*/
int extractRGChromHistogram(const cv::Mat& src, std::vector<float>& features, int bins) {
  // Edge case: empty image
  if (src.empty()) {
    std::println(stderr, "Error: Empty image for histogram extraction");
    return -1;
  }

  // Create 2D histogram (bins x bins) initialized to zero
  cv::Mat histogram = cv::Mat::zeros(bins, bins, CV_32F);

  // Iterate through all pixels using row pointers for efficiency
  for (int i = 0; i < src.rows; i++) {
    const cv::Vec3b* rowPtr = src.ptr<cv::Vec3b>(i);  // pointer to row i
    for (int j = 0; j < src.cols; j++) {
      // get the RGB values
      float B = rowPtr[j][0];
      float G = rowPtr[j][1];
      float R = rowPtr[j][2];

      // compute divisor, handle black pixels
      float divisor = R + G + B;
      divisor = divisor > 0.0f ? divisor : 1.0f; // avoid divide by zero

      // compute rg chromaticity
      float r = R / divisor; // r and g are in [0, 1] range
      float g = G / divisor;

      // compute bin indices with proper rounding (+0.5)
      int rIndex = static_cast<int>(r * (bins - 1) + 0.5f);
      int gIndex = static_cast<int>(g * (bins - 1) + 0.5f);

      // increment histogram bin
      histogram.at<float>(rIndex, gIndex) += 1.0f;
    }
  }

  // Flatten 2D histogram to 1D feature vector (raw counts)
  // normalization is done during histogram intersection
  features.clear(); // clear the features vector before writing
  features.reserve(bins * bins); // reserve space to avoid multiple reallocations
  
  // loop through histogram and push values to features vector
  for (int i = 0; i < bins; i++) { // r bin
    // 
    const float* histRowPtr = histogram.ptr<float>(i);
    for (int j = 0; j < bins; j++) { // g bin
      features.push_back(histRowPtr[j]); // push count for the (r, g) bin
    }
  }

  return 0;
}

/*
  Multi-Histogram Features (Task 3)
  - Splits image into top and bottom halves
  - Computes RGB histogram for each half separately
  - Uses 8 bins per channel (R, G, B): 8x8x8 = 512 bins per histogram
  - Total feature vector: 1024 values (512 top + 512 bottom)
  - Captures both color distribution and spatial layout

  Input:
    src - input image (cv::Mat)
    features - output feature vector (std::vector<float>)

  Output:
    int - 0 on success, -1 on error
*/
int extractMultiHistogram(const cv::Mat& src, std::vector<float>& features) {
  if (src.empty()) return -1;
  
  features.clear();
  int bins = 8;
  int midRow = src.rows / 2;
  
  // top and bottom halves
  cv::Mat topHalf = src(cv::Rect(0, 0, src.cols, midRow));
  cv::Mat bottomHalf = src(cv::Rect(0, midRow, src.cols, src.rows - midRow));
  
  cv::Mat halves[] = {topHalf, bottomHalf};
  
  for (int h = 0; h < 2; h++) {
    std::vector<float> hist(512, 0);
    
    for (int y = 0; y < halves[h].rows; y++) {
      for (int x = 0; x < halves[h].cols; x++) {
        cv::Vec3b pixel = halves[h].at<cv::Vec3b>(y, x);
        int r = pixel[2] * bins / 256;
        int g = pixel[1] * bins / 256;
        int b = pixel[0] * bins / 256;
        if (r >= bins) r = bins - 1;
        if (g >= bins) g = bins - 1;
        if (b >= bins) b = bins - 1;
        hist[r * 64 + g * 8 + b]++;
      }
    }
    
    for (int i = 0; i < 512; i++) {
      features.push_back(hist[i]);
    }
  }
  return 0;
}
  