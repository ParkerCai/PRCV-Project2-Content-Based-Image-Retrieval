/*
  Parker Cai
  Jenny Nguyen
  February 2, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  Implementation of feature extraction functions for CBIR.
*/

#include "features.h"
#include <iostream>
#include <print>  // for modern C++ printing (C++23)
#include "csv_util/csv_util.h"
#include <filesystem>


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
  Extract 2D Histogram over RG Chromaticity from the image

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
    const float* histRowPtr = histogram.ptr<float>(i);
    for (int j = 0; j < bins; j++) { // g bin
      features.push_back(histRowPtr[j]); // push count for the (r, g) bin
    }
  }

  return 0;
}


/*
  Extract 3D Histogram over RGB Chromaticity from the image

  Chromaticity removes intensity information, keeping only color ratios:
    r = R / (R + G + B)
    g = G / (R + G + B)
    b = 1 - (r + g)

    r + g + b = 1, so we only need two dimensions (r and g) to represent the chromaticity.

  Input:
    src - input image (cv::Mat), using const to prevent modifying src img (safety)
    features - output histogram as flattened vector (bins * bins values)
    bins - number of bins for each dimension (default 16 bins)
*/
int extractRGBChromHistogram(const cv::Mat& src, std::vector<float>& features, int bins) {
  // Edge case: empty image
  if (src.empty()) {
    std::println(stderr, "Error: Empty image for histogram extraction");
    return -1;
  }

  // Create 3D histogram (bins x bins x bins) initialized to zero
  cv::Mat histogram = cv::Mat::zeros(bins, bins * bins, CV_32F); // using 2D Mat to store 3D histogram

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

      // compute rgb chromaticity
      float r = R / divisor; // r, g, b are in [0, 1] range
      float g = G / divisor;
      float b = 1.0f - (r + g);  // r + g + b = 1

      // compute bin indices with proper rounding (+0.5)
      int rIndex = static_cast<int>(r * (bins - 1) + 0.5f);
      int gIndex = static_cast<int>(g * (bins - 1) + 0.5f);
      int bIndex = static_cast<int>(b * (bins - 1) + 0.5f);

      // increment histogram bin
      // Mat is bins rows x (bins*bins) cols, so row = rIndex, col = gIndex*bins+bIndex
      histogram.at<float>(rIndex, gIndex * bins + bIndex) += 1.0f;
    }
  }

  // Flatten 2D histogram to 1D feature vector (raw counts)
  // normalization is done during histogram intersection
  features.clear(); // clear the features vector before writing
  features.reserve(bins * bins * bins); // reserve space to avoid multiple reallocations

  // loop through histogram and push values to features vector
  for (int i = 0; i < bins; i++) { // r bin (rows)
    const float* histRowPtr = histogram.ptr<float>(i);
    for (int j = 0; j < bins * bins; j++) { // g*bins+b (cols)
      features.push_back(histRowPtr[j]);
    }
  }

  return 0;
}


/*
  Multi-Histogram Features
  
  Splits image horizontally and makes RGB histogram for top and bottom separately
  This captures color distribution plus some spatial info
  
  512 bins per half (8*8*8), 1024 total
*/
int extractMultiHistogram(const cv::Mat& src, std::vector<float>& features) {

  // check if image is empty
  if (src.empty()) return -1;

  features.clear(); // clear out old features
  
  int bins = 8; // using 8 bins per color channel

  // split image into top and bottom halves
  int midRow = src.rows / 2;
  
  // cv::Rect is (x, y, width, height)
  cv::Mat top = src(cv::Rect(0, 0, src.cols, midRow));
  cv::Mat bottom = src(cv::Rect(0, midRow, src.cols, src.rows - midRow));
  cv::Mat halves[] = {top, bottom};
  
  for (int h = 0; h < 2; h++) {
    std::vector<float> hist(512, 0); // 8x8x8 = 512 bins
    
    // go through every pixel
    for (int y = 0; y < halves[h].rows; y++) {
      for (int x = 0; x < halves[h].cols; x++) {
        cv::Vec3b pixel = halves[h].at<cv::Vec3b>(y, x);

        int r = pixel[2] * bins / 256; 
        int g = pixel[1] * bins / 256; 
        int b = pixel[0] * bins / 256; 

        if (r >= bins) r = bins - 1;
        if (g >= bins) g = bins - 1;
        if (b >= bins) b = bins - 1;

        int idx = r * 64 + g * 8 + b;  
        hist[idx]++; 
      }
    }
    
    // add this histogram to our feature vector
    for (int i = 0; i < 512; i++) {
      features.push_back(hist[i]);
    }
  }
  
  return 0;
}

/*
  Extract Texture and Color Features 

  Combines texture (Sobel edges) and color (RGB histogram)
  
  Texture: 16 bins for gradient magnitudes (0-255 range)
  Color: 8 bins per RGB channel = 8*8*8 = 512 bins
  Total 528 features
*/
int extractTextureAndColor(const cv::Mat& src, std::vector<float>& features) {
  if (src.empty()) return -1;
  features.clear();
  
  // Get texture features from edge detection
  // convert to grayscale
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  
  // run Sobel in both x and y directions
  cv::Mat sobelX, sobelY;
  cv::Sobel(gray, sobelX, CV_16S, 1, 0);
  cv::Sobel(gray, sobelY, CV_16S, 0, 1);
  
  // combine the x and y gradients to get magnitude
  cv::Mat magnitude;
  cv::Mat absX, absY;
  cv::convertScaleAbs(sobelX, absX);
  cv::convertScaleAbs(sobelY, absY);
  cv::addWeighted(absX, 0.5, absY, 0.5, 0, magnitude);
  
  // make a histogram of the edge strengths (16 bins)
  std::vector<float> texHist(16, 0);
  for (int y = 0; y < magnitude.rows; y++) {
    for (int x = 0; x < magnitude.cols; x++) {
      int val = magnitude.at<uchar>(y, x);
      int bin = val * 16 / 256;  
      if (bin >= 16) bin = 15; 
      texHist[bin]++;
    }
  }
  
  // put texture histogram into features vector
  for (int i = 0; i < 16; i++) {
    features.push_back(texHist[i]);
  }
  
  // get color features with RGB histogram
  std::vector<float> colorHist(512, 0);
  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
      int r = pixel[2] * 8 / 256;  // convert to bin index
      int g = pixel[1] * 8 / 256;
      int b = pixel[0] * 8 / 256;
      if (r >= 8) r = 7; 
      if (g >= 8) g = 7;
      if (b >= 8) b = 7;
      colorHist[r * 64 + g * 8 + b]++;  
    }
  }
  
  // add color histogram after the texture features
  for (int i = 0; i < 512; i++) {
    features.push_back(colorHist[i]);
  }
  
  return 0;
}
  
/*
  Extract Custom Features for Portraits

  Combines DNN embedding with skin tone and brightness from center region.
  The DNN gives general features, skin/brightness help match portraits specifically.
  
  529 total: 512 DNN (from CSV) + 16 skin bins + 1 brightness
*/
int extractCustomFeaturesWithEmbedding(const cv::Mat& src, const std::vector<float>& embedding, std::vector<float>& features) {
  features.clear();
  
  // add DNN embedding first
  features.insert(features.end(), embedding.begin(), embedding.end());
  
  // work with center region
  int cx = src.cols / 2, cy = src.rows / 2;
  int size = std::min(cx, cy) / 2;
  cv::Mat centerImg = src(cv::Rect(cx - size, cy - size, size * 2, size * 2));
  
  cv::Mat hsv;
  cv::cvtColor(centerImg, hsv, cv::COLOR_BGR2HSV);
  
  // build skin tone histogram 
  std::vector<float> skinHist(16, 0);
  for (int y = 0; y < hsv.rows; y++) {
    for (int x = 0; x < hsv.cols; x++) {
      cv::Vec3b p = hsv.at<cv::Vec3b>(y, x);
      if (p[0] <= 50 && p[1] >= 20 && p[1] <= 150 && p[2] >= 50) {
        skinHist[std::min(p[0] * 16 / 256, 15)]++;
      }
    }
  }
  features.insert(features.end(), skinHist.begin(), skinHist.end());
  
  // add brightness
  cv::Mat gray;
  cv::cvtColor(centerImg, gray, cv::COLOR_BGR2GRAY);
  features.push_back(cv::mean(gray)[0]);
  
  return 0;
}