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
  Multi-Histogram Features (Task 3)
  
  Splits the image in half (top/bottom) and makes a separate RGB histogram 
  for each half. This way we get color info AND some spatial info about 
  where colors appear in the image.
  
  Each histogram uses 8 bins per channel (R,G,B) = 8*8*8 = 512 bins
  So total we have 512 values for top half + 512 for bottom = 1024 features
  
  Input:
    src - input image (cv::Mat)
    features - output feature vector (std::vector<float>)

  Returns: 0 if success, -1 if error
*/
int extractMultiHistogram(const cv::Mat& src, std::vector<float>& features) {

  // check if image is empty
  if (src.empty()) return -1;

  features.clear(); // clear out old features
  
  int bins = 8; // using 8 bins per color channel

  // split image into top and bottom halves
  int midRow = src.rows / 2;
  
  // cv::Rect is (x, y, width, height)
  cv::Mat topHalf = src(cv::Rect(0, 0, src.cols, midRow));
  cv::Mat bottomHalf = src(cv::Rect(0, midRow, src.cols, src.rows - midRow));

  cv::Mat halves[] = {topHalf, bottomHalf};
  
  // do this for both top and bottom
  for (int h = 0; h < 2; h++) {
    std::vector<float> hist(512, 0); // 8x8x8 = 512 bins
    
    // go through every pixel
    for (int y = 0; y < halves[h].rows; y++) {
      for (int x = 0; x < halves[h].cols; x++) {
        cv::Vec3b pixel = halves[h].at<cv::Vec3b>(y, x);

        // figure out which bin each channel goes in
        // divide by 256 not 255 so that 255 stays in bin 7
        int r = pixel[2] * bins / 256; 
        int g = pixel[1] * bins / 256; 
        int b = pixel[0] * bins / 256; 

        // make sure we don't go out of bounds
        // (can happen with value 255: 255*8/256 = 7.96 which rounds to 8)
        if (r >= bins) r = bins - 1;
        if (g >= bins) g = bins - 1;
        if (b >= bins) b = bins - 1;

        // convert (r,g,b) to a single index in the histogram
        // basically flattening a 3D array into 1D
        int index = r * 64 + g * 8 + b; // ranges from 0 to 511
        hist[index]++;
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
  Extract Texture and Color Features (Task 4)

  This function combines two types of features: texture (using edges) and color.
  
  For texture: we use Sobel edge detection to find gradient magnitudes, then 
  make a histogram with 16 bins (covering values 0-255)
  
  For color: we make an RGB histogram with 8 bins per channel (8*8*8 = 512 bins)
  
  Final feature vector has 16 + 512 = 528 total values

  Input:
    src - input image (cv::Mat), using const to prevent modifying src img (safety)
    features - output feature vector (std::vector<float>)
*/
int extractTextureAndColor(const cv::Mat& src, std::vector<float>& features) {
  if (src.empty()) return -1;
  features.clear();
  
  // First get texture features from edge detection
  // need to convert to grayscale first
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
      int bin = val * 16 / 256;  // converts 0-255 range into 0-15 bins
      if (bin >= 16) bin = 15;  // safety check
      texHist[bin]++;
    }
  }
  
  // put texture histogram into features vector
  for (int i = 0; i < 16; i++) {
    features.push_back(texHist[i]);
  }
  
  // now get color features with RGB histogram
  // same approach as Task 3 but for the whole image
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
      colorHist[r * 64 + g * 8 + b]++;  // flatten into 1D
    }
  }
  
  // add color histogram after the texture features
  for (int i = 0; i < 512; i++) {
    features.push_back(colorHist[i]);
  }
  
  return 0;
}
  