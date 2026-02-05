/*
  Parker Cai
  February 3, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  Contains prototypes for feature extraction functions for CBIR.
*/

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// Prototypes
int extractBaselineFeatures(const cv::Mat& image, std::vector<float>& features);

#endif // FEATURES_H
