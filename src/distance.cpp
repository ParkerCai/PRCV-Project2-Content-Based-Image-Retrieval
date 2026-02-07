/*
  Parker Cai
  Jenny Nguyen
  February 2, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  Implementation of distance metric functions for comparing image feature vectors.
*/

#include "distance.h"
#include <iostream>


/*
  Sum of Squared Difference (SSD)
  - Compute with L-2 norm: Euclidean distance
    (but we don't care about the sqrt due to monotonicity)
  - SSD or SSE formula: d(a, b) = Σ (aᵢ - bᵢ)²

  Input:
    featuresA - first feature vector (std::vector<float>)
    featuresB - second feature vector (std::vector<float>)

  Output:
    float - SSD value
*/
float sumOfSquaredDifference(const std::vector<float>& featuresA,
  const std::vector<float>& featuresB) {
  float sum = 0.0f;

  for (int i = 0; i < featuresA.size(); i++) {
    // x*x is faster than pow(x, 2)
    sum += (featuresA[i] - featuresB[i]) * (featuresA[i] - featuresB[i]);
  }

  return sum;
}


/*
  Histogram Intersection Distance
  - Measures similarity between two histograms
  - Formula: intersection = Σ min(Aᵢ, Bᵢ) for normalized histograms
  - Returns (1 - intersection) so smaller values = more similar
  - Normalizes histograms before comparison

  Input:
    histA - first histogram (std::vector<float>)
    histB - second histogram (std::vector<float>)

  Output:
    float - distance value in range [0, 1] (0 = identical, 1 = no overlap)
*/
float histogramIntersection(const std::vector<float>& histA,
  const std::vector<float>& histB) {
  // Check for size mismatch
  if (histA.size() != histB.size() || histA.empty()) {
    return 1.0f;  // Maximum distance if invalid input
  }

  // Compute sums for normalization
  float sumA = 0.0f, sumB = 0.0f;
  for (size_t i = 0; i < histA.size(); i++) {
    sumA += histA[i];
    sumB += histB[i];
  }

  // Avoid division by zero
  if (sumA < 1.0f || sumB < 1.0f) {
    return 1.0f;
  }

  // Compute normalized histogram intersection
  float intersection = 0.0f;
  for (size_t i = 0; i < histA.size(); i++) {
    float normA = histA[i] / sumA;
    float normB = histB[i] / sumB;
    intersection += std::min(normA, normB);
  }

  // Return distance (1 - similarity)
  // intersection is in range [0, 1], so distance is also in [0, 1]
  return 1.0f - intersection;
}

/*
  Multi-Histogram Distance (Task 3)
  - Compares two multi-histogram features using histogram intersection
  - Computes intersection separately for top and bottom halves
  - Uses equal weighting: 50% top half, 50% bottom half
  - Formula: distance = 1 - ((intersect_top + intersect_bot) / 2)
  - Normalizes each histogram before comparison

  Input:
    f1 - first feature vector (1024 values: 512 top + 512 bottom)
    f2 - second feature vector (1024 values: 512 top + 512 bottom)

  Output:
    float - distance value in range [0, 1] (0 = identical, 1 = no overlap)
*/
float multiHistogramDistance(const std::vector<float>& f1, const std::vector<float>& f2) {
  if (f1.size() != 1024 || f2.size() != 1024) return 1.0f;
  
  // normalize and compare top half
  float sum1_top = 0, sum2_top = 0;
  for (int i = 0; i < 512; i++) {
    sum1_top += f1[i];
    sum2_top += f2[i];
  }
  
  float intersect_top = 0;
  for (int i = 0; i < 512; i++) {
    float n1 = f1[i] / sum1_top;
    float n2 = f2[i] / sum2_top;
    intersect_top += std::min(n1, n2);
  }
  
  // normalize and compare bottom half
  float sum1_bot = 0, sum2_bot = 0;
  for (int i = 512; i < 1024; i++) {
    sum1_bot += f1[i];
    sum2_bot += f2[i];
  }
  
  float intersect_bot = 0;
  for (int i = 512; i < 1024; i++) {
    float n1 = f1[i] / sum1_bot;
    float n2 = f2[i] / sum2_bot;
    intersect_bot += std::min(n1, n2);
  }
  
  // average them
  float avg = (intersect_top + intersect_bot) / 2.0f;
  return 1.0f - avg;
}