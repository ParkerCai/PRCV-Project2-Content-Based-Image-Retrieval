/*
  Parker Cai
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
