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
             distance = 1 - intersection (identical histograms have distance 0, no overlap has distance 1)
  - Returns (1 - intersection) so smaller values = more similar
  - Normalizes histograms before comparison

  Input:
    histA - first histogram (std::vector<float>)
    histB - second histogram (std::vector<float>)

  Output:
    float - distance value in range [0, 1] (0 = identical, 1 = no overlap)
*/
float histogramIntersectionDistance(const std::vector<float>& histA,
  const std::vector<float>& histB) {
  // Check for size mismatch
  if (histA.size() != histB.size() || histA.empty()) {
    return 1.0f;  // Maximum distance if invalid input
  }

  // Compute sums for normalization (sum of buckets)
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
  // Check for size mismatch - both must be exactly 1024 bins
  if (f1.size() != 1024 || f2.size() != 1024) return 1.0f;

  // Compute sums for normalization of top half histograms
  float sum1_top = 0, sum2_top = 0;
  for (int i = 0; i < 512; i++) {
    sum1_top += f1[i];
    sum2_top += f2[i];
  }

  // Compute normalized histogram intersection for top half
  float intersect_top = 0;
  for (int i = 0; i < 512; i++) {
    // Normalize bin counts to create probability distributions
    float n1 = f1[i] / sum1_top;
    float n2 = f2[i] / sum2_top;
    // Accumulate minimum of normalized values (intersection measure)
    intersect_top += std::min(n1, n2);
  }

  // Compute sums for normalization of bottom half histograms
  float sum1_bot = 0, sum2_bot = 0;
  for (int i = 512; i < 1024; i++) {
    sum1_bot += f1[i];
    sum2_bot += f2[i];
  }

  // Compute normalized histogram intersection for bottom half
  float intersect_bot = 0;
  for (int i = 512; i < 1024; i++) {
    // Normalize bin counts to create probability distributions
    float n1 = f1[i] / sum1_bot;
    float n2 = f2[i] / sum2_bot;
    // Accumulate minimum of normalized values (intersection measure)
    intersect_bot += std::min(n1, n2);
  }

  // Average the intersection values from both regions
  // This gives equal weight to top and bottom halves
  float avg = (intersect_top + intersect_bot) / 2.0f;

  // Return distance (1 - average intersection)
  return 1.0f - avg;
}


/*
  Texture and Color Distance (Task 4)
  - Compares both texture and color features using histogram intersection
  - Uses equal weighting: 50% texture similarity + 50% color similarity
  - Formula: distance = 1 - ((texture_intersection + color_intersection) / 2)
  - Normalizes each histogram separately before comparison

  Input:
    f1 - first feature vector (528 values: 16 texture + 512 color)
    f2 - second feature vector (528 values: 16 texture + 512 color)

  Output:
    float - distance value in range [0, 1] (0 = identical, 1 = no overlap)
*/
float textureAndColorDistance(const std::vector<float>& f1, const std::vector<float>& f2) {
  // Check for size mismatch
  if (f1.size() != 528 || f2.size() != 528) return 1.0f;

  // Part 1: Compare texture histograms (first 16 values)
  // Compute sums for normalization
  float sum1_t = 0, sum2_t = 0;
  for (int i = 0; i < 16; i++) {
    sum1_t += f1[i];
    sum2_t += f2[i];
  }

  // Compute normalized histogram intersection for texture
  float intersect_t = 0;
  for (int i = 0; i < 16; i++) {
    intersect_t += std::min(f1[i] / sum1_t, f2[i] / sum2_t);
  }

  // Part 2: Compare color histograms (next 512 values)
  // Compute sums for normalization
  float sum1_c = 0, sum2_c = 0;
  for (int i = 16; i < 528; i++) {
    sum1_c += f1[i];
    sum2_c += f2[i];
  }

  // Compute normalized histogram intersection for color
  float intersect_c = 0;
  for (int i = 16; i < 528; i++) {
    intersect_c += std::min(f1[i] / sum1_c, f2[i] / sum2_c);
  }

  // Combine with equal weighting (average of the two intersections)
  float avg = (intersect_t + intersect_c) / 2.0f;
  return 1.0f - avg;
}