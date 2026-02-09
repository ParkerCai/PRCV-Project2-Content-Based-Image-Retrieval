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
  Multi-Histogram Distance 
  Compares top and bottom halves of the image separately
  
  f1, f2 - 1024 bin histograms (512 for top half, 512 for bottom)
  Uses histogram intersection on each half then averages them
  Returns distance where 0 = same image, 1 = totally different
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
  Texture and Color Distance
  Combines texture histogram (16 bins) and color histogram (512 bins)
  Uses histogram intersection on each part, then averages the results
  
  f1, f2 - feature vectors (528 floats each)
  returns distance [0,1] where 0 = identical
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


/*
  Cosine Distance 
  - Compares two feature vectors using cosine similarity
  - Formula: d(vA, vB) = 1 - cos(theta) 
             where theta is the angle between the two vectors.
             distance = 1 - (dot_product(a, b) / (norm(a) * norm(b)))
             where norm(a) is the Euclidean norm of vector a

  - Returns cosine-distance (smaller values = more similar)

  Input:
    vA - first feature vector (std::vector<float>)
    vB - second feature vector (std::vector<float>)

  Output:
    float - distance value in range [0, 1] (0 = identical, 1 = no overlap)
*/
float cosineDistance(const std::vector<float>& vA,
                     const std::vector<float>& vB) {
    // Check for size mismatch
    if (vA.size() != vB.size()) return 1.0f;

    // Compute dot product and Euclidean norms
    float dotProduct = 0.0f;
    float normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < vA.size(); i++) {
        dotProduct += vA[i] * vB[i];
        normA += vA[i] * vA[i];
        normB += vB[i] * vB[i];
    }

    // Handle division by zero
    if (normA < 1.0f || normB < 1.0f) return 1.0f;

    // normA = std::sqrt(normA);
    // normB = std::sqrt(normB); // faster to do this in one step
    float normAB = std::sqrt(normA * normB);
    
    // Compute cosine similarity
    float similarity = dotProduct / normAB;
    return 1.0f - similarity;
}

/*
  Custom Distance for Face Portrait Matching 
  
  Combines DNN features (512 values), skin histogram (16), and brightness (1)
  Weights: 70% for DNN cosine distance, 20% skin tone, 10% brightness
  Returns combined distance - lower means better match
*/

float customDistance(const std::vector<float>& f1, const std::vector<float>& f2) {
  
  // cosine distance on DNN features
  float dot = 0, mag1 = 0, mag2 = 0;
  for (int i = 0; i < 512; i++) {
    dot += f1[i] * f2[i];
    mag1 += f1[i] * f1[i];
    mag2 += f2[i] * f2[i];
  }
  
  float similarity = dot / (sqrt(mag1) * sqrt(mag2));
  similarity = std::min(similarity, 1.0f);  // clamp to avoid -0
  float dnn_dist = 1.0 - similarity;
  
  // skin tone histogram comparison
  float s1 = 0, s2 = 0;
  for (int i = 512; i < 528; i++) {
    s1 += f1[i];
    s2 += f2[i];
  }
  float inter = 0;
  for (int i = 512; i < 528; i++) {
    inter += std::min(f1[i]/s1, f2[i]/s2);
  }
  float skin_dist = 1.0 - inter;
  
  // brightness
  float bright_dist = std::abs(f1[528] - f2[528]) / 255.0;
  
  // weighted combo - tweaked these to get better results
  return 0.7 * dnn_dist + 0.2 * skin_dist + 0.1 * bright_dist;
}