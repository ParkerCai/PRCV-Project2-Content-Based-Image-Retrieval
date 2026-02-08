/*
  Parker Cai
  Jenny Nguyen
  January 24, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  Contains prototypes for distance metric functions for feature comparison
*/

#ifndef DISTANCE_H
#define DISTANCE_H

#include <vector>


/// Prototypes
float sumOfSquaredDifference(const std::vector<float>& features1,
  const std::vector<float>& features2);

float histogramIntersectionDistance(const std::vector<float>& histA,
  const std::vector<float>& histB);

// multi-histogram distance function 
float multiHistogramDistance(const std::vector<float>& features1,
  const std::vector<float>& features2);

//texture and color distance function 
float textureAndColorDistance(const std::vector<float>& f1,
  const std::vector<float>& f2);

// cosine distance for DNN embeddings
float cosineDistance(const std::vector<float>& vA,
  const std::vector<float>& vB);

#endif // DISTANCE_H
