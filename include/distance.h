/*
  Parker Cai
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

#endif // DISTANCE_H
