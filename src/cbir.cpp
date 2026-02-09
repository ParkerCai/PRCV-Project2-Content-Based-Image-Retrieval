/*
  Parker Cai
  Jenny Nguyen
  February 3, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  CBIR main program - finds similar images using various feature types
  and distance metrics.
*/


#include <iostream>
#include <vector>
#include <string>
#include <print>  // for modern C++ printing (C++23)
#include <filesystem>  // for directory traversal (cross-platform)
#include <opencv2/opencv.hpp>
#include "features.h"
#include "distance.h"
#include "csv_util/csv_util.h"  // for reading csv files
#include "unordered_map"  // for storing image features O(1) lookup

enum CBIRExitCode {
  Success = 0,
  MissingArg = 1,
  ImageLoadFailed = 2
};

enum FeatureType {
  Baseline,
  RGChromHistogram,
  RGBChromHistogram,
  MultiHistogram,
  TextureAndColor,
  DNNEmbedding,
  CustomDesign
};

// Helper function to check if a file is an image based on extension
bool isImageFile(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  return ext == ".jpg" || ext == ".png" || ext == ".ppm" || ext == ".tif";
}

// Helper function to the embedding for a filename from the csv file
std::vector<float> getEmbedding(
  const std::string& filename, 
  const std::unordered_map<std::string, int>& lookupIndex,
  const std::vector<std::vector<float>>& embeddings) {
  // Fast method: O(1) lookup with hash map, using auto keyword to infer type
  auto it = lookupIndex.find(filename);  // search for key
  if (it == lookupIndex.end()) {  // not found
    return {};  // return empty vector
  }
  return embeddings[it->second];  // use the [2nd] index to get the embedding
}


/*
  Standard main function with command line arguments for
  Content-based Image Retrieval.

  Usage:
  ./cbir.exe <query_image> <image_database_directory> [feature_type]
  ./cbir.exe data/olympus/pic.0164.jpg data/olympus rghistogram
  feature_type options:
    baseline  - 7x7 center pixel block (default)
    rghistogram - 2D rg chromaticity histogram with intersection
    rgbhistogram - 3D rgb chromaticity histogram with intersection
    multihistogram - multi-histogram with custom distance
    textureandcolor - combined texture and color features with custom distance
    dnnembedding - DNN embedding with cosine distance
    customdesign - custom features and distance function 
*/
int main(int argc, char* argv[]) {
  // 1. parse command line arguments
  // Error handling for missing arguments
  if (argc < 3) {
    std::println("Usage: {} <query_image> <image_database_directory> [feature_type]", argv[0]);
    std::println("  feature_type: baseline (default), rghistogram, rgbhistogram, multihistogram, textureandcolor, customdesign, dnnembedding");
    exit(MissingArg);  // exit with error code
  }

  cv::Mat src;  // read image from file specified in command line argument
  std::string imageDir = argv[2];  // image directory path

  // Parse feature type
  FeatureType featureType = Baseline;
  if (argc >= 4) {
    std::string featureArg = argv[3];
    if (featureArg == "rghistogram") {
      featureType = RGChromHistogram;
    }
    else if (featureArg == "rgbhistogram") {
      featureType = RGBChromHistogram;
    }
    else if (featureArg == "multihistogram") {
      featureType = MultiHistogram;
    }
    else if (featureArg == "textureandcolor") {
      featureType = TextureAndColor;
    }
    else if (featureArg == "dnnembedding") {
      featureType = DNNEmbedding;
      // requires a csv file
      if (argc < 5) {
        std::println(stderr, "Error: Missing csv file for DNN embedding");
        exit(MissingArg);
      } 
    }
    else if (featureArg == "custom") {
      featureType = CustomDesign;
    }
  }
  

  // Read and load the query image
  src = cv::imread(argv[1]);
  // Error handling: empty image
  if (src.empty()) {
    std::println(stderr, "Error: Failed to load query image {}", argv[1]);
    exit(ImageLoadFailed);
  }


  // 2. Read directory
  std::vector<std::string> imageFiles;
  for (const auto& entry : std::filesystem::directory_iterator(imageDir)) {
    if (entry.is_regular_file() && isImageFile(entry.path())) {
      imageFiles.push_back(entry.path().string());
    }
  }


  // For DNN: load embeddings from CSV file
  // use the csv util to read the csv file
  std::vector<char*> csvFilenames;
  std::vector<std::vector<float>> csvEmbeddings;
  std::unordered_map<std::string, int> csvLookupIndex;  // just store the index of each filename

  // 3. Extract features from query image
  std::vector<float> queryFeatures;
  int status;

  if (featureType == RGChromHistogram) {
    std::println("2D RG Chromaticity Histogram (16x16 bins) with Histogram Intersection");
    status = extractRGChromHistogram(src, queryFeatures, 16);
  }
  else if (featureType == RGBChromHistogram) {
    std::println("3D RGB Chromaticity Histogram (8x8x8 bins) with Histogram Intersection");
    status = extractRGBChromHistogram(src, queryFeatures, 8);
  }
  else if (featureType == MultiHistogram) {
    std::println("Multi-histogram");
    status = extractMultiHistogram(src, queryFeatures);
  }
  else if (featureType == TextureAndColor) {
    std::println("Texture + Color");
    status = extractTextureAndColor(src, queryFeatures);
  }
  else if (featureType == DNNEmbedding) {
    // read the csv file using the csv util
    int csvStatus = read_image_data_csv(argv[4], csvFilenames, csvEmbeddings, 0);
    if (csvStatus != 0) {
      std::println(stderr, "Error: Failed to read CSV file {}", argv[4]);
      exit(ImageLoadFailed);
    }
    std::println("Loaded {} embeddings from CSV", (int)csvFilenames.size());

    // build a lookup map for the csv file
    for (int i = 0; i < csvFilenames.size(); i++) {
      csvLookupIndex[csvFilenames[i]] = i; // just the index to save memory
    }

    // find the index of the query image in the csv file
    // get just the filename from the query path
    std::filesystem::path queryPath(argv[1]); // full path: data/olympus/pic.0164.jpg
    std::string queryFilename = queryPath.filename().string(); // get just the filename: pic.0164.jpg

    // Slow method: O(n) lookup
    // bool found = false;
    // for (int i = 0; i < csvFilenames.size(); i++) {
    //   if (strcmp(csvFilenames[i], queryFilename.c_str()) == 0) {
    //     queryFeatures = csvEmbeddings[i];  // assign directly
    //     found = true;
    //     break;
    //   }
    // }
    // if (!found) {
    //   std::println(stderr, "Error: Query image {} not found in CSV file", queryFilename);
    //   exit(ImageLoadFailed);
    // }

    // Fast method: O(1) lookup with hash map, using auto keyword to infer type
    queryFeatures = getEmbedding(queryFilename, csvLookupIndex, csvEmbeddings);
    if (queryFeatures.empty()) {  // not found
      std::println(stderr, "Error: Query image {} not found in CSV file", queryFilename);
      exit(ImageLoadFailed);
    }
    status = 0;
  }
  else if (featureType == CustomDesign) {
    // read the csv file using the csv util (ONCE)
    int csvStatus = read_image_data_csv(const_cast<char*>("data/ResNet18_olym.csv"), csvFilenames, csvEmbeddings, 0);
    if (csvStatus != 0) {
      std::println(stderr, "Error: Failed to read CSV file");
      exit(ImageLoadFailed);
    }
    std::println("Loaded {} embeddings from CSV", (int)csvFilenames.size());

    // build lookup map
    for (int i = 0; i < csvFilenames.size(); i++) {
      csvLookupIndex[csvFilenames[i]] = i;
    }

    std::println("Custom (DNN + skin + brightness)");
    
    // Get query embedding
    std::filesystem::path queryPath(argv[1]);
    std::string queryFilename = queryPath.filename().string();
    std::vector<float> queryEmbedding = getEmbedding(queryFilename, csvLookupIndex, csvEmbeddings);
    
    if (queryEmbedding.empty()) {
      std::println(stderr, "Error: Query image {} not found in CSV", queryFilename);
      exit(ImageLoadFailed);
    }
  
  // Extract custom features
   status = extractCustomFeaturesWithEmbedding(src, queryEmbedding, queryFeatures);
  }
  else {
    std::println("Baseline features (7x7 center block) with SSD");
    status = extractBaselineFeatures(src, queryFeatures);
  }

  // Error handling for feature extraction failure
  if (status != 0) {
    std::println(stderr, "Error: Failed to extract features from query image");
    exit(ImageLoadFailed);
  }


  // 4. Sort images by distance
  std::vector<std::pair<float, std::string>> distances;

  // iterate through all images in the directory
  for (const auto& imageFile : imageFiles) {
    cv::Mat image = cv::imread(imageFile);

    // Error handling for image loading failure
    if (image.empty()) {
      std::println(stderr, "Error: Failed to load image {}", imageFile);
      continue;
    }

    std::vector<float> features;
    int extractStatus;

    if (featureType == RGChromHistogram) {
      extractStatus = extractRGChromHistogram(image, features, 16);
    }
    else if (featureType == RGBChromHistogram) {
      extractStatus = extractRGBChromHistogram(image, features, 8);
    }
    else if (featureType == MultiHistogram) {
      extractStatus = extractMultiHistogram(image, features);
    }
    else if (featureType == TextureAndColor) {
      extractStatus = extractTextureAndColor(image, features);
    }
    else if (featureType == DNNEmbedding) {
      // get the filename from the image path
      std::filesystem::path imagePath(imageFile);
      std::string imageFilename = imagePath.filename().string();

      // O(1) lookup in hash map
      features = getEmbedding(imageFilename, csvLookupIndex, csvEmbeddings);
      extractStatus = features.empty() ? -1 : 0;
    } 
    else if (featureType == CustomDesign) {
      // Get embedding for this image
      std::filesystem::path imagePath(imageFile);
      std::string imageFilename = imagePath.filename().string();
      std::vector<float> imgEmbedding = getEmbedding(imageFilename, csvLookupIndex, csvEmbeddings);
      
      if (imgEmbedding.empty()) {
        extractStatus = -1;
      } else {
        extractStatus = extractCustomFeaturesWithEmbedding(image, imgEmbedding, features);
      }
  }
    else {
      extractStatus = extractBaselineFeatures(image, features);
    }

    // Error handling for feature extraction failure
    if (extractStatus != 0) {
      std::println(stderr, "Error: Failed to extract features from image {}", imageFile);
      continue;
    }

    // compute distance and store it with the filename
    float distance;
    if (featureType == RGChromHistogram) {
      distance = histogramIntersectionDistance(queryFeatures, features);
    }
    else if (featureType == RGBChromHistogram) {
      distance = histogramIntersectionDistance(queryFeatures, features);
    }
    else if (featureType == MultiHistogram) {
      distance = multiHistogramDistance(queryFeatures, features);
    }
    else if (featureType == TextureAndColor) {
      distance = textureAndColorDistance(queryFeatures, features);
    }
    else if (featureType == DNNEmbedding) {
      distance = cosineDistance(queryFeatures, features);
    }
    else if (featureType == CustomDesign) {
      distance = customDistance(queryFeatures, features);
    }
    else {
      distance = sumOfSquaredDifference(queryFeatures, features);
    }

    distances.push_back(std::make_pair(distance, imageFile)); // {distance, filename} pair
  }
  std::sort(distances.begin(), distances.end());

  // 4.5 Display top 4 results (query image + top 3 matches)
  std::println("\nTop 4 similar images:");
  for (int i = 0; i < 4 && i < (int)distances.size(); i++) {
    std::filesystem::path p(distances[i].second); // get filename from path
    std::println("{}: {} (distance: {:.6f})", i + 1, p.filename().string(), distances[i].first); // round to 6 decimal places
  }

  // Create combined display
  cv::Mat display;
  std::vector<cv::Mat> images;
  images.push_back(src);

  for (int i = 1; i < 4 && i < distances.size(); i++) {
    cv::Mat match = cv::imread(distances[i].second);
    images.push_back(match);
  }

  cv::hconcat(images, display);  // combine horizontally
  cv::imshow("Query and Top 3 Matches", display);
  cv::waitKey(0);

  return Success;
}
