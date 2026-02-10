/*
  Parker Cai
  Jenny Nguyen
  February 8, 2026
  CS5330 - Project 2: Content-based Image Retrieval

  CBIR GUI Application using Dear ImGui
*/

#include <iostream>
#include <vector>
#include <string>
#include <print>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#include <commdlg.h>
#include <shlobj.h>
#include <dwmapi.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "dwmapi.lib")
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#endif

#include "features.h"
#include "distance.h"
#include "csv_util.h"

// ============================================================================
// Types and State
// ============================================================================

enum FeatureType {
  Baseline, RGChromHistogram, RGBChromHistogram, MultiHistogram,
  TextureAndColor, DNNEmbedding, CustomDesign, FeatureTypeCount
};

const char* featureTypeNames[] = {
  "Baseline (7x7 center block)", "RG Chromaticity Histogram",
  "RGB Chromaticity Histogram", "Multi-Histogram",
  "Texture + Color", "DNN Embedding", "Custom Design"
};

struct SearchResult {
  std::string filepath, filename;
  float distance;
  GLuint textureId = 0;
  int width = 0, height = 0;
};

struct AppState {
  char queryImagePath[512] = "";
  char imageDatabaseDir[512] = "data/olympus";
  char csvFilePath[512] = "data/ResNet18_olym.csv";
  int selectedFeatureType = 0;

  cv::Mat queryImage;
  GLuint queryTextureId = 0;
  int queryWidth = 0, queryHeight = 0;

  bool isSearching = false, hasResults = false;
  std::vector<SearchResult> results;
  int numResultsToShow = 4;

  std::vector<char*> csvFilenames;
  std::vector<std::vector<float>> csvEmbeddings;
  std::unordered_map<std::string, int> csvLookupIndex;
  bool embeddingsLoaded = false;

  std::string statusMessage = "Ready. Drag & drop an image or click Browse.";
  float dpiScale = 1.0f;
  float splitRatio = 0.4f;
};

static AppState g_app;

// ============================================================================
// Helpers
// ============================================================================

// Truncate a path in the middle: "C:\Users\use...\pic001.jpg"
std::string truncatePathMiddle(const std::string& fullPath, float maxWidth) {
  if (ImGui::CalcTextSize(fullPath.c_str()).x <= maxWidth) return fullPath;

  std::string filename = std::filesystem::path(fullPath).filename().string();
  std::string ellipsis = "...\\";
  float availForPrefix = maxWidth - ImGui::CalcTextSize(filename.c_str()).x - ImGui::CalcTextSize(ellipsis.c_str()).x;

  if (availForPrefix < ImGui::CalcTextSize("C:\\").x)
    return ellipsis + filename;

  std::string prefix;
  for (char c : fullPath) {
    std::string candidate = prefix + c;
    if (ImGui::CalcTextSize(candidate.c_str()).x > availForPrefix) break;
    prefix = candidate;
  }
  return prefix + ellipsis + filename;
}

bool isImageFile(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".ppm" || ext == ".tif" || ext == ".bmp";
}

GLuint matToTexture(const cv::Mat& mat, int& outWidth, int& outHeight) {
  if (mat.empty()) return 0;

  cv::Mat rgb;
  cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);

  GLuint textureId;
  glGenTextures(1, &textureId);
  glBindTexture(GL_TEXTURE_2D, textureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);

  outWidth = rgb.cols;
  outHeight = rgb.rows;
  return textureId;
}

void freeTexture(GLuint& textureId) {
  if (textureId != 0) { glDeleteTextures(1, &textureId); textureId = 0; }
}

std::vector<float> getEmbedding(const std::string& filename) {
  auto it = g_app.csvLookupIndex.find(filename);
  return (it != g_app.csvLookupIndex.end()) ? g_app.csvEmbeddings[it->second] : std::vector<float>{};
}

// Load a query image from path, updating texture and state
void loadQueryImage(const std::string& path) {
  strncpy(g_app.queryImagePath, path.c_str(), sizeof(g_app.queryImagePath) - 1);
  freeTexture(g_app.queryTextureId);
  g_app.queryImage = cv::imread(path);
  if (!g_app.queryImage.empty())
    g_app.queryTextureId = matToTexture(g_app.queryImage, g_app.queryWidth, g_app.queryHeight);
}

// Extract features for any feature type (returns 0 on success)
int extractFeatures(FeatureType type, const cv::Mat& image, std::vector<float>& features,
                    const std::string& filename = "") {
  switch (type) {
    case RGChromHistogram:  return extractRGChromHistogram(image, features, 16);
    case RGBChromHistogram: return extractRGBChromHistogram(image, features, 8);
    case MultiHistogram:    return extractMultiHistogram(image, features);
    case TextureAndColor:   return extractTextureAndColor(image, features);
    case DNNEmbedding:
      features = getEmbedding(filename);
      return features.empty() ? -1 : 0;
    case CustomDesign: {
      auto emb = getEmbedding(filename);
      return emb.empty() ? -1 : extractCustomFeaturesWithEmbedding(image, emb, features);
    }
    default: return extractBaselineFeatures(image, features);
  }
}

// Compute distance between two feature vectors
float computeDistance(FeatureType type, const std::vector<float>& a, const std::vector<float>& b) {
  switch (type) {
    case RGChromHistogram:
    case RGBChromHistogram: return histogramIntersectionDistance(a, b);
    case MultiHistogram:    return multiHistogramDistance(a, b);
    case TextureAndColor:   return textureAndColorDistance(a, b);
    case DNNEmbedding:      return cosineDistance(a, b);
    case CustomDesign:      return customDistance(a, b);
    default:                return sumOfSquaredDifference(a, b);
  }
}

// Render two lines of centered gray text in the available region
void renderCenteredText(const char* line1, const char* line2) {
  ImVec4 gray(0.5f, 0.5f, 0.5f, 1.0f);
  float w = ImGui::GetContentRegionAvail().x;
  float h = ImGui::GetContentRegionAvail().y;
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (h - ImGui::GetTextLineHeightWithSpacing() * 2) * 0.5f);
  ImGui::SetCursorPosX((w - ImGui::CalcTextSize(line1).x) * 0.5f);
  ImGui::TextColored(gray, "%s", line1);
  ImGui::SetCursorPosX((w - ImGui::CalcTextSize(line2).x) * 0.5f);
  ImGui::TextColored(gray, "%s", line2);
}

// ============================================================================
// File Dialogs (Windows Native)
// ============================================================================

#ifdef _WIN32
std::string openFileDialog() {
  char filename[MAX_PATH] = "";
  OPENFILENAMEA ofn;
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = "Image Files\0*.jpg;*.jpeg;*.png;*.ppm;*.tif;*.bmp\0All Files\0*.*\0";
  ofn.lpstrFile = filename;
  ofn.nMaxFile = MAX_PATH;
  ofn.lpstrTitle = "Select Query Image";
  ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;
  return GetOpenFileNameA(&ofn) ? std::string(filename) : "";
}

std::string openFolderDialog() {
  char path[MAX_PATH] = "";
  BROWSEINFOA bi = { 0 };
  bi.lpszTitle = "Select Image Database Folder";
  bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
  LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
  if (pidl) { SHGetPathFromIDListA(pidl, path); CoTaskMemFree(pidl); return path; }
  return "";
}
#else
std::string openFileDialog() { return ""; }
std::string openFolderDialog() { return ""; }
#endif

// ============================================================================
// GLFW Callbacks
// ============================================================================

void dropCallback(GLFWwindow* window, int count, const char** paths) {
  if (count <= 0) return;
  std::filesystem::path p(paths[0]);
  if (isImageFile(p)) {
    loadQueryImage(paths[0]);
    g_app.statusMessage = "Dropped: " + p.filename().string();
  } else {
    g_app.statusMessage = "Error: Not a valid image file";
  }
}

void errorCallback(int error, const char* description) {
  std::println(stderr, "GLFW Error {}: {}", error, description);
}

// ============================================================================
// CBIR Search
// ============================================================================

void performSearch() {
  if (g_app.queryImage.empty()) { g_app.statusMessage = "Error: No query image loaded"; return; }
  if (!std::filesystem::exists(g_app.imageDatabaseDir)) {
    std::println(stderr, "CWD: {}", std::filesystem::current_path().string());
    std::println(stderr, "DB path: '{}'", g_app.imageDatabaseDir);
    std::println(stderr, "Absolute: {}", std::filesystem::absolute(g_app.imageDatabaseDir).string());
    g_app.statusMessage = "Error: Image database directory not found";
    return;
  }

  g_app.isSearching = true;
  g_app.statusMessage = "Searching...";
  for (auto& r : g_app.results) freeTexture(r.textureId);
  g_app.results.clear();

  auto type = static_cast<FeatureType>(g_app.selectedFeatureType);

  // Load DNN embeddings if needed
  if ((type == DNNEmbedding || type == CustomDesign) && !g_app.embeddingsLoaded) {
    const char* csv = (type == DNNEmbedding) ? g_app.csvFilePath : "data/ResNet18_olym.csv";
    if (read_image_data_csv(const_cast<char*>(csv), g_app.csvFilenames, g_app.csvEmbeddings, 0) != 0) {
      g_app.statusMessage = "Error: Failed to load CSV embeddings";
      g_app.isSearching = false;
      return;
    }
    for (size_t i = 0; i < g_app.csvFilenames.size(); i++)
      g_app.csvLookupIndex[g_app.csvFilenames[i]] = static_cast<int>(i);
    g_app.embeddingsLoaded = true;
  }

  // Extract query features
  std::vector<float> queryFeatures;
  std::string queryFilename = std::filesystem::path(g_app.queryImagePath).filename().string();
  if (extractFeatures(type, g_app.queryImage, queryFeatures, queryFilename) != 0) {
    g_app.statusMessage = "Error: Failed to extract query features";
    g_app.isSearching = false;
    return;
  }

  // Scan database and compute distances
  std::vector<std::pair<float, std::string>> distances;
  for (const auto& entry : std::filesystem::directory_iterator(g_app.imageDatabaseDir)) {
    if (!entry.is_regular_file() || !isImageFile(entry.path())) continue;
    cv::Mat image = cv::imread(entry.path().string());
    if (image.empty()) continue;

    std::vector<float> features;
    if (extractFeatures(type, image, features, entry.path().filename().string()) != 0) continue;
    distances.push_back({computeDistance(type, queryFeatures, features), entry.path().string()});
  }

  // Sort and build results, skipping the self-match (query image with distance ~0)
  std::sort(distances.begin(), distances.end());
  int startIdx = (!distances.empty() && distances[0].first < 1e-4f) ? 1 : 0;
  int n = std::min(g_app.numResultsToShow, static_cast<int>(distances.size()) - startIdx);
  for (int i = 0; i < n; i++) {
    SearchResult r;
    r.filepath = distances[startIdx + i].second;
    r.filename = std::filesystem::path(r.filepath).filename().string();
    r.distance = distances[startIdx + i].first;
    cv::Mat img = cv::imread(r.filepath);
    if (!img.empty()) r.textureId = matToTexture(img, r.width, r.height);
    g_app.results.push_back(r);
  }

  g_app.hasResults = true;
  g_app.isSearching = false;
  g_app.statusMessage = "Found " + std::to_string(distances.size() - startIdx) + " images. Showing top " + std::to_string(n) + ".";
}

// ============================================================================
// UI Panels
// ============================================================================

void renderLeftPanel(float totalHeight) {
  ImGui::Text("Query Image:");
  ImGui::Spacing();

  float controlsHeight = 260.0f * g_app.dpiScale;
  if (g_app.selectedFeatureType == DNNEmbedding)
    controlsHeight += 40.0f * g_app.dpiScale;

  // Query image or placeholder
  if (g_app.queryTextureId != 0) {
    float availWidth = ImGui::GetContentRegionAvail().x;
    float aspectRatio = static_cast<float>(g_app.queryWidth) / g_app.queryHeight;
    float displayWidth = availWidth;
    float displayHeight = displayWidth / aspectRatio;
    float maxHeight = totalHeight - controlsHeight - ImGui::GetCursorPosY();
    if (displayHeight > maxHeight) {
      displayHeight = maxHeight;
      displayWidth = displayHeight * aspectRatio;
    }
    ImGui::Image((ImTextureID)(intptr_t)g_app.queryTextureId, ImVec2(displayWidth, displayHeight));
    ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "%s",
      std::filesystem::path(g_app.queryImagePath).filename().string().c_str());
  } else {
    float placeholderHeight = totalHeight - controlsHeight - ImGui::GetCursorPosY();
    if (placeholderHeight > 100.0f) {
      ImGui::BeginChild("ImagePlaceholder", ImVec2(-1, placeholderHeight), true);
      renderCenteredText("Drag & drop an image here", "or use Browse button below");
      ImGui::EndChild();
    }
  }

  // Push controls to bottom
  float panelBottom = totalHeight - controlsHeight;
  if (ImGui::GetCursorPosY() < panelBottom)
    ImGui::SetCursorPosY(panelBottom);

  ImGui::Separator();
  ImGui::Spacing();

  // Query Image path
  ImGui::Text("Query Image:");
  ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("(or drag & drop)").x + ImGui::GetCursorPosX());
  ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(or drag & drop)");

  float buttonColWidth = ImGui::CalcTextSize("Set Directory").x + ImGui::GetStyle().FramePadding.x * 2 + ImGui::GetStyle().ItemSpacing.x;
  float pathFieldWidth = ImGui::GetContentRegionAvail().x - buttonColWidth;
  std::string truncPath = truncatePathMiddle(g_app.queryImagePath, pathFieldWidth - ImGui::GetStyle().FramePadding.x * 2);
  ImGui::SetNextItemWidth(pathFieldWidth);
  char truncBuf[512];
  strncpy(truncBuf, truncPath.c_str(), sizeof(truncBuf) - 1);
  truncBuf[sizeof(truncBuf) - 1] = '\0';
  ImGui::InputText("##querypath", truncBuf, sizeof(truncBuf), ImGuiInputTextFlags_ReadOnly);
  if (ImGui::IsItemHovered() && g_app.queryImagePath[0] != '\0')
    ImGui::SetTooltip("%s", g_app.queryImagePath);
  ImGui::SameLine();
  if (ImGui::Button("Browse...", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
    std::string path = openFileDialog();
    if (!path.empty()) {
      loadQueryImage(path);
      g_app.statusMessage = "Loaded: " + std::filesystem::path(path).filename().string();
    }
  }

  // Database Directory
  ImGui::Spacing();
  ImGui::Text("Database:");
  ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("(point to your image library)").x + ImGui::GetCursorPosX());
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.4f, 1.0f), "(point to your image library)");
  ImGui::SetNextItemWidth(-buttonColWidth);
  ImGui::InputText("##dbpath", g_app.imageDatabaseDir, sizeof(g_app.imageDatabaseDir));
  ImGui::SameLine();
  if (ImGui::Button("Set Directory##db", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
    std::string path = openFolderDialog();
    if (!path.empty())
      strncpy(g_app.imageDatabaseDir, path.c_str(), sizeof(g_app.imageDatabaseDir) - 1);
  }

  // CSV file for DNN embedding
  if (g_app.selectedFeatureType == DNNEmbedding) {
    ImGui::Text("CSV File:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##csvpath", g_app.csvFilePath, sizeof(g_app.csvFilePath));
  }

  // Feature Type
  ImGui::Spacing();
  ImGui::Text("Feature Type:");
  ImGui::SetNextItemWidth(-1);
  if (ImGui::Combo("##featuretype", &g_app.selectedFeatureType, featureTypeNames, FeatureTypeCount))
    g_app.embeddingsLoaded = false;

  // Results slider
  ImGui::Spacing();
  ImGui::Text("Results:");
  ImGui::SetNextItemWidth(-1);
  ImGui::SliderInt("##numresults", &g_app.numResultsToShow, 3, 20);

  // Status + Search button at bottom
  float buttonWidth = 120 * g_app.dpiScale;
  float buttonHeight = 30 * g_app.dpiScale;
  float remaining = ImGui::GetContentRegionAvail().y - buttonHeight - ImGui::GetStyle().ItemSpacing.y;
  if (remaining > 0) ImGui::Dummy(ImVec2(0, remaining));

  float rowY = ImGui::GetCursorPosY();
  ImGui::SetCursorPosY(rowY + buttonHeight - ImGui::GetTextLineHeight());
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", g_app.statusMessage.c_str());
  ImGui::SetCursorPos(ImVec2(ImGui::GetContentRegionMax().x - buttonWidth, rowY));
  if (ImGui::Button("Search", ImVec2(buttonWidth, buttonHeight)))
    performSearch();
}

void renderResultsGrid() {
  float minPadding = 10.0f * g_app.dpiScale;
  float panelWidth = ImGui::GetContentRegionAvail().x;

  int columns = std::max(1, std::min(2, static_cast<int>(panelWidth / (200.0f * g_app.dpiScale))));
  float thumbSize = (panelWidth - minPadding * (columns - 1)) / columns;
  float gap = (columns > 1) ? (panelWidth - thumbSize * columns) / (columns - 1) : 0.0f;

  int col = 0;
  for (size_t i = 0; i < g_app.results.size(); i++) {
    const auto& result = g_app.results[i];
    ImGui::BeginGroup();

    if (result.textureId != 0) {
      float aspectRatio = static_cast<float>(result.width) / result.height;
      float displayWidth = thumbSize, displayHeight = displayWidth / aspectRatio;
      if (displayHeight > thumbSize) {
        displayHeight = thumbSize;
        displayWidth = displayHeight * aspectRatio;
      }
      ImGui::Image((ImTextureID)(intptr_t)result.textureId, ImVec2(displayWidth, displayHeight));
    }

    float imageRightAbs = ImGui::GetItemRectMax().x;

    // Filename (left) and rank+distance (right)
    std::string displayName = result.filename;
    if (displayName.length() > 16) displayName = displayName.substr(0, 13) + "...";
    char rankText[32];
    snprintf(rankText, sizeof(rankText), "#%d  %.4f", static_cast<int>(i + 1), result.distance);

    ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "%s", displayName.c_str());
    ImGui::SameLine();
    float rankTextWidth = ImGui::CalcTextSize(rankText).x;
    ImVec2 cursorScreen = ImGui::GetCursorScreenPos();
    float windowRightEdge = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    float textRightEdge = std::min(imageRightAbs, windowRightEdge);
    ImGui::SetCursorScreenPos(ImVec2(textRightEdge - rankTextWidth, cursorScreen.y));
    ImVec4 rankColor = (i == 0) ? ImVec4(1.0f, 0.8f, 0.0f, 1.0f) : ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
    ImGui::TextColored(rankColor, "%s", rankText);

    ImGui::EndGroup();

    col++;
    if (col < columns && i < g_app.results.size() - 1) {
      ImGui::SameLine(0, gap);
    } else {
      col = 0;
      ImGui::Spacing();
      ImGui::Spacing();
    }
  }
}

void renderRightPanel() {
  ImGui::Text("Top Matches (sorted by distance):");
  ImGui::Spacing();

  if (g_app.hasResults && !g_app.results.empty()) {
    renderResultsGrid();
  } else if (!g_app.isSearching) {
    renderCenteredText("No results yet.", "Select a query image and click Search.");
  }
}

// ============================================================================
// Main UI
// ============================================================================

void renderUI() {
  ImGuiIO& io = ImGui::GetIO();
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(io.DisplaySize);

  ImGui::Begin("CBIR", nullptr,
    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
    ImGuiWindowFlags_NoBringToFrontOnFocus);

  // Keyboard shortcuts (only when not typing in a text field)
  if (!io.WantTextInput) {
    if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter))
      performSearch();
    if (ImGui::IsKeyPressed(ImGuiKey_Q))
      glfwSetWindowShouldClose(glfwGetCurrentContext(), GLFW_TRUE);
  }

  ImGui::PushFont(io.Fonts->Fonts[0]);
  ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Content-Based Image Retrieval");
  ImGui::PopFont();

  float totalWidth = ImGui::GetContentRegionAvail().x;
  float totalHeight = ImGui::GetContentRegionAvail().y;
  float splitterWidth = 6.0f * g_app.dpiScale;
  float leftW = totalWidth * g_app.splitRatio - splitterWidth * 0.5f;
  float rightW = totalWidth * (1.0f - g_app.splitRatio) - splitterWidth * 0.5f;

  // Left panel
  ImGui::BeginChild("LeftPanel", ImVec2(leftW, totalHeight), true);
  renderLeftPanel(totalHeight);
  ImGui::EndChild();

  // Splitter
  ImGui::SameLine();
  ImGui::InvisibleButton("##splitter", ImVec2(splitterWidth, totalHeight));
  if (ImGui::IsItemActive()) {
    g_app.splitRatio += ImGui::GetIO().MouseDelta.x / totalWidth;
    g_app.splitRatio = std::clamp(g_app.splitRatio, 0.2f, 0.6f);
  }
  if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::GetWindowDrawList()->AddRectFilled(
      ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(100, 150, 255, 120));
  } else {
    ImGui::GetWindowDrawList()->AddRectFilled(
      ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(80, 80, 80, 100));
  }

  // Right panel
  ImGui::SameLine();
  ImGui::BeginChild("RightPanel", ImVec2(rightW, totalHeight), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
  renderRightPanel();
  ImGui::EndChild();

  ImGui::End();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  glfwSetErrorCallback(errorCallback);
  if (!glfwInit()) { std::println(stderr, "Failed to initialize GLFW"); return 1; }

  // DPI scale
  float xscale = 1.0f, yscale = 1.0f;
  if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) glfwGetMonitorContentScale(mon, &xscale, &yscale);
  g_app.dpiScale = xscale;

  int windowWidth = static_cast<int>(1400 * xscale);
  int windowHeight = static_cast<int>(900 * xscale);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "CBIR - Content-Based Image Retrieval", nullptr, nullptr);
  if (!window) { std::println(stderr, "Failed to create GLFW window"); glfwTerminate(); return 1; }

  // Center window on screen
  if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) {
    const GLFWvidmode* mode = glfwGetVideoMode(mon);
    glfwSetWindowPos(window, (mode->width - windowWidth) / 2, (mode->height - windowHeight) / 2);
  }
  glfwShowWindow(window);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

#ifdef _WIN32
  {
    HWND hwnd = glfwGetWin32Window(window);
    BOOL useDarkMode = TRUE;
    DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &useDarkMode, sizeof(useDarkMode));
    COLORREF captionColor = RGB(25, 25, 30);
    DwmSetWindowAttribute(hwnd, 35 /*DWMWA_CAPTION_COLOR*/, &captionColor, sizeof(captionColor));
  }
#endif

  // Window icon
  {
    cv::Mat iconImg = cv::imread("src/gui/app_icon.png", cv::IMREAD_UNCHANGED);
    if (iconImg.empty())
      iconImg = cv::imread(std::filesystem::path(argv[0]).parent_path().string() + "/../src/gui/app_icon.png", cv::IMREAD_UNCHANGED);
    if (!iconImg.empty()) {
      cv::cvtColor(iconImg, iconImg, cv::COLOR_BGRA2RGBA);
      GLFWimage glfwIcon{iconImg.cols, iconImg.rows, iconImg.data};
      glfwSetWindowIcon(window, 1, &glfwIcon);
    }
  }

  glfwSetDropCallback(window, dropCallback);

  // ImGui setup
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.Fonts->AddFontDefault();
  io.FontGlobalScale = xscale;

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(xscale);
  style.WindowRounding = 0.0f;
  style.FrameRounding = 4.0f;
  style.GrabRounding = 4.0f;
  style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
  style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
  style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.29f, 0.48f, 0.54f);

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL2_Init();

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    renderUI();

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // Cleanup
  freeTexture(g_app.queryTextureId);
  for (auto& r : g_app.results) freeTexture(r.textureId);
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
