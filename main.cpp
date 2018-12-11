#include <opencv2/core/core.hpp>

#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>
#include <stdio.h>

#include "metrics.h"
#include "utils.h"

using namespace cv;
using namespace std;

namespace fs = std::experimental::filesystem;

void createOutputDirectories() {
  fs::create_directory("output");
  fs::create_directory("output/noise");
  fs::create_directory("output/blur");
}

struct ResultData {
  FILTER_TYPE filter_type;
  METRIC_TYPE metric_type;
  double score;

  ResultData(FILTER_TYPE in_filter_type, METRIC_TYPE in_metric_type, double in_score) : 
    filter_type(in_filter_type), metric_type(in_metric_type), score(in_score) {}
};

int main( int argc, char** argv ) {
  string path;
  string output = "";
  int fileIterator = 0;
  vector<ResultData> results;

  cout << "Please specify directory with images: ";
  cin >> path;

  cout << "Creating output directories: " << endl;
  createOutputDirectories();

  for (const auto & p : fs::directory_iterator(path)) {
    string originalPath = p.path();

    cout << "Processing " + originalPath << endl;
    Mat originalImage = imread(originalPath, IMREAD_COLOR);

    fileIterator++;
    string currentWorkingDirectory = "output/noise/" + to_string(fileIterator);
    fs::create_directory(currentWorkingDirectory);

    cout << "Prepairing distorted image variations" << endl;
    cout << "Applying Gaussian noise" << endl;
    Mat gNoise(originalImage.size(),originalImage.type());
    AddGaussianNoise_Opencv(originalImage,gNoise,0,25.0);
    cout << "Blurring image" << endl;
    Mat gBlur;
    GaussianBlur(originalImage, gBlur, Size(), 1, 1);

    cout << "Starting tests on noise removal: " << endl;
    for (int metricIterator = METRIC_TYPE::PSNR; metricIterator <= METRIC_TYPE::BRISQUE; metricIterator++) {
      METRIC_TYPE metric_type = static_cast<METRIC_TYPE>(metricIterator);
      string metricsDirectory = currentWorkingDirectory + "/" + metricToString(metric_type);
      fs::create_directory(metricsDirectory);
      imwrite(metricsDirectory + "/_noise.jpg", gNoise);
      for (int filterIterator = FILTER_TYPE::GAUSSIAN; filterIterator <= FILTER_TYPE::NLMEANS; filterIterator++) {
        FILTER_TYPE filter_type = static_cast<FILTER_TYPE>(filterIterator);
        Mat bestResult = findBestParams(originalImage, gNoise, metric_type, filter_type);
        imwrite(metricsDirectory + "/" + filterToString(filter_type) + ".jpg", bestResult);
        double percentage = 0;
        switch (metric_type) {
          case METRIC_TYPE::PSNR : {
            double originalPSNR = getPSNR(originalImage, gNoise);
            double restoredPSNR = getPSNR(originalImage, bestResult);
            percentage = percentageIncrease(originalPSNR, restoredPSNR);
            break;
          }
          case METRIC_TYPE::MSSIM : {
            double originalMSSIM = getMSSIM(originalImage, gNoise);
            double restoredMSSIM = getMSSIM(originalImage, bestResult);
            percentage = percentageIncrease(originalMSSIM, restoredMSSIM);
            break;
          }
          case METRIC_TYPE::BRISQUE : {
            double originalBRISQUE = getBRISQUE(gNoise);
            double restoredBRISQUE = getBRISQUE(bestResult);
            percentage = percentageDecrease(originalBRISQUE, restoredBRISQUE);
            break;
          }
        }
        cout << "percentage increase:" + to_string(percentage) + "%" << endl;
        results.push_back(ResultData(filter_type, metric_type, percentage));
      }
    }

    currentWorkingDirectory = "output/blur/" + to_string(fileIterator);
    fs::create_directory(currentWorkingDirectory);
    cout << "Starting tests on deblurring: " << endl;
    for (int metricIterator = METRIC_TYPE::PSNR; metricIterator <= METRIC_TYPE::BRISQUE; metricIterator++) {
      METRIC_TYPE metric_type = static_cast<METRIC_TYPE>(metricIterator);
      string metricsDirectory = currentWorkingDirectory + "/" + metricToString(metric_type);
      fs::create_directory(metricsDirectory);
      imwrite(metricsDirectory + "/_blur.jpg", gBlur);
      for (int filterIterator = FILTER_TYPE::UNSHARP_MASK; filterIterator <= FILTER_TYPE::UNSHARP_MASK; filterIterator++) {
        FILTER_TYPE filter_type = static_cast<FILTER_TYPE>(filterIterator);
        Mat bestResult = findBestParams(originalImage, gBlur, metric_type, filter_type);
        imwrite(metricsDirectory + "/" + filterToString(filter_type) + ".jpg", bestResult);
        double percentage = 0;
        switch (metric_type) {
          case METRIC_TYPE::PSNR : {
            double originalPSNR = getPSNR(originalImage, gNoise);
            double restoredPSNR = getPSNR(originalImage, bestResult);
            percentage = percentageIncrease(originalPSNR, restoredPSNR);
            break;
          }
          case METRIC_TYPE::MSSIM : {
            double originalMSSIM = getMSSIM(originalImage, gNoise);
            double restoredMSSIM = getMSSIM(originalImage, bestResult);
            percentage = percentageIncrease(originalMSSIM, restoredMSSIM);
            break;
          }
          case METRIC_TYPE::BRISQUE : {
            double originalBRISQUE = getBRISQUE(gNoise);
            double restoredBRISQUE = getBRISQUE(bestResult);
            percentage = percentageDecrease(originalBRISQUE, restoredBRISQUE);
            break;
          }
        }
        cout << "percentage increase:" + to_string(percentage) + "%" << endl;
        results.push_back(ResultData(filter_type, metric_type, percentage));
      }
    }

  }

  cout << "All images processed" << endl;
  // ofstream out("output/results.txt");
  // out << output;
  // out.close();
  // cout << "Scores can be found in output.txt" << endl;
  return 0;
}