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
  fs::create_directory("output/results");
  fs::create_directory("output/results_total");
}

int main( int argc, char** argv ) {
  string path;
  string output = "";
  int fileIterator = 0;

  cout << "Please specify directory with images: ";
  cin >> path;

  cout << "Creating output directories: " << endl;
  createOutputDirectories();

  vector<ResultData> totalResults;
  int totalImages = 0;

  for (const auto & p : fs::directory_iterator(path)) {
    string originalPath = p.path();
    totalImages++;

    vector<ResultData> resultsNoise;
    vector<ResultData> resultsBlur;

    cout << "Processing " + originalPath << endl;
    Mat originalImage = imread(originalPath, IMREAD_COLOR);

    fileIterator++;
    string currentWorkingDirectory = "output/noise/" + to_string(fileIterator);
    fs::create_directory(currentWorkingDirectory);

    cout << "Prepairing distorted image variations" << endl;
    cout << "Applying Gaussian noise" << endl;
    Mat gNoise(originalImage.size(),originalImage.type());
    AddGaussianNoise_Opencv(originalImage,gNoise,0,12.0);
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
        resultsNoise.push_back(ResultData(filter_type, metric_type, percentage));
        totalResults.push_back(ResultData(filter_type, metric_type, percentage));
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

        FILTER_TYPE filter_type = FILTER_TYPE::UNSHARP_MASK;
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
        resultsBlur.push_back(ResultData(filter_type, metric_type, percentage));
        totalResults.push_back(ResultData(filter_type, metric_type, percentage));
    }

    for (int metricIterator = METRIC_TYPE::PSNR; metricIterator <= METRIC_TYPE::BRISQUE; metricIterator++) {
      METRIC_TYPE metric_type = static_cast<METRIC_TYPE>(metricIterator);
      string metricsDirectory = currentWorkingDirectory + "/" + metricToString(metric_type);
      fs::create_directory(metricsDirectory);
      imwrite(metricsDirectory + "/_blur.jpg", gBlur);

        FILTER_TYPE filter_type = FILTER_TYPE::KERNEL_SHARPENING;
        Mat bestResult(gBlur.size(),gBlur.type());
        sharpenWithKernel(gBlur, bestResult);
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
        resultsBlur.push_back(ResultData(filter_type, metric_type, percentage));
        totalResults.push_back(ResultData(filter_type, metric_type, percentage));
    }

    currentWorkingDirectory = "output/results/" + to_string(fileIterator);
    fs::create_directory(currentWorkingDirectory);

    cout << "Writing results on disk" << endl;
    ofstream outNoise(currentWorkingDirectory + "/results_noise.txt");
    outNoise << getTotalResults(resultsNoise);
    outNoise.close();
    ofstream outBlur(currentWorkingDirectory + "/results_blur.txt");
    outBlur << getTotalResults(resultsBlur);
    outBlur.close();
    cout << "Scores wrote on disk" << endl;
  }

  cout << "Writing total scores on disk" << endl;
  string totalOutpuDir = "output/results_total";
  fs::create_directory(totalOutpuDir);
  ofstream outTotalNoise(totalOutpuDir + "/results_noise.txt");
  for (int filterIterator = FILTER_TYPE::GAUSSIAN; filterIterator <= FILTER_TYPE::UNSHARP_MASK; filterIterator++) {
    FILTER_TYPE filter_type = static_cast<FILTER_TYPE>(filterIterator);
    outTotalNoise << getSummaryResults(totalResults, filter_type, totalImages);
    outTotalNoise << endl;
  }

  outTotalNoise.close();
  cout << "All images processed" << endl;
  cout << "Total images processed: " + to_string(totalImages) << endl;
  return 0;
}