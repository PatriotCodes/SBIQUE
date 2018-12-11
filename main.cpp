#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>
#include <stdio.h>

#include "metrics.h"
#include "utils.h"
#include "sharpen.h"

using namespace cv;
using namespace std;

namespace fs = std::experimental::filesystem;

enum METRIC_TYPE { PSNR, MSSIM, BRISQUE };

inline const string metricToString(METRIC_TYPE v) {
  switch (v) {
    case METRIC_TYPE::PSNR:    return "PSNR";
    case METRIC_TYPE::MSSIM:   return "MSSIM";
    case METRIC_TYPE::BRISQUE: return "BRISQUE";
    default:                   return "[Unknown METRIC_TYPE]";
  }
}

enum FILTER_TYPE { GAUSSIAN, BILATERAL, NLMEANS };

inline const string filterToString(FILTER_TYPE v) {
  switch (v) {
    case FILTER_TYPE::GAUSSIAN:   return "gaussian";
    case FILTER_TYPE::BILATERAL:  return "bilateral";
    case FILTER_TYPE::NLMEANS:    return "non-local-means";
    default:                      return "[Unknown FILTER_TYPE]";
  }
}

void createOutputDirectories() {
  fs::create_directory("output");
  fs::create_directory("output/noise");
  fs::create_directory("output/blur");
}

Mat findBestParams(Mat originalImage, Mat distortedImage, METRIC_TYPE metric_type, FILTER_TYPE filter_type) {
  int d = 8, pd = 0;
  int g1 = 20, g2 = 20;
  int pg1 = 0, pg2 = 0;
  int phase = 0;
  int stepCounter = 0;
  while (d != pd || g1 != pg1 || g2 != pg2) { 
    int *i;
    if (phase == 0) {
      pd = d;
      i = &d;
    } else if (phase == 1) {
      pg1 = g1;
      i = &g1;
    } else {
      pg2 = g2;
      i = &g2;
    }
    double pMsim = 0;
    double maxMsim = -100000;
    int maxI = 0;
    for (*i = 1; *i < 50; *i += 1) {
      if (filter_type == FILTER_TYPE::GAUSSIAN && ((*i) % 2 == 0)) {
        continue;
      }
      stepCounter++;
      cout << "Removing noise using " + filterToString(filter_type) + "(" + metricToString(metric_type) + ") filter step: " << stepCounter << endl;
      cout << "d: " << d << ";  g1: " << g1 << ";  g2: " << g2 << endl;
      Mat gProcessed(originalImage.size(),originalImage.type());
      switch (filter_type) {
        case FILTER_TYPE::GAUSSIAN : {
          GaussianBlur(distortedImage, gProcessed, Size( d, d ), g1, g2 );
          break;
        }
        case FILTER_TYPE::BILATERAL : {
          bilateralFilter(distortedImage, gProcessed, d, g1, g2 );
          break;
        }
        case FILTER_TYPE::NLMEANS : {
          fastNlMeansDenoisingColored(distortedImage, gProcessed, d, g1, g2 );
          break;
        }
      }
      double score = 0;
      switch (metric_type) {
        case METRIC_TYPE::PSNR : {
          score = getPSNR(originalImage, gProcessed);
          break;
        }
        case METRIC_TYPE::MSSIM : {
          score = getMSSIM(originalImage, gProcessed);
          break;
        }
        case METRIC_TYPE::BRISQUE : {
          score = -getBRISQUE(gProcessed);
          break;
        }
      }
      cout << "Score: " << score << endl;
      if (score > maxMsim) {
        maxMsim = score;
        maxI = (*i);
      }
      if (pMsim > score && (*i) > 10)
        break;
      pMsim = score;
    }
    *i = maxI;
    phase = (phase >= 2) ? 0 : (phase + 1);
  }
  cout << "Best params" << endl;
  cout << "d: " << d << ";  g1: " << g1 << ";  g2: " << g2 << endl;
  Mat gProcessed(originalImage.size(),originalImage.type());
  switch (filter_type) {
    case FILTER_TYPE::GAUSSIAN : {
      GaussianBlur(distortedImage, gProcessed, Size( d, d ), g1, g2 );
      break;
    }
    case FILTER_TYPE::BILATERAL : {
      bilateralFilter(distortedImage, gProcessed, d, g1, g2 );
      break;
    }
    case FILTER_TYPE::NLMEANS : {
      fastNlMeansDenoisingColored(distortedImage, gProcessed, d, g1, g2 );
      break;
    }
  }
  return gProcessed;
}

int main( int argc, char** argv ) {
  string path;
  string output = "";
  int fileIterator = 0;

  cout << "Please specify directory with images: ";
  cin >> path;

  cout << "Creating output directories: " << endl;
  createOutputDirectories();

  for (const auto & p : fs::directory_iterator(path)) {
    string originalPath = p.path();

    cout << "String noise removal tests: " << endl;
    cout << "Processing " + originalPath << endl;
    Mat originalImage = imread(originalPath, IMREAD_COLOR);

    fileIterator++;
    string currentWorkingDirectory = "output/noise/" + to_string(fileIterator);
    fs::create_directory(currentWorkingDirectory);

    // string blurredName = originalPath + "_gaussian_blur.jpg";
    // string unsharpMaskName = originalPath + "_unsharp_mask.jpg";
    // cout << "Blurring image with gaussian blur " << endl;
    // Mat blurred;
    // GaussianBlur(originalImage, blurred, Size(), 1, 1);
    // imwrite(blurredName, blurred);
    // Mat sharpened = unsharpMask(blurred,1,5,2);
    // imwrite(unsharpMaskName, sharpened);

    cout << "Applying Gaussian noise " << endl;
    Mat gNoise(originalImage.size(),originalImage.type());
    AddGaussianNoise_Opencv(originalImage,gNoise,0,25.0);

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
      }
    }

  }
  cout << "All images processed" << endl;
  // ofstream out("output.txt");
  // out << output;
  // out.close();
  // cout << "Scores can be found in output.txt" << endl;
  return 0;
}