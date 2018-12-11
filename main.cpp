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
#include "formatUtils.h"
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

enum FILTER_TYPE { BILATERAL, NLMEANS };

inline const string filterToString(FILTER_TYPE v) {
  switch (v) {
    case FILTER_TYPE::BILATERAL:    return "bilateral";
    case FILTER_TYPE::NLMEANS:   return "non-local-means";
    default:                   return "[Unknown FILTER_TYPE]";
  }
}

bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean=0.0, double StdDev=10.0) {
    if(mSrc.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return false;
    }
    Mat mSrc_16SC;
    Mat mGaussian_noise = Mat(mSrc.size(),CV_16SC3);
    randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));

    mSrc.convertTo(mSrc_16SC,CV_16SC3);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(mDst,mSrc.type());

    return true;
}

string getPSNRscores(Mat originalImage, Mat gNoise, Mat gBlur, Mat gBilateral, Mat gNl) {
  string output = "";
  double originalPSNR = getPSNR(originalImage, gNoise);
  output += "PSNR score (gaussian blur): ";
  double gaussianPSNR = getPSNR(originalImage, gBlur);
  output += percentageIncrease(originalPSNR,gaussianPSNR);
  output += '\n';
  output += "PSNR score (bilateral filter): ";
  double bilateralPSNR = getPSNR(originalImage, gBilateral);
  output += percentageIncrease(originalPSNR,bilateralPSNR);
  output += '\n';
  output += "PSNR score (non-local-means): ";
  double nlPSNR = getPSNR(originalImage, gNl);
  output += percentageIncrease(originalPSNR,nlPSNR);
  output += '\n';
  return output;
}

string getMSSIMscores(Mat originalImage, Mat gNoise, Mat gBlur, Mat gBilateral, Mat gNl) {
  string output = "";
  double originalMSSIM = getMSSIM(originalImage, gNoise);
  output += "MSSIM score (gaussian blur): ";
  double gaussianMSSIM = getMSSIM(originalImage, gBlur);
  output += percentageIncrease(originalMSSIM,gaussianMSSIM);
  output += '\n';
  output += "MSSIM score (bilateral filter): ";
  double bilateralMSSIM = getMSSIM(originalImage, gBilateral);
  output += percentageIncrease(originalMSSIM,bilateralMSSIM);
  output += '\n';
  output += "MSSIM score (non-local-means): ";
  double nlMSSIM = getMSSIM(originalImage, gNl);
  output += percentageIncrease(originalMSSIM,nlMSSIM);
  output += '\n';
  return output;
}

string getBRISQUEscores(Mat gNoise, Mat gBlur, Mat gBilateral, Mat gNl) {
  string output = "";
  double originalBRISQUE = getBRISQUE(gNoise);
  output += "BRISQUE score (gaussian blur): ";
  double gaussianBRISQUE = getBRISQUE(gBlur);
  output += percentageDecrease(originalBRISQUE,gaussianBRISQUE);
  output += '\n';
  output += "BRISQUE score (bilateral filter): ";
  double bilateralBRISQUE = getBRISQUE(gBilateral);
  output += percentageDecrease(originalBRISQUE,bilateralBRISQUE);
  output += '\n';
  output += "BRISQUE score (non-local-means): ";
  double nlBRISQUE = getBRISQUE(gNl);
  output += percentageDecrease(originalBRISQUE,nlBRISQUE);
  output += '\n';
  return output;
}

bool deleteFile(string fileName) {
  if(remove(fileName.c_str()) != 0 )
    return false;
  else
    return true;
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
      stepCounter++;
      cout << "Removing noise using " + filterToString(filter_type) + "(" + metricToString(metric_type) + ") filter step: " << stepCounter << endl;
      cout << "d: " << d << ";  g1: " << g1 << ";  g2: " << g2 << endl;
      Mat gProcessed(originalImage.size(),originalImage.type());
      switch (filter_type) {
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
      for (int filterIterator = FILTER_TYPE::BILATERAL; filterIterator <= FILTER_TYPE::NLMEANS; filterIterator++) {
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