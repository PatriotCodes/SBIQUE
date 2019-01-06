#ifndef BR_UTILS_H
#define BR_UTILS_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <string>
#include <iostream>

#include "metrics.h"
#include "sharpen.h"

using namespace std;
using namespace cv;

enum METRIC_TYPE { PSNR, MSSIM, BRISQUE };

inline const string metricToString(METRIC_TYPE v) {
  switch (v) {
    case METRIC_TYPE::PSNR:    return "PSNR";
    case METRIC_TYPE::MSSIM:   return "MSSIM";
    case METRIC_TYPE::BRISQUE: return "BRISQUE";
    default:                   return "[Unknown METRIC_TYPE]";
  }
}

enum FILTER_TYPE { GAUSSIAN, BILATERAL, NLMEANS, UNSHARP_MASK };

inline const string filterToString(FILTER_TYPE v) {
  switch (v) {
    case FILTER_TYPE::GAUSSIAN:     return "gaussian";
    case FILTER_TYPE::BILATERAL:    return "bilateral";
    case FILTER_TYPE::NLMEANS:      return "non-local-means";
    case FILTER_TYPE::UNSHARP_MASK: return "unsharp mask";
    default:                        return "[Unknown FILTER_TYPE]";
  }
}

struct ResultData {
  FILTER_TYPE filter_type;
  METRIC_TYPE metric_type;
  double score;

  ResultData(FILTER_TYPE in_filter_type, METRIC_TYPE in_metric_type, double in_score) : 
    filter_type(in_filter_type), metric_type(in_metric_type), score(in_score) {}

  string toString() {
    return filterToString(filter_type) + "  " + metricToString(metric_type) + "  score: " + to_string(score) + '\n';
  }
};

double percentageIncrease(double originalValue, double NewValue);
double percentageDecrease(double originalValue, double NewValue);
bool deleteFile(string fileName);
bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean=0.0, double StdDev=10.0);
Mat findBestParams(Mat originalImage, Mat distortedImage, METRIC_TYPE metric_type, FILTER_TYPE filter_type);
string getTotalResults(vector<ResultData> results);
string getSummaryResults(vector<ResultData> results, FILTER_TYPE filter_type, int totalImages);

namespace privateFunctions {
	double getScore(METRIC_TYPE metric_type, Mat originalImage, Mat processedImage);
	Mat getProcessedImage(Mat original_image, Mat distorted_image, FILTER_TYPE filter_type, int d, int g1, int g2);
}

using namespace privateFunctions;

#endif