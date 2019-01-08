#include "utils.h"

double percentageIncrease(double originalValue, double NewValue) {
   double increase = NewValue - originalValue;
   return increase / originalValue * 100;
}

double percentageDecrease(double originalValue, double NewValue) {
  double decrease = originalValue - NewValue;
  return decrease / originalValue * 100;
}

bool deleteFile(string fileName) {
  if(remove(fileName.c_str()) != 0 )
    return false;
  else
    return true;
}

bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean, double StdDev) {
    Mat mSrc_16SC;
    Mat mGaussian_noise = Mat(mSrc.size(),CV_16SC3);
    randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));
    mSrc.convertTo(mSrc_16SC,CV_16SC3);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(mDst,mSrc.type());
    return true;
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
      cout << "Processing image using " + filterToString(filter_type) + "(" + metricToString(metric_type) + ") filter step: " << stepCounter << endl;
      cout << "d: " << d << ";  g1: " << g1 << ";  g2: " << g2 << endl;
      Mat gProcessed = getProcessedImage(originalImage, distortedImage, filter_type, d, g1, g2);
      double score = getScore(metric_type, originalImage, gProcessed);
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
  Mat gProcessed = getProcessedImage(originalImage, distortedImage, filter_type, d, g1, g2);
  return gProcessed;
}

string getTotalResults(vector<ResultData> results) {
  string output = "";
  for (auto &result : results) {
    output += result.toString();
    output += '\n';
  }
  return output;
}

string getSummaryResults(vector<ResultData> results, FILTER_TYPE filter_type, int totalImages) {
  string output = "";
  double scorePSNR = 0;
  double scoreMSSIM = 0;
  double scoreBRISQUE = 0;
  for (auto &result : results) {
    if (result.filter_type == filter_type) {
      switch (result.metric_type) {
        case METRIC_TYPE::PSNR : {
          scorePSNR += result.score;
          break;
        }
        case METRIC_TYPE::MSSIM : {
          scoreMSSIM += result.score;
          break;
        }
        case METRIC_TYPE::BRISQUE : {
          scoreBRISQUE += result.score;
          break;
        }
      }
    }
  }
  scorePSNR = scorePSNR / totalImages;
  scoreBRISQUE = scoreBRISQUE / totalImages;
  scoreMSSIM = scoreMSSIM / totalImages;
  output += "Total scores for " + filterToString(filter_type) + ": " + '\n';
  output += "PSNR: " + to_string(scorePSNR) + '\n';
  output += "MSSIM: " + to_string(scoreMSSIM) + '\n';
  output += "BRISQUE: " + to_string(scoreBRISQUE) + '\n';
  return output;
}

namespace privateFunctions {

  double getScore(METRIC_TYPE metric_type, Mat originalImage, Mat processedImage) {
    double score = 0;
    switch (metric_type) {
      case METRIC_TYPE::PSNR : {
        score = getPSNR(originalImage, processedImage);
        break;
      }
      case METRIC_TYPE::MSSIM : {
        score = getMSSIM(originalImage, processedImage);
        break;
      }
      case METRIC_TYPE::BRISQUE : {
        score = -getBRISQUE(processedImage);
        break;
      }
    }
    return score;
  }

  Mat getProcessedImage(Mat original_image, Mat distorted_image, FILTER_TYPE filter_type, int d, int g1, int g2) {
    Mat gProcessed(original_image.size(), original_image.type());
    switch (filter_type) {
      case FILTER_TYPE::GAUSSIAN : {
        GaussianBlur(distorted_image, gProcessed, Size( d, d ), g1, g2);
        break;
      }
      case FILTER_TYPE::BILATERAL : {
        bilateralFilter(distorted_image, gProcessed, d, g1, g2 );
        break;
      }
      case FILTER_TYPE::NLMEANS : {
        fastNlMeansDenoisingColored(distorted_image, gProcessed, d, g1, g2);
        break;
      }
      case FILTER_TYPE::UNSHARP_MASK : {
        gProcessed = unsharpMask(distorted_image, d, g1 ,g2);
        break;
      }
    }
    return gProcessed;
  }

}