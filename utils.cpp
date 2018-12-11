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
        case FILTER_TYPE::UNSHARP_MASK : {
          gProcessed = unsharpMask(distortedImage,d,g1,g2);
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
    case FILTER_TYPE::UNSHARP_MASK : {
      gProcessed = unsharpMask(distortedImage,d,g1,g2);
      break;
    }
  }
  return gProcessed;
}