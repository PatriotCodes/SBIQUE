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