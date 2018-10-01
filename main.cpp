#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>

#include "metrics.h"

using namespace cv;
using namespace std;

bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst,double Mean=0.0, double StdDev=10.0)
{
    if(mSrc.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }
    Mat mSrc_16SC;
    Mat mGaussian_noise = Mat(mSrc.size(),CV_16SC3);
    randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));

    mSrc.convertTo(mSrc_16SC,CV_16SC3);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(mDst,mSrc.type());

    return true;
}

struct testImage {
  string originalPath;
  Mat originalImage;
  Mat noiseImage;
};

int main( int argc, char** argv )
{
  string path;
  cout << "Please specify directory with images: ";
  cin >> path;
  vector<testImage> images;
  for (const auto & p : std::experimental::filesystem::directory_iterator(path)) {
      testImage tmp;
      tmp.originalPath = p.path();
      tmp.originalImage = imread(tmp.originalPath, IMREAD_COLOR);
      Mat mColorNoise(tmp.originalImage.size(),tmp.originalImage.type());
      string noiseName = tmp.originalPath + "_noise.jpg";
      cout << "Applying noise to " + tmp.originalPath << endl;
      AddGaussianNoise_Opencv(tmp.originalImage,mColorNoise,0,10.0);
      imwrite(noiseName, mColorNoise);
      tmp.noiseImage = imread(noiseName, IMREAD_COLOR);
      images.push_back(tmp);
  }
  cout << "Noise was applied to all images" << endl;
  string output = "";
  for (auto image : images) {
    cout << "Processing " + image.originalPath << endl;
    output += image.originalPath;
    output += '\n';
    output += "PSNR score: ";
    output += to_string(getPSNR(image.originalImage, image.noiseImage));
    output += '\n';
    output += "MSSIM score: ";
    output += to_string(getMSSIM(image.originalImage, image.noiseImage));
    output += '\n';
    output += "BRISQUE score (original): ";
    output += to_string(getBRISQUE(image.originalImage));
    output += '\n';
    output += "BRISQUE score (noise): ";
    output += to_string(getBRISQUE(image.noiseImage));
    output += '\n';
    output += '\n';
  }
  cout << "All images processed" << endl;
  ofstream out("output.txt");
  out << output;
  out.close();
  cout << "Scores can be found in output.txt" << endl;
  return 0;
}