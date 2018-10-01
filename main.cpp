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

int main( int argc, char** argv )
{
  string path;
  cout << "Please specify directory with images: ";
  cin >> path;
  string output = "";
  for (const auto & p : std::experimental::filesystem::directory_iterator(path)) {
    string originalPath = p.path();
    cout << "Processing " + originalPath << endl;
    Mat originalImage = imread(originalPath, IMREAD_COLOR);

    string gNoiseName = originalPath + "_gaussian_noise.jpg";
    string gBlurName = originalPath + "_gaussian_blur.jpg";
    string gBilateralName = originalPath + "_bilateral_filter.jpg";

    cout << "Applying Gaussian noise " << endl;
    Mat gNoise(originalImage.size(),originalImage.type());
    AddGaussianNoise_Opencv(originalImage,gNoise,0,10.0);
    imwrite(gNoiseName, gNoise);

    cout << "Removing noise using Gaussian blur " << endl;
    Mat gBlur(originalImage.size(),originalImage.type());
    GaussianBlur(originalImage, gBlur, Size( 3, 3 ), 0, 0 );
    imwrite(gBlurName, gBlur);

    cout << "Removing noise using bilateral filter " << endl;
    Mat gBilateral(originalImage.size(),originalImage.type());
    bilateralFilter (originalImage, gBilateral, 2, 4, 4 );
    imwrite(gBilateralName, gBilateral);

    cout << "Getting scores " << endl;
    output += originalPath;
    output += '\n';

    output += "PSNR score (noise): ";
    output += to_string(getPSNR(originalImage, gNoise));
    output += '\n';
    output += "PSNR score (gaussian blur): ";
    output += to_string(getPSNR(originalImage, gBlur));
    output += '\n';
    output += "PSNR score (bilateral filter): ";
    output += to_string(getPSNR(originalImage, gBilateral));
    output += '\n';

    output += "MSSIM score (noise): ";
    output += to_string(getMSSIM(originalImage, gNoise));
    output += '\n';
    output += "MSSIM score (gaussian blur): ";
    output += to_string(getMSSIM(originalImage, gBlur));
    output += '\n';
    output += "MSSIM score (bilateral filter): ";
    output += to_string(getMSSIM(originalImage, gBilateral));
    output += '\n';

    output += "BRISQUE score (original): ";
    output += to_string(getBRISQUE(originalImage));
    output += '\n';
    output += "BRISQUE score (noise): ";
    output += to_string(getBRISQUE(gNoise));
    output += '\n';
    output += "BRISQUE score (gaussian blur): ";
    output += to_string(getBRISQUE(gBlur));
    output += '\n';
    output += "BRISQUE score (bilateral filter): ";
    output += to_string(getBRISQUE(gBilateral));
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