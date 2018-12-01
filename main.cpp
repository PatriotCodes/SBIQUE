#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>

#include "metrics.h"

using namespace cv;
using namespace std;

bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst,double Mean=0.0, double StdDev=10.0) {
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

string percentageIncrease(double originalValue, double NewValue) {
    double increase = NewValue - originalValue;
    double increasePerc = increase / originalValue * 100;
    return to_string(increasePerc) + "%";
}

string percentageDecrease(double originalValue, double NewValue) {
  double decrease = originalValue - NewValue;
  double increasePerc = decrease / originalValue * 100;
  return to_string(increasePerc) + "%";
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
    string gNoiseName2 = originalPath + "_gaussian_noise_high.jpg";
    string gBlurName = originalPath + "_gaussian_blur.jpg";
    string gBlurName2 = originalPath + "_gaussian_blur_high.jpg";
    string gBilateralName = originalPath + "_bilateral_filter.jpg";
    string gBilateralName2 = originalPath + "_bilateral_filter_high.jpg";
    string gNlName = originalPath + "_non_local.jpg";
    string gNlName2 = originalPath + "_non_local_high.jpg";

    cout << "Applying Gaussian noise " << endl;
    Mat gNoise(originalImage.size(),originalImage.type());
    AddGaussianNoise_Opencv(originalImage,gNoise,0,10.0);
    imwrite(gNoiseName, gNoise);
    Mat gNoise2(originalImage.size(),originalImage.type());
    AddGaussianNoise_Opencv(originalImage,gNoise2,0,25.0);
    imwrite(gNoiseName2, gNoise2);

    cout << "Removing noise using Gaussian blur " << endl;
    Mat gBlur(originalImage.size(),originalImage.type());
    GaussianBlur(gNoise, gBlur, Size( 3, 3 ), 0, 0 );
    imwrite(gBlurName, gBlur);
    Mat gBlur2(originalImage.size(),originalImage.type());
    GaussianBlur(gNoise2, gBlur2, Size( 3, 3 ), 0, 0 );
    imwrite(gBlurName2, gBlur2);

    cout << "Removing noise using bilateral filter " << endl;
    Mat gBilateral(originalImage.size(),originalImage.type());
    bilateralFilter (gNoise, gBilateral, 8, 20, 20 );
    imwrite(gBilateralName, gBilateral);
    Mat gBilateral2(originalImage.size(),originalImage.type());
    bilateralFilter (gNoise2, gBilateral2, 16, 40, 40 );
    imwrite(gBilateralName2, gBilateral2);

    cout << "Removing noise using non-local-means algorythm " << endl;
    Mat gNl(originalImage.size(),originalImage.type());
    fastNlMeansDenoisingColored(gNoise, gNl, 3, 7, 21 );
    imwrite(gNlName, gNl);
    Mat gNl2(originalImage.size(),originalImage.type());
    fastNlMeansDenoisingColored(gNoise2, gNl2, 3, 7, 21 );
    imwrite(gNlName2, gNl2);

    cout << "Getting scores " << endl;
    output += originalPath;
    output += '\n';

    output += "Low noise level: ";
    output += '\n';

    output += getPSNRscores(originalImage,gNoise,gBlur,gBilateral,gNl);
    output += getMSSIMscores(originalImage,gNoise,gBlur,gBilateral,gNl);
    output += getBRISQUEscores(gNoise,gBlur,gBilateral,gNl);

    output += "High noise level: ";
    output += '\n';

    output += getPSNRscores(originalImage,gNoise2,gBlur2,gBilateral2,gNl2);
    output += getMSSIMscores(originalImage,gNoise2,gBlur2,gBilateral2,gNl2);
    output += getBRISQUEscores(gNoise2,gBlur2,gBilateral2,gNl2);
    
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