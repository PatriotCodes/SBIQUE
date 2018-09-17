#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

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
    String imageName("img/original-scaled-image.jpg");
    String noiseImageName("img/gaussian_noise.jpg");
    if( argc > 1)
    {
        imageName = argv[1];
    }
    Mat image;
    Mat image2;
    image = imread( imageName, IMREAD_COLOR );
    Mat mColorNoise(image.size(),image.type()); 
    if( image.empty()) 
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    AddGaussianNoise_Opencv(image,mColorNoise,0,10.0);
    imwrite(noiseImageName, mColorNoise);
    image2 = imread(noiseImageName, IMREAD_COLOR);
    cout << "MSSIM score: " + to_string(getMSSIM(image, image2)) << endl;
    cout << "PSNR score: " + to_string(getPSNR(image, image2)) << endl;
    cout << "BRISQUE score (original): " + to_string(getBRISQUE(imageName)) << endl;
    cout << "BRISQUE score (noise): " + to_string(getBRISQUE(noiseImageName)) << endl;
    waitKey(0);
    return 0;
}