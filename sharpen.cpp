#include "sharpen.h"

Mat unsharpMask(const Mat image, double sigma, double threshold, double amount) {
	Mat blurred;
	Mat img = image.clone();	
	GaussianBlur(img, blurred, Size(), sigma, sigma);
	Mat lowContrastMask = abs(img - blurred) < threshold;
	Mat sharpened = img * ( 1 + amount ) + blurred * ( -amount);
	img.copyTo(sharpened, lowContrastMask);
	return sharpened;
}

void sharpenWithKernel(const cv::Mat &image, cv::Mat &result) {

	// Construct kernel (all entries initialized to 0)
	Mat kernel(3,3,CV_32F,Scalar(0));
	// assigns kernel values
	kernel.at<float>(1,1)= 5.0;
	kernel.at<float>(0,1)= -1.0;
	kernel.at<float>(2,1)= -1.0;
	kernel.at<float>(1,0)= -1.0;
	kernel.at<float>(1,2)= -1.0;

	//filter the image
	filter2D(image,result,image.depth(),kernel);

}