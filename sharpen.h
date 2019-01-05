#ifndef SHARPEN_H
#define SHARPEN_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

Mat unsharpMask(const Mat image, double sigma, double threshold, double amount);

#endif