#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

#include "brisque.h"
#include "libsvm/svm.h"

float getBRISQUE(string imagename);
double getMSSIM( const Mat& i1, const Mat& i2);
double getPSNR(const Mat& I1, const Mat& I2);