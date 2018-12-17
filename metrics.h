#ifndef METRICS_H
#define METRICS_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

#include "brisque.h"
#include "libsvm/svm.h"

float getBRISQUE(const Mat& image);
double getMSSIM(const Mat& i1, const Mat& i2);
double getPSNR(const Mat& I1, const Mat& I2);

#endif