#include <string>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

double percentageIncrease(double originalValue, double NewValue);
double percentageDecrease(double originalValue, double NewValue);
bool deleteFile(string fileName);
bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean=0.0, double StdDev=10.0);