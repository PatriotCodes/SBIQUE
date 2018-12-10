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
