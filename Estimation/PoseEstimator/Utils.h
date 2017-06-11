#ifndef UTILS_H
#define UTILS_H

#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"

//using namespace cv;


namespace Utils {
	int ReadVecFromString(const char *str, float *f, int count);
	int ReadMatFromString(const char *str, CvMat *Mat);
	int ReadFloatArrayFromString(const char *str, float *arr, int num);
	void IncreaseFaceRectSize( cv::Rect& face, int imgWid, int imgHt );
	void AdjustFaceRectSize( cv::Rect& face, cv::Rect& eye, int imgWid, int imgHt );
	void procrustes(float*, float*, int, float*, float*);
	void CopyImageToMat(IplImage *pImg, CvMat *pMat);
	int alignDataInverse(float *pdat, float *tform, int numPts, float *pout);
};

#endif