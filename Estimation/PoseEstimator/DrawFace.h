#ifndef DRAW_FACE_H
#define DRAW_FACE_H

#include "opencv\cv.h"
#include "opencv\highgui.h"

//using namespace cv;

void DrawConnected(cv::Mat& image, float *pdat, int step, int *order, int cnt, bool bNamePts);
void DrawFaceShape(cv::Mat& image, CvMat *xy, bool bNamePts=false);

#endif