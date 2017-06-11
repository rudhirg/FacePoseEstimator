#ifndef DRAW_FACE_IMM_H
#define DRAW_FACE_IMM_H

#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "FacePoints.h"

//using namespace cv;

void DrawConnected_IMM(cv::Mat& image, float *pdat, int step, int *order, int cnt, bool bNamePts);
void DrawFaceShape_IMM(cv::Mat& image, CvMat *xy, bool bNamePts=false);
FIDUCIALS getFiducialPointFromIndex( int facePointIndex );
bool getFiducialPointFromIndex( int facePointIndex, int *order, int sz );

#endif