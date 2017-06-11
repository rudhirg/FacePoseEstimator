#include "stdafx.h"

#include "DrawFace.h"


static int orFace[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 23, 22, 21, 0};
static int orEbrowL[] = {24, 23, 22, 21, 26, 25, 24};
static int orEbrowR[] = {18, 19, 20, 15, 16, 17, 18};
static int orEyeL[] = {27, 30, 29, 31, 27, 31, 29, 28, 27};
static int orEyeR[] = {34, 35, 32, 36, 34, 36, 32, 33, 34};
static int orNose[] = {37, 38, 39, 40, 46, 41, 47, 42, 43, 44, 45, 37};
static int orMouth[] = {48, 59, 58, 57, 56, 55, 54, 53, 52, 50, 49, 48, 60, 61, 62, 63, 64, 65, 48};




void DrawFaceShape(cv::Mat& image, CvMat *xy, bool bNamePts)
{
	
	// Draw face:
	float *pdat =xy->data.fl;
	int step = xy->step/sizeof(float);
	
	DrawConnected(image, pdat, step, orFace, sizeof(orFace)/sizeof(int), bNamePts);
	DrawConnected(image, pdat, step, orEbrowL, sizeof(orEbrowL)/sizeof(int), bNamePts);
	DrawConnected(image, pdat, step, orEbrowR, sizeof(orEbrowR)/sizeof(int), bNamePts);
	DrawConnected(image, pdat, step, orEyeL, sizeof(orEyeL)/sizeof(int), bNamePts);
	DrawConnected(image, pdat, step, orEyeR, sizeof(orEyeR)/sizeof(int), bNamePts);
	DrawConnected(image, pdat, step, orNose, sizeof(orNose)/sizeof(int), bNamePts);
	DrawConnected(image, pdat, step, orMouth, sizeof(orMouth)/sizeof(int), bNamePts);
	
}


void DrawConnected(cv::Mat& image, float *pdat, int step, int *order, int cnt, bool bNamePts)
{
	int i;

	double x0, x1, y0, y1;
	CvMat img = image;
	
	for(i=0;i<cnt-1; i++)
	{
		x0 = pdat[order[i]*2];
		y0 = pdat[order[i]*2 + 1];

		x1 = pdat[order[i+1]*2];
		y1 = pdat[order[i+1]*2+ 1];

		cvLine(&img, cvPoint((int)x0, (int)y0), cvPoint((int)x1, (int)y1), CV_RGB(255, 0, 0), 1, 8);
		cvCircle(&img, cvPoint((int)x0, (int)y0), 1, CV_RGB(0, 255, 0), 2, 8, 0);
		cvCircle(&img, cvPoint((int)x1, (int)y1), 1, CV_RGB(0, 255, 0), 2, 8, 0);

		if( bNamePts == true ) {
			std::stringstream ss;
			ss << order[i];
			cv::putText(image, ss.str(), cv::Point(x0, y0-10), cv::FONT_HERSHEY_PLAIN, 1,
				cv::Scalar::all(255), 2, 2);
		}
	}

}

