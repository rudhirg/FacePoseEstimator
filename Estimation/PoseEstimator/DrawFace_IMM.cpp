#include "stdafx.h"

#include "DrawFace_IMM.h"


static int immFace[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static int immEbrowL[] = {29, 30, 31, 32, 33};
static int immEbrowR[] = {34, 35, 36, 37, 38};
static int immEyeL[] = {13, 14, 15, 16, 17, 18, 19, 20, 13};
static int immEyeR[] = {21, 22, 23, 24, 25, 26, 27, 28, 21};
static int immNose[] = {47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 47};
static int immMouth[] = {39, 40, 41, 42, 43, 44, 45, 46, 39};




void DrawFaceShape_IMM(cv::Mat& image, CvMat *xy, bool bNamePts)
{
	
	// Draw face:
	float *pdat =xy->data.fl;
	int step = xy->step/sizeof(float);
	
	DrawConnected_IMM(image, pdat, step, immFace, sizeof(immFace)/sizeof(int), bNamePts);
	DrawConnected_IMM(image, pdat, step, immEbrowL, sizeof(immEbrowL)/sizeof(int), bNamePts);
	DrawConnected_IMM(image, pdat, step, immEbrowR, sizeof(immEbrowR)/sizeof(int), bNamePts);
	DrawConnected_IMM(image, pdat, step, immEyeL, sizeof(immEyeL)/sizeof(int), bNamePts);
	DrawConnected_IMM(image, pdat, step, immEyeR, sizeof(immEyeR)/sizeof(int), bNamePts);
	DrawConnected_IMM(image, pdat, step, immNose, sizeof(immNose)/sizeof(int), bNamePts);
	DrawConnected_IMM(image, pdat, step, immMouth, sizeof(immMouth)/sizeof(int), bNamePts);
	
}


void DrawConnected_IMM(cv::Mat& image, float *pdat, int step, int *order, int cnt, bool bNamePts)
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


FIDUCIALS getFiducialPointFromIndex( int facePointIndex ) {
	bool ret = false;
	FIDUCIALS fd = FIDUCIALS::NONE;
	if( ret == false ) { 
		ret = getFiducialPointFromIndex(facePointIndex, immEbrowL, sizeof(immEbrowL)/sizeof(int));
		fd = FIDUCIALS::LEFT_EYEBROW;
	}
	if( ret == false ) { 
		ret = getFiducialPointFromIndex(facePointIndex, immEbrowR, sizeof(immEbrowR)/sizeof(int));
		fd = FIDUCIALS::RIGHT_EYEBROW;
	}
	if( ret == false ) { 
		ret = getFiducialPointFromIndex(facePointIndex, immEyeL, sizeof(immEyeL)/sizeof(int));
		fd = FIDUCIALS::LEFT_EYE;
	}
	if( ret == false ) { 
		ret = getFiducialPointFromIndex(facePointIndex, immEyeR, sizeof(immEyeR)/sizeof(int));
		fd = FIDUCIALS::RIGHT_EYE;
	}
	if( ret == false ) { 
		ret = getFiducialPointFromIndex(facePointIndex, immMouth, sizeof(immMouth)/sizeof(int));
		fd = FIDUCIALS::LIPS;
	}
	if( ret == false ) { 
		ret = getFiducialPointFromIndex(facePointIndex, immNose, sizeof(immNose)/sizeof(int));
		fd = FIDUCIALS::NOSE;
	}

	return ( ret == false ? FIDUCIALS::NONE: fd );
}

bool getFiducialPointFromIndex( int facePointIndex, int *order, int sz) {
	for(int i = 0; i < sz ; i++ ) {
		if( order[i] == facePointIndex )
			return true;
	}

	return false;
}


