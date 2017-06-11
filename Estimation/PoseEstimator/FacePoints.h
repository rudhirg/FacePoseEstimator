#ifndef FACE_POINTS_H
#define FACE_POINTS_H

#include <vector>
#include "opencv\cv.h"
#include "opencv\highgui.h"

using namespace std;

enum FIDUCIALS {
	LEFT_EYE,
	RIGHT_EYE,
	NOSE,
	LEFT_EYEBROW,
	RIGHT_EYEBROW,
	LIPS,
	NONE
} ;

struct FPoints{
	std::vector<cv::Point>		points;	
} ;

class FacePoints {
public:
	//std::vector<FPoints>		facePoints;
	std::map<FIDUCIALS, FPoints>	facePoints;

	FacePoints();

};

#endif