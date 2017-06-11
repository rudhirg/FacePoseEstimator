#ifndef FACE_DETECT_H
#define FACE_DETECT_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

#include "Config.h"

using namespace std;
//using namespace cv;

class FaceDetector {
public:
	cv::CascadeClassifier				m_cascade;
	cv::CascadeClassifier				m_eyecascade;
public:
	FaceDetector();

	void init();
	vector<cv::Rect> detectFace(cv::Mat& img);
	vector<cv::Rect> detectEyes(cv::Mat& img);
};

#endif