#include "stdafx.h"
#include "FaceDetector.h"

FaceDetector::FaceDetector() {

}

void FaceDetector::init() {
	m_cascade.load(FACE_DETECT_MODEL);
	m_eyecascade.load(EYE_DETECT_MODEL);
}

vector<cv::Rect> FaceDetector::detectFace(cv::Mat& img) {
	vector<cv::Rect> faces;
	faces.push_back(cv::Rect(0,0,0,0));

	m_cascade.detectMultiScale( img, faces,
        1.1, 2, 0
        |CV_HAAR_FIND_BIGGEST_OBJECT
        |CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
		cv::Size(30, 30) );

	return faces;
}

vector<cv::Rect> FaceDetector::detectEyes(cv::Mat& img) {
	vector<cv::Rect> eyes;
	eyes.push_back(cv::Rect(0,0,0,0));

	m_eyecascade.detectMultiScale( img, eyes,
        1.1, 2, 0
        |CV_HAAR_FIND_BIGGEST_OBJECT
        |CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
		cv::Size(30, 30) );

	return eyes;
}
