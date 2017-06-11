#ifndef POSE_EST_H
#define POSE_EST_H

#include <string>
#include "Config.h"
#include "PoseModel.h"
#include "FaceDetector.h"
#include "opencv/cv.h"
#include "DrawFace.h"
#include "DrawFace_IMM.h"
#include "Array.hh"

using namespace std;
//using namespace cv;

class ThreadData{
public:
	void*			thisPtr;
	int				fPoint;
	int				lPoint;
	void*			coeffs;

	// thread memory
	float*			m_pA;
	float*			m_pRL;
	float*			m_pNr;
	CvMat*			m_Response;

	ThreadData() {
		m_pA = NULL;
		m_pRL = NULL;
		m_pNr = NULL;
		m_Response = NULL;

		thisPtr = NULL;
		coeffs = NULL;
	}

	~ThreadData() {
		if( m_pA != NULL )
			free(m_pA);
		if( m_pRL != NULL )
			free(m_pRL);
		if( m_pNr != NULL )
			free(m_pNr);
		if( m_Response != NULL )
			cvReleaseMat(&m_Response);
	}
};

class PoseEstimator {
public:
	string					m_strModelFileName;
	PoseModel*				m_poseModel;
	FaceDetector*			m_faceDetector;

	// current test image
	cv::Mat						m_testMat;
	cv::Mat						m_debugImg;

	// initial guess
	CvMat*					m_initialXY;
	// aligned xy
	CvMat*					m_alignedXY;
	// initial transform mat; [scale scale centreX centerY]
	float					m_transform[4];

	Matrix<double>			m_G; // this matrix is big, so kept it member var
	Matrix<double>			m_CI; // this matrix is big, so kept it member var
	Matrix<double>			m_CE; // this matrix is big, so kept it member var

	ThreadData*				m_threadDataList[NUM_THREADS];

public:
	PoseEstimator();

	// estimates pose
	void estimatePose(string testImgName);
	// extra api for android
	FacePoints estimatePose(cv::Mat testImage, cv::Rect& faceRect, cv::Rect& eyeRect);
	// loads model
	void loadModel();
	// make first initial guess
	void makeInitialGuess(cv::Rect& faceRect, cv::Rect& eyeRect);
	// search for pose
	void searchPose();
	void searchSVM(float*);
	void searchSVMThreaded(float*);
	//void searchThreaded(int first_pt, int last_pt, float* quadCoeffs);
	void searchThreaded(ThreadData* tData);
	void jointOptimize(float* quadCoeffs);

	void svmMatch(CvMat& matNr, CvMat* weights, float rho, CvMat *Response);

	void threadInit();
	static void ThreadStaticEntryPoint(void* arg);
};

#endif