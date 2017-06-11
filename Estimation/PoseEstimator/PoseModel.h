#ifndef POSE_MODEL_H
#define POSE_MODEL_H

#include <string>
#include <iostream>

#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "Utils.h"
#include "Config.h"

using namespace std;
//using namespace cv;

class ShapeModel {
public:
	int								m_numEigenVals;
	int								m_numFacePoints;
	CvMat*							m_eigenVals;
	CvMat*							m_eigenVects;
	CvMat*							m_meanShape;

	// preprocessed
	CvMat*							m_I_PPt; // I - PP'
	CvMat*							m_2alphaI_PPtSq; // 2*alpha*(I-PPt)'(I-PPt)

	CvMat*							m_eigenNr; // P_nr = P/sqrt(Lambda) -> eigen vect divided by eigenval
	CvMat*							m_2H;
	CvMat*							m_F;
	
	CvMat*							m_2alphaWtB;
	CvMat*							m_BBase;

	ShapeModel();
};

class PatchModel {
public:
	int								m_numPatches;
	CvMat*							m_weights[100];
	float*							m_rhos;
	int								m_patchSize[2];

	PatchModel();
};

class SvmWorkspace {
public:
	float*						m_pA;
	float*						m_pRL;
	float*						m_pXcoord;
	float*						m_pYcoord;
	float*						m_pNr;

	SvmWorkspace();

};

class PoseModel {
public:
	int							m_numFacePoints;
	ShapeModel*					m_shapeModel;
	PatchModel*					m_patchModel;
	SvmWorkspace*				m_pSvmWs;

public:
	PoseModel();

	int loadModel(string modelFile);
};


#endif