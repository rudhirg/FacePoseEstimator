#include "stdafx.h"
#include "PoseModel.h"
#include "tinyxml.h"

PoseModel::PoseModel() {
	m_numFacePoints = 0;
	m_shapeModel = new ShapeModel();
	m_patchModel = new PatchModel();
	m_pSvmWs = new SvmWorkspace();
}

int PoseModel::loadModel(string modelFile) {
	TiXmlDocument doc(modelFile.c_str());

	bool loadok = doc.LoadFile();
	if(!loadok)
	{
		return -1;
	}

	TiXmlHandle docHandle( &doc );
	
	TiXmlElement *Element = docHandle.FirstChild("root").FirstChild("shapeModel").FirstChild("numEigenVals").ToElement();
	const char *str = Element->GetText();
	this->m_shapeModel->m_numEigenVals = atoi(str);

	Element = docHandle.FirstChild("root").FirstChild("shapeModel").FirstChild("numPts").ToElement();
	str = Element->GetText();
	this->m_shapeModel->m_numFacePoints = atoi(str);

	// ShapeModel.MeanShape:
	Element = docHandle.FirstChild("root").FirstChild("shapeModel").FirstChild("meanShape").ToElement();
	str = Element->GetText();
	this->m_shapeModel->m_meanShape = cvCreateMat(1, this->m_shapeModel->m_numFacePoints*2, CV_32FC1);
	int NumRead = Utils::ReadVecFromString(str, this->m_shapeModel->m_meanShape->data.fl, this->m_shapeModel->m_numFacePoints*2);
	if(NumRead != this->m_shapeModel->m_numFacePoints*2)
	{
		return -1;
	}

	// ShapeModel.Evalues:
	Element = docHandle.FirstChild("root").FirstChild("shapeModel").FirstChild("eigenVals").ToElement();
	str = Element->GetText();
	this->m_shapeModel->m_eigenVals = cvCreateMat(1, this->m_shapeModel->m_numEigenVals, CV_32FC1);
	NumRead = Utils::ReadVecFromString(str, this->m_shapeModel->m_eigenVals->data.fl, this->m_shapeModel->m_numEigenVals);
	if(NumRead != this->m_shapeModel->m_numEigenVals)
	{
		return -1;
	}

	// ShapeModel.Evectors:
	Element = docHandle.FirstChild("root").FirstChild("shapeModel").FirstChild("eigenVects").ToElement();
	str = Element->GetText();
	this->m_shapeModel->m_eigenVects = cvCreateMat(this->m_shapeModel->m_numFacePoints*2, this->m_shapeModel->m_numEigenVals, CV_32FC1);
	NumRead = Utils::ReadMatFromString(str, this->m_shapeModel->m_eigenVects);

	
	// PatchModel.NumPatches:
	Element = docHandle.FirstChild("root").FirstChild("patchModel").FirstChild("NumPatches").ToElement();
	str = Element->GetText();
	this->m_patchModel->m_numPatches = atoi(str);

	// PatchModel.PatchSize:
	Element = docHandle.FirstChild("root").FirstChild("patchModel").FirstChild("PatchSize").ToElement();
	str = Element->GetText();
	this->m_patchModel->m_patchSize[0] = atoi(str);

	while(*str<='9' && *str>='0')
		str++;
	if(*str == ' ')
		str++;

	if(*str == 0)
		return -1;
	
	this->m_patchModel->m_patchSize[1] = atoi(str);

	// PatchModel.weights:
	Element = docHandle.FirstChild("root").FirstChild("patchModel").FirstChild("Weights").ToElement();
	str = Element->GetText();

	CvMat *tempW = cvCreateMat(this->m_patchModel->m_patchSize[1]*this->m_patchModel->m_patchSize[0], 
															this->m_patchModel->m_numPatches, CV_32FC1);
	CvMat *tempWt = cvCreateMat(this->m_patchModel->m_numPatches, 
						this->m_patchModel->m_patchSize[1]*this->m_patchModel->m_patchSize[0], CV_32FC1); 
	Utils::ReadMatFromString(str, tempW);

	cvT(tempW, tempWt);

	float *pw = tempWt->data.fl;
	CvMat *ptempxx = cvCreateMat(this->m_patchModel->m_patchSize[1], this->m_patchModel->m_patchSize[0], CV_32FC1);
	CvMat *ptempxxn = cvCreateMat(this->m_patchModel->m_patchSize[1], this->m_patchModel->m_patchSize[0], CV_32FC1);

	CvMat wmat = cvMat(this->m_patchModel->m_patchSize[0], this->m_patchModel->m_patchSize[1], CV_32FC1, pw);
	for(int i=0;i<this->m_patchModel->m_numPatches;i++)
	{
		wmat.data.fl = pw;
		
		cvT(&wmat, ptempxx);
		cvNormalize(ptempxx, ptempxxn, 1.0, -1.0, CV_MINMAX);

		pw+= this->m_patchModel->m_patchSize[1]*this->m_patchModel->m_patchSize[0];

		// Resize to half template size:

		//this->m_patchModel.WeightMats[i] = cvCreateMat(this->m_patchModel.PatchSize[1]/2, this->m_patchModel.PatchSize[0]/2, CV_32FC1);
		//cvResize(ptempxxn, this->m_patchModel.WeightMats[i]);

		this->m_patchModel->m_weights[i] = cvCreateMat(this->m_patchModel->m_patchSize[1], this->m_patchModel->m_patchSize[0], CV_32FC1);
		cvCopy(ptempxxn, this->m_patchModel->m_weights[i]);
	}

	cvReleaseMat(&ptempxx);
	cvReleaseMat(&ptempxxn);

	cvReleaseMat(&tempW);
	cvReleaseMat(&tempWt);

	// rhos
	this->m_patchModel->m_rhos = (float*)malloc( this->m_patchModel->m_numPatches*sizeof(float));
	Element = docHandle.FirstChild("root").FirstChild("patchModel").FirstChild("rho").ToElement();
	str = Element->GetText();

	Utils::ReadFloatArrayFromString(str, this->m_patchModel->m_rhos, this->m_patchModel->m_numPatches );


	// Do partial calculations:
	int NumX = this->m_shapeModel->m_numFacePoints*2;

	this->m_shapeModel->m_2alphaI_PPtSq =cvCreateMat(NumX, NumX, CV_32FC1); 

	this->m_shapeModel->m_I_PPt = cvCreateMat(NumX, NumX, CV_32FC1);
	CvMat *pIMat = cvCreateMat(NumX, NumX, CV_32FC1);

	cvSetIdentity(pIMat);
	
	cvGEMM(this->m_shapeModel->m_eigenVects, this->m_shapeModel->m_eigenVects, -1.0, pIMat, 1, this->m_shapeModel->m_I_PPt, CV_GEMM_B_T);
	
	float alpha = (float)OPTM_ERROR_WEIGHT;
	cvGEMM(this->m_shapeModel->m_I_PPt, this->m_shapeModel->m_I_PPt, 2*alpha, 0, 0, this->m_shapeModel->m_2alphaI_PPtSq, CV_GEMM_A_T);

	cvReleaseMat(&pIMat);


	this->m_shapeModel->m_eigenNr = cvCreateMat(this->m_shapeModel->m_eigenVects->rows, this->m_shapeModel->m_eigenVects->cols, CV_32FC1);
	float *pEnrMatDat = this->m_shapeModel->m_eigenNr->data.fl;
	float *pEvecDat = this->m_shapeModel->m_eigenVects->data.fl;
	float *pEvalues = this->m_shapeModel->m_eigenVals->data.fl;

	for(int j=0;j<NumX;j++)
	{
		for(int i=0; i<this->m_shapeModel->m_numEigenVals;i++)
		{
			*pEnrMatDat++ = (*pEvecDat++)/sqrt(pEvalues[i]);
		}
	}

	// Initialize temporary working data. 
	this->m_shapeModel->m_2H = cvCreateMat(NumX, NumX, CV_32FC1);
	cvZero(this->m_shapeModel->m_2H);

	
	this->m_shapeModel->m_F = cvCreateMat(NumX, 1, CV_32FC1);

	this->m_shapeModel->m_2alphaWtB = cvCreateMat(NumX, 1, CV_32FC1);

	this->m_shapeModel->m_BBase = cvCreateMat(this->m_shapeModel->m_numEigenVals, 1, CV_32FC1);


	// Initialize workspace data:
	int szToAlloc = (SEARCH_REG_Y+1+5)*(SEARCH_REG_X+1+5); // 5 is extra buffer 
	this->m_pSvmWs->m_pA = (float*)malloc(szToAlloc*3*sizeof(float));	
	this->m_pSvmWs->m_pRL = (float*)malloc(szToAlloc*sizeof(float));		
	// OPTM - not needed
	//this->m_pSvmWs->m_pXcoord = (float*)malloc(65536*sizeof(float));		
	//this->m_pSvmWs->m_pYcoord = (float*)malloc(65536*sizeof(float));		

	szToAlloc = (SEARCH_REG_X + m_patchModel->m_patchSize[0] + 5)*(SEARCH_REG_Y + m_patchModel->m_patchSize[1] + 5); 
	this->m_pSvmWs->m_pNr = (float*)malloc(szToAlloc*2*sizeof(float));	
	
	return 0;
}

ShapeModel::ShapeModel() {
	this->m_numEigenVals = 0;
	m_numFacePoints = 0;

	m_eigenVals = NULL;
	m_eigenVects = NULL;
	m_meanShape = NULL;


	m_I_PPt = NULL; 
	m_2alphaI_PPtSq = NULL;

	m_eigenNr = NULL;
	m_2H = NULL;
	m_F = NULL;

	m_2alphaWtB = NULL;
	m_BBase = NULL;

}

PatchModel::PatchModel() {
	m_numPatches = 0;
	m_patchSize[0] = 0;
	m_patchSize[1] = 0;
}

SvmWorkspace::SvmWorkspace() {
	m_pA = NULL;
	m_pRL = NULL;
	m_pXcoord = NULL;
	m_pYcoord = NULL;
	m_pNr = NULL;
}