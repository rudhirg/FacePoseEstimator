#include "stdafx.h"
#include <windows.h>
#include "PoseEstimator.h"
#include <stdio.h>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "QuadProg\QuadProg++.hh"
#include <time.h>
#include <ctime>
#include "FacePoints.h"
#include <process.h>

using namespace std;

PoseEstimator::PoseEstimator() {
	m_strModelFileName = MODEL_FILE_PATH;
	m_poseModel = new PoseModel();
	m_faceDetector = new FaceDetector();
	
	m_faceDetector->init();
}

void PoseEstimator::estimatePose(string testImgName) {
	cv::Mat img = cv::imread(testImgName, CV_LOAD_IMAGE_GRAYSCALE);
	m_testMat = img.clone();
	m_debugImg = img.clone();

	int imgWid = m_testMat.cols;
	int imgHt = m_testMat.rows;

	// face detection
	vector<cv::Rect> faces = m_faceDetector->detectFace( m_testMat );
	if( faces.size() == 0 ) {
		cout << "No face detected!!" << endl; 
		return;
	}

	// eye detection
	vector<cv::Rect> eyes = m_faceDetector->detectEyes( m_testMat );
	if( eyes.size() == 0 ) {
		cout << "No eyes detected!!" << endl; 
		eyes.push_back(cv::Rect(0,0,0,0));
		//return;
	}

	// take the first face
	cv::Rect faceRect = faces[0];
	cv::Rect eyeRect = eyes[0];

	/*
	FacePoints fp = estimatePose(img, faceRect, eyeRect);
	vector<cv::Point> le = fp.facePoints[0].points;
	for(int i = 0; i < le.size(); i++)
		cout << le[i].x << "," << le[i].y << endl;
	*/
	// increase the face size
	//Utils::IncreaseFaceRectSize( faceRect, imgWid, imgHt );
	Utils::AdjustFaceRectSize( faceRect, eyeRect, imgWid, imgHt );

	// get the face img
	cv::rectangle( m_debugImg, cvPoint(cvRound(faceRect.x), cvRound(faceRect.y)),
                       cvPoint(cvRound((faceRect.x + faceRect.width-1)), cvRound((faceRect.y + faceRect.height-1))),
					   cv::Scalar( 255, 0, 0 ), 3, 8, 0);
	cv::rectangle( m_debugImg, cvPoint(cvRound(eyeRect.x), cvRound(eyeRect.y)),
                       cvPoint(cvRound((eyeRect.x + eyeRect.width-1)), cvRound((eyeRect.y + eyeRect.height-1))),
					   cv::Scalar( 255, 0, 0 ), 3, 8, 0);

	string wname = "face";
	cv::imshow(wname, m_debugImg);
	cv::waitKey();

	this->makeInitialGuess(faceRect, eyeRect);

	// show all points
	for(int i = 0; i < m_initialXY->rows; i++ ) {
		float x = m_initialXY->data.fl[ i*m_initialXY->cols + 0 ];
		float y = m_initialXY->data.fl[ i*m_initialXY->cols + 1 ];
		cv::circle(m_debugImg, cv::Point(x, y), 2, cv::Scalar(255,0,0), 1, 8, 0);
	}

	//imshow(wname, m_testMat);
	//waitKey();

	searchPose();
}

/*
testImage should be grayscale
faceRect should not be all 0's
eyeRect - if no eyes detected, send cv::Rect(0,0,0,0)
*/
FacePoints PoseEstimator::estimatePose(cv::Mat testImage, cv::Rect& faceRect, cv::Rect& eyeRect) {
	cv::Mat img = testImage;
	m_testMat = img.clone();
	m_debugImg = img.clone();

	int imgWid = m_testMat.cols;
	int imgHt = m_testMat.rows;

	// increase the face size
	//Utils::IncreaseFaceRectSize( faceRect, imgWid, imgHt );
	Utils::AdjustFaceRectSize( faceRect, eyeRect, imgWid, imgHt );

	// get the face img
	cv::rectangle( m_debugImg, cvPoint(cvRound(faceRect.x), cvRound(faceRect.y)),
                       cvPoint(cvRound((faceRect.x + faceRect.width-1)), cvRound((faceRect.y + faceRect.height-1))),
					   cv::Scalar( 255, 0, 0 ), 3, 8, 0);
	cv::rectangle( m_debugImg, cvPoint(cvRound(eyeRect.x), cvRound(eyeRect.y)),
                       cvPoint(cvRound((eyeRect.x + eyeRect.width-1)), cvRound((eyeRect.y + eyeRect.height-1))),
					   cv::Scalar( 255, 0, 0 ), 3, 8, 0);

	string wname = "face";
	cv::imshow(wname, m_debugImg);
	cv::waitKey();

	this->makeInitialGuess(faceRect, eyeRect);

	// show all points
	for(int i = 0; i < m_initialXY->rows; i++ ) {
		float x = m_initialXY->data.fl[ i*m_initialXY->cols + 0 ];
		float y = m_initialXY->data.fl[ i*m_initialXY->cols + 1 ];
		cv::circle(m_debugImg, cv::Point(x, y), 2, cv::Scalar(255,0,0), 1, 8, 0);
	}

	//imshow(wname, m_testMat);
	//waitKey();

	searchPose();

	// get all the face points in the data structure
	FacePoints fp;
	for(int i = 0; i < m_initialXY->rows; i++ ) {
		float x = m_initialXY->data.fl[ i*m_initialXY->cols + 0 ];
		float y = m_initialXY->data.fl[ i*m_initialXY->cols + 1 ];

		cv::Point pt = cv::Point(x, y);
		FIDUCIALS fd = getFiducialPointFromIndex( i );
		fp.facePoints[ fd ].points.push_back( pt );
	}

	return fp;
}

void PoseEstimator::searchPose() {
	int nIter = NUM_ITERATONS;
	ShapeModel* shapeModel = m_poseModel->m_shapeModel;
	PatchModel* patchModel = m_poseModel->m_patchModel;
	int numPatches = patchModel->m_numPatches;

	time_t startTime, endTime;

	time(&startTime);
	const clock_t begin_time = clock();

	for(int iter = 0; iter < nIter; iter++) {
		float coeffs[8*100] = {0.0};
		searchSVM( coeffs );
		//searchSVMThreaded( coeffs );

		this->jointOptimize( coeffs );

		//m_debugImg = this->m_testMat.clone();

		//// draw face
		//DrawFaceShape_IMM(m_debugImg, this->m_initialXY, false );
		//imshow("xyz", m_debugImg);
		//waitKey(2);
	}

	time(&endTime);
	cout << "Took time: " << difftime(endTime, startTime) << endl;
	cout << "Total Time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

	DrawFaceShape_IMM(m_debugImg, this->m_initialXY, false );
	cv::imshow("xyz", m_debugImg);
	cv::waitKey();
	// draw face
}

void PoseEstimator::threadInit() {
	cout << "Thread memory init" << endl;

	for(int i = 0; i < NUM_THREADS; i++) {
		ThreadData *tData = new ThreadData();
		m_threadDataList[i] = tData;

		int szToAlloc = (SEARCH_REG_Y+1+5)*(SEARCH_REG_X+1+5); // 5 is extra buffer 
		tData->m_pA = (float*)malloc(szToAlloc*3*sizeof(float));	
		tData->m_pRL = (float*)malloc(szToAlloc*sizeof(float));		

		szToAlloc = (SEARCH_REG_X + m_poseModel->m_patchModel->m_patchSize[0] + 5)*\
								(SEARCH_REG_Y + m_poseModel->m_patchModel->m_patchSize[1] + 5); 
		tData->m_pNr = (float*)malloc(szToAlloc*2*sizeof(float));

		tData->m_Response = cvCreateMat(SEARCH_REG_Y + 1, SEARCH_REG_X + 1, CV_32FC1);
	}
}

void PoseEstimator::searchSVMThreaded(float* quadCoeffs) {
	ShapeModel* shapeModel = m_poseModel->m_shapeModel;
	PatchModel* patchModel = m_poseModel->m_patchModel;
	SvmWorkspace* svmW = m_poseModel->m_pSvmWs;
	CvMat* pMeanShape = shapeModel->m_meanShape;
	int numPatches = patchModel->m_numPatches;

	// do procrustes analysis
	// align initial points to mean shape points
	Utils::procrustes(this->m_initialXY->data.fl, pMeanShape->data.fl, shapeModel->m_numFacePoints, 
												this->m_alignedXY->data.fl, this->m_transform );

	// create threads
	int numPoints = shapeModel->m_numFacePoints;
	int lPoint = -1;
	int sz = (int)ceil((double)numPoints/NUM_THREADS);

	HANDLE threads[NUM_THREADS] = {0};
	for(int i = 0; i < NUM_THREADS; i++) {
		int fPoint = lPoint+1;
		lPoint = min(fPoint + sz-1, numPoints-1);
		//ThreadData *tData = new ThreadData();
		ThreadData *tData = m_threadDataList[i];
		tData->fPoint = fPoint;
		tData->lPoint = lPoint;
		tData->coeffs = quadCoeffs;
		tData->thisPtr = this;

		threads[i] = (HANDLE)_beginthread( &PoseEstimator::ThreadStaticEntryPoint, 0, (void*)tData );
	}

	for(int i = 0; i < NUM_THREADS; i++) {
		WaitForSingleObject( threads[i], INFINITE );
		cout << "Thread: " << i << " ended" << endl;
	}

}

void PoseEstimator::ThreadStaticEntryPoint(void* arg) {
	ThreadData *tData = (ThreadData*)arg;

	PoseEstimator *pEst = (PoseEstimator*)tData->thisPtr;   // the tricky cast

	
	int fPoint = tData->fPoint;
	int lPoint = tData->lPoint;
	float* quadCoeffs = (float*)tData->coeffs;
	
	cout << "Starting processing from: " << fPoint << ", to: " << lPoint << endl;
	//pEst->searchThreaded(fPoint, lPoint, quadCoeffs);    // now call the true entry-point-function
	pEst->searchThreaded(tData);    // now call the true entry-point-function
}

void PoseEstimator::searchThreaded(/*int first_pt, int last_pt, float* quadcoeffs*/ ThreadData* tData) {
	ShapeModel* shapeModel = m_poseModel->m_shapeModel;
	PatchModel* patchModel = m_poseModel->m_patchModel;
	SvmWorkspace* svmW = m_poseModel->m_pSvmWs;
	CvMat* pMeanShape = shapeModel->m_meanShape;
	int numPatches = patchModel->m_numPatches;

	int sWid = 0, sHt = 0;
	sWid = SEARCH_REG_X + patchModel->m_patchSize[0];
	sHt = SEARCH_REG_Y + patchModel->m_patchSize[1];

	// inverse of transform matrix
	float tm[9], invtm[9];
	tm[0] = m_transform[0]; tm[1] = m_transform[1]; tm[2] = m_transform[2]; 
	tm[3] = -m_transform[1]; tm[4] = m_transform[0]; tm[5] = m_transform[3];
	tm[6] = 0; tm[7] = 0; tm[8] = 1;

	CvMat MatTm = cvMat(3, 3, CV_32FC1, tm);
	CvMat invMatTm = cvMat(3, 3, CV_32FC1, invtm);
	cvInvert(&MatTm, &invMatTm);

	float *pxy = m_alignedXY->data.fl;

	// allocate memory
	// OPTM POSSIBLE
	/*
	int szToAlloc = (SEARCH_REG_Y+1+5)*(SEARCH_REG_X+1+5); // 5 is extra buffer 
	float* pA = (float*)malloc(szToAlloc*3*sizeof(float));	
	float* pRL = (float*)malloc(szToAlloc*sizeof(float));		

	szToAlloc = (SEARCH_REG_X + patchModel->m_patchSize[0] + 5)*(SEARCH_REG_Y + patchModel->m_patchSize[1] + 5); 
	float* pNr = (float*)malloc(szToAlloc*2*sizeof(float));
	*/
	
	int first_pt = tData->fPoint;
	int last_pt = tData->lPoint;
	float* quadCoeffs = (float*)tData->coeffs;

	float* pA = tData->m_pA;
	float* pRL = tData->m_pRL;
	float* pNr = tData->m_pNr;
	CvMat* Response = tData->m_Response;

	// crop patch around the face point
	for(int i = first_pt; i <= last_pt; i++) {
		const clock_t begin_point_time = clock();

		// calculate transform matrix
		float m[6];
		CvMat M = cvMat( 2, 3, CV_32F, m );
		//CvMat *Response = cvCreateMat(SEARCH_REG_Y + 1, SEARCH_REG_X + 1, CV_32FC1);// OPTM POSSIBLE
		CvMat *weights = patchModel->m_weights[i];
		float rho = patchModel->m_rhos[i];
		
		float x0c = *(pxy + i*2), y0c = *(pxy + i*2 + 1);

		m[0] = invtm[0];
		m[1] = invtm[1];
		m[3] = -m[1];
		m[4] = m[0];
		// defines the center point of the src mat
		m[2] = invtm[2] + invtm[0]*x0c + invtm[1]*y0c;
		m[5] = invtm[5] - invtm[1]*x0c + invtm[0]*y0c;

		CvMat img = this->m_testMat;
		// change to CV_32FC1
		// THREAD_TODO: m_pNr common
		CvMat pCrop = cvMat(sHt, sWid, CV_32FC1, pNr);
		const clock_t begin_quad_time = clock();
		cvGetQuadrangleSubPix(&img, &pCrop, &M);
		//cout << "Time for quad point: " << float( clock () - begin_quad_time ) << endl;
		CvMat Matnr = cvMat(pCrop.height, pCrop.width, CV_32FC1, pNr);

		/*
		cv::Mat croppedImg = cv::Mat(&pCrop);

		IplImage *pCrop = cvCreateImage(cvSize(sWid, sHt), 8, 1);
		cvGetQuadrangleSubPix(&img, pCrop, &M);

		CvMat Matnr = cvMat(pCrop->height, pCrop->width, CV_32FC1, this->m_poseModel->m_pSvmWs->m_pNr);
		Utils::CopyImageToMat(pCrop, &Matnr);
		Mat croppedImg = Mat(pCrop);*/

		// get the response image
		// TODO use the svm here
		const clock_t begin_templ_time = clock();
		cvMatchTemplate(&Matnr, weights, Response, CV_TM_CCORR_NORMED);
		//cout << "Time for match temp: " << float( clock () - begin_templ_time ) << endl;
		//svmMatch(Matnr, weights, rho, Response);

		const clock_t begin_rest_time = clock();
		cvNormalize(Response, Response, 1.0, 0.0, CV_MINMAX);

		// fit quadratic function over the response

		// fid the sum of response
		//CvScalar sumResp = cvSum(Response);

		CvMat MatXcoord = cvMat(Response->rows, Response->cols, CV_32FC1, svmW->m_pXcoord);
		CvMat MatYcoord = cvMat(Response->rows, Response->cols, CV_32FC1, svmW->m_pYcoord);
		
		float *pxd = svmW->m_pXcoord;
		float *pyd = svmW->m_pYcoord;
		int stepxy = MatXcoord.step/sizeof(float);
		int rr= Response->rows, rc = Response->cols;
		
		// fill the mat with coord values
		// we create mat x as [1 2 3 ..; 1 2 3 ..; .... ]
		// mat y as [ 1 1 1 ..; 2 2 2 ..; 3 3 3..; .... ]
		// commented as we do not have to create it, we can use (i%wr) and floor(i/wr) as x, y
		// as we have used down in code
		// OPTM 1.0
		/*for (int j=0; j<rr;j++)
		{
			for(int i=0;i<rc;i++)
			{
				*pxd++ = (float)i;
				*pyd++ = (float)j;
			}
		}*/

		// Use max response position as center.
		double maxv, respAtMean;
		CvPoint maxLoc;

		cvMinMaxLoc(Response, 0, &maxv, 0, &maxLoc, 0);
		respAtMean = Response->data.fl[ (rr-1)/2*rc + (rc-1)/2 ];
		float centerx = (float)maxLoc.x;
		float centery = (float)maxLoc.y;

		// [debug] write the response image
		/*cv::Mat respImg;
		cv::Mat respConvImg;
		respImg = cv::Mat(Response);
		respImg.convertTo(respConvImg, CV_8UC1, 255, 0);
		circle(respConvImg, Point(centerx, centery), 2, Scalar(0,0, 0), 3, 8, 0);
		std::stringstream sstm;
		sstm << "resp" << i << ".jpg";
		imwrite(sstm.str().c_str(), respConvImg);*/


		/*fit: (x-x0)^2*a + (y-y0)^2*b + c = 0;
		given x = [x1... xn], y = [y1... yn]
		subject to:
		-inf < a < 0
		-inf < b < 0
		-inf < c < inf

		this is least squares fit with conditions, solve it with quadratic programming:
		let: w = [a b c]', 
		   H = [(x1-x0)^2 (y1-y0)^2 1
				(x2-x0)^2 (y2-y0)^2 1
				...
				(xn-x0)^2 (yn-y0)^2 1]
		   r = [r11 r12 ... rnn]';

		problem formulated as:
		minimize(norm2(H*w - r))
		subject to: 
			a < 0
			b < 0

		where: norm2(H*w - r) = (H*w - r)'*(H*w - r)
							  = w'*H'*H*w - 2*r'H*w + r^2  */

		// make H matrix
		int wr, hr;
		wr = Response->rows;
		hr = Response->cols;

		// THREAD_TODO: m_pA common
		CvMat MatH = cvMat(wr*hr, 3, CV_32FC1, pA);
		// THREAD_TODO: m_pRL common
		CvMat MatrL = cvMat(wr*hr, 1, CV_32FC1, pRL);
		
		float *pHd = pA;
		float *prLd = pRL;
		float *prd = Response->data.fl;
		
		pxd = svmW->m_pXcoord;
		pyd = svmW->m_pYcoord;

		float x2t, y2t;
		int wrhr = wr*hr;
		float x2tc, y2tc;
		float hr2 = 1.0f/hr/hr, wr2 = 1.0f/wr/wr;

		for(int j=0;j<wrhr; j++)
		{
			x2t = (j % wr);//*pxd++; // using optimization OPTM 1.0 as above
			y2t = (int)(j / wr);//*pyd++; // using optimization OPTM 1.0 as above
			
			x2tc = x2t-centerx;		// x-x0
			y2tc = y2t-centery;		// y-y0
			x2t = x2tc*x2tc*hr2;	// (x-x0)^2
			y2t = y2tc*y2tc*wr2;	// (y-y0)^2
			
			*pHd++ = x2t;
			*pHd++ = y2t;
			*pHd++ = 1;

			*prLd++ = *prd++;
		}
		
		// Prepare G, g0, CI, ci0;
		float Dat2HtH[9];
		CvMat Mat2HtH = cvMat(3, 3, CV_32FC1, Dat2HtH);
		cvGEMM(&MatH, &MatH, 2.0, 0, 0, &Mat2HtH, CV_GEMM_A_T);

		float Dat_2Ht_rl[3];
		CvMat Mat_2Ht_rL = cvMat(3, 1, CV_32FC1, Dat_2Ht_rl);
		cvGEMM(&MatH, &MatrL, -2.0, 0, 0, &Mat_2Ht_rL, CV_GEMM_A_T);

		// Quadratic programming:
		// G = 2*Ht*H

		Matrix<double> G, CE, CI;
		Vector<double> g0, ce0, ci0, coeffs;
		
		G.resize(3, 3);
		float *ph2hd = Dat2HtH;
		G[0][0] = ph2hd[0];
		G[0][1] = ph2hd[1];
		G[0][2] = ph2hd[2];
		G[1][0] = ph2hd[3];
		G[1][1] = ph2hd[4];
		G[1][2] = ph2hd[5];
		G[2][0] = ph2hd[6];
		G[2][1] = ph2hd[7];
		G[2][2] = ph2hd[8];

		// g0 = 2*Ht*rl
		g0.resize(3);
		float *p2htrld= Dat_2Ht_rl;
		g0[0] = p2htrld[0];
		g0[1] = p2htrld[1];
		g0[2] = p2htrld[2];

		CE.resize(0, 3, 0);
		//ce0.resize(0, 3);

		// CI = [-1 0 0; 0 -1 0; 0 0 -1]; ci0 = [0; 0; +inf]; constraints a<0, b<0
		CI.resize(0, 3, 3);
		CI[0][0]= -1;
		CI[1][1] = -1;
		CI[2][2] = -1;

		ci0.resize(3);
		ci0[0] = 0;
		ci0[1] = 0;
		ci0[2] = 1e8;

		// coeffs = final a b c
		coeffs.resize(0, 3);
		solve_quadprog(G, g0, CE, ce0, CI, ci0, coeffs);

		// Sanity check on results...
		double a=coeffs[0]/wr/wr, b=coeffs[1]/hr/hr, c=coeffs[2];
		if(a > 1e-5 || b > 1e-5)
		{
			printf("Warning: convex fitting result incorrect: %4.6f, %4.6f\n", a, b);
			if(a>0.0) a= -a;
			if(b>0.0) b= -b;

		}

		quadCoeffs[0+8*i] = (float)b;
		quadCoeffs[1+8*i] = 0;
		quadCoeffs[2+8*i] = -2*(float)b*centery;
		quadCoeffs[3+8*i] = (float)a;
		quadCoeffs[4+8*i] = -2*(float)a*centerx;
		quadCoeffs[5+8*i] = (float)a*centerx*centerx + (float)b*centery*centery + (float)c;
		quadCoeffs[6+8*i] = centerx;
		quadCoeffs[7+8*i] = centery;

		//cvReleaseMat(&Response);

		// just to visualize
		float w0 = (float)(SEARCH_REG_X + 1)/2;
		float x_pad = patchModel->m_patchSize[0]/2.0;
		float y_pad = patchModel->m_patchSize[1]/2.0;
		/*croppedImg.convertTo(croppedImg, CV_8UC1);
		circle(croppedImg, Point(centerx + x_pad, centery + y_pad), 2, Scalar(255,255, 0), 3, 8, 0);
		std::stringstream ss;
		ss << i;
		putText(croppedImg, ss.str(), Point(1, hr-2), FONT_HERSHEY_SCRIPT_SIMPLEX, 1,
										Scalar::all(255), 2, 2);
		imshow("xyz", croppedImg);
		waitKey();*/
		//cout << "max resp ( " << i << " ): " << maxv  << ", mean: " << respAtMean << endl;

	}

}

void PoseEstimator::searchSVM(float* quadCoeffs) {
	ShapeModel* shapeModel = m_poseModel->m_shapeModel;
	PatchModel* patchModel = m_poseModel->m_patchModel;
	SvmWorkspace* svmW = m_poseModel->m_pSvmWs;
	CvMat* pMeanShape = shapeModel->m_meanShape;
	int numPatches = patchModel->m_numPatches;

	int sWid = 0, sHt = 0;
	sWid = SEARCH_REG_X + patchModel->m_patchSize[0];
	sHt = SEARCH_REG_Y + patchModel->m_patchSize[1];

	const clock_t begin_time = clock();

	// do procrustes analysis
	// align initial points to mean shape points
	Utils::procrustes(this->m_initialXY->data.fl, pMeanShape->data.fl, shapeModel->m_numFacePoints, 
												this->m_alignedXY->data.fl, this->m_transform );

	// inverse of transform matrix
	float tm[9], invtm[9];
	tm[0] = m_transform[0]; tm[1] = m_transform[1]; tm[2] = m_transform[2]; 
	tm[3] = -m_transform[1]; tm[4] = m_transform[0]; tm[5] = m_transform[3];
	tm[6] = 0; tm[7] = 0; tm[8] = 1;

	CvMat MatTm = cvMat(3, 3, CV_32FC1, tm);
	CvMat invMatTm = cvMat(3, 3, CV_32FC1, invtm);
	cvInvert(&MatTm, &invMatTm);

	float *pxy = m_alignedXY->data.fl;

	// use for visualization only
	float visxy[200];
	float *alignedxy = this->m_alignedXY->data.fl;

	// allocate memory
	int szToAlloc = (SEARCH_REG_Y+1+5)*(SEARCH_REG_X+1+5); // 5 is extra buffer 
	float* pA = (float*)malloc(szToAlloc*3*sizeof(float));	
	float* pRL = (float*)malloc(szToAlloc*sizeof(float));		

	szToAlloc = (SEARCH_REG_X + patchModel->m_patchSize[0] + 5)*(SEARCH_REG_Y + patchModel->m_patchSize[1] + 5); 
	float* pNr = (float*)malloc(szToAlloc*2*sizeof(float));

	// crop patch around the face point
	for(int i = 0; i < shapeModel->m_numFacePoints; i++) {
		const clock_t begin_point_time = clock();

		// calculate transform matrix
		float m[6];
		CvMat M = cvMat( 2, 3, CV_32F, m );
		CvMat *Response = cvCreateMat(SEARCH_REG_Y + 1, SEARCH_REG_X + 1, CV_32FC1);
		CvMat *weights = patchModel->m_weights[i];
		float rho = patchModel->m_rhos[i];
		
		float x0c = *(pxy + i*2), y0c = *(pxy + i*2 + 1);

		m[0] = invtm[0];
		m[1] = invtm[1];
		m[3] = -m[1];
		m[4] = m[0];
		// defines the center point of the src mat
		m[2] = invtm[2] + invtm[0]*x0c + invtm[1]*y0c;
		m[5] = invtm[5] - invtm[1]*x0c + invtm[0]*y0c;

		CvMat img = this->m_testMat;
		// change to CV_32FC1
		// THREAD_TODO: m_pNr common
		CvMat pCrop = cvMat(sHt, sWid, CV_32FC1, pNr);
		const clock_t begin_quad_time = clock();
		cvGetQuadrangleSubPix(&img, &pCrop, &M);
		//cout << "Time for quad point: " << float( clock () - begin_quad_time ) << endl;
		// THREAD_TODO: m_pNr common
		CvMat Matnr = cvMat(pCrop.height, pCrop.width, CV_32FC1, pNr);

		/*
		cv::Mat croppedImg = cv::Mat(&pCrop);

		IplImage *pCrop = cvCreateImage(cvSize(sWid, sHt), 8, 1);
		cvGetQuadrangleSubPix(&img, pCrop, &M);

		CvMat Matnr = cvMat(pCrop->height, pCrop->width, CV_32FC1, this->m_poseModel->m_pSvmWs->m_pNr);
		Utils::CopyImageToMat(pCrop, &Matnr);
		Mat croppedImg = Mat(pCrop);*/

		// get the response image
		// TODO use the svm here
		const clock_t begin_templ_time = clock();
		cvMatchTemplate(&Matnr, weights, Response, CV_TM_CCORR_NORMED);
		//cout << "Time for match temp: " << float( clock () - begin_templ_time ) << endl;
		//svmMatch(Matnr, weights, rho, Response);

		const clock_t begin_rest_time = clock();
		cvNormalize(Response, Response, 1.0, 0.0, CV_MINMAX);

		// fit quadratic function over the response

		// fid the sum of response
		//CvScalar sumResp = cvSum(Response);

		CvMat MatXcoord = cvMat(Response->rows, Response->cols, CV_32FC1, svmW->m_pXcoord);
		CvMat MatYcoord = cvMat(Response->rows, Response->cols, CV_32FC1, svmW->m_pYcoord);
		
		float *pxd = svmW->m_pXcoord;
		float *pyd = svmW->m_pYcoord;
		int stepxy = MatXcoord.step/sizeof(float);
		int rr= Response->rows, rc = Response->cols;
		
		// fill the mat with coord values
		// we create mat x as [1 2 3 ..; 1 2 3 ..; .... ]
		// mat y as [ 1 1 1 ..; 2 2 2 ..; 3 3 3..; .... ]
		// commented as we do not have to create it, we can use (i%wr) and floor(i/wr) as x, y
		// as we have used down in code
		// OPTM 1.0
		/*for (int j=0; j<rr;j++)
		{
			for(int i=0;i<rc;i++)
			{
				*pxd++ = (float)i;
				*pyd++ = (float)j;
			}
		}*/

		// Use max response position as center.
		double maxv, respAtMean;
		CvPoint maxLoc;

		cvMinMaxLoc(Response, 0, &maxv, 0, &maxLoc, 0);
		respAtMean = Response->data.fl[ (rr-1)/2*rc + (rc-1)/2 ];
		float centerx = (float)maxLoc.x;
		float centery = (float)maxLoc.y;

		// [debug] write the response image
		/*cv::Mat respImg;
		cv::Mat respConvImg;
		respImg = cv::Mat(Response);
		respImg.convertTo(respConvImg, CV_8UC1, 255, 0);
		circle(respConvImg, Point(centerx, centery), 2, Scalar(0,0, 0), 3, 8, 0);
		std::stringstream sstm;
		sstm << "resp" << i << ".jpg";
		imwrite(sstm.str().c_str(), respConvImg);*/


		/*fit: (x-x0)^2*a + (y-y0)^2*b + c = 0;
		given x = [x1... xn], y = [y1... yn]
		subject to:
		-inf < a < 0
		-inf < b < 0
		-inf < c < inf

		this is least squares fit with conditions, solve it with quadratic programming:
		let: w = [a b c]', 
		   H = [(x1-x0)^2 (y1-y0)^2 1
				(x2-x0)^2 (y2-y0)^2 1
				...
				(xn-x0)^2 (yn-y0)^2 1]
		   r = [r11 r12 ... rnn]';

		problem formulated as:
		minimize(norm2(H*w - r))
		subject to: 
			a < 0
			b < 0

		where: norm2(H*w - r) = (H*w - r)'*(H*w - r)
							  = w'*H'*H*w - 2*r'H*w + r^2  */

		// make H matrix
		int wr, hr;
		wr = Response->rows;
		hr = Response->cols;

		// THREAD_TODO: m_pA common
		CvMat MatH = cvMat(wr*hr, 3, CV_32FC1, pA);
		// THREAD_TODO: m_pRL common
		CvMat MatrL = cvMat(wr*hr, 1, CV_32FC1, pRL);
		
		float *pHd = pA;
		float *prLd = pRL;
		float *prd = Response->data.fl;
		
		pxd = svmW->m_pXcoord;
		pyd = svmW->m_pYcoord;

		float x2t, y2t;
		int wrhr = wr*hr;
		float x2tc, y2tc;
		float hr2 = 1.0f/hr/hr, wr2 = 1.0f/wr/wr;

		for(int j=0;j<wrhr; j++)
		{
			x2t = (j % wr);//*pxd++; // using optimization OPTM 1.0 as above
			y2t = (int)(j / wr);//*pyd++; // using optimization OPTM 1.0 as above
			
			x2tc = x2t-centerx;		// x-x0
			y2tc = y2t-centery;		// y-y0
			x2t = x2tc*x2tc*hr2;	// (x-x0)^2
			y2t = y2tc*y2tc*wr2;	// (y-y0)^2
			
			*pHd++ = x2t;
			*pHd++ = y2t;
			*pHd++ = 1;

			*prLd++ = *prd++;
		}
		
		// Prepare G, g0, CI, ci0;
		float Dat2HtH[9];
		CvMat Mat2HtH = cvMat(3, 3, CV_32FC1, Dat2HtH);
		cvGEMM(&MatH, &MatH, 2.0, 0, 0, &Mat2HtH, CV_GEMM_A_T);

		float Dat_2Ht_rl[3];
		CvMat Mat_2Ht_rL = cvMat(3, 1, CV_32FC1, Dat_2Ht_rl);
		cvGEMM(&MatH, &MatrL, -2.0, 0, 0, &Mat_2Ht_rL, CV_GEMM_A_T);

		// Quadratic programming:
		// G = 2*Ht*H

		Matrix<double> G, CE, CI;
		Vector<double> g0, ce0, ci0, coeffs;
		
		G.resize(3, 3);
		float *ph2hd = Dat2HtH;
		G[0][0] = ph2hd[0];
		G[0][1] = ph2hd[1];
		G[0][2] = ph2hd[2];
		G[1][0] = ph2hd[3];
		G[1][1] = ph2hd[4];
		G[1][2] = ph2hd[5];
		G[2][0] = ph2hd[6];
		G[2][1] = ph2hd[7];
		G[2][2] = ph2hd[8];

		// g0 = 2*Ht*rl
		g0.resize(3);
		float *p2htrld= Dat_2Ht_rl;
		g0[0] = p2htrld[0];
		g0[1] = p2htrld[1];
		g0[2] = p2htrld[2];

		CE.resize(0, 3, 0);
		//ce0.resize(0, 3);

		// CI = [-1 0 0; 0 -1 0; 0 0 -1]; ci0 = [0; 0; +inf]; constraints a<0, b<0
		CI.resize(0, 3, 3);
		CI[0][0]= -1;
		CI[1][1] = -1;
		CI[2][2] = -1;

		ci0.resize(3);
		ci0[0] = 0;
		ci0[1] = 0;
		ci0[2] = 1e8;

		// coeffs = final a b c
		coeffs.resize(0, 3);
		solve_quadprog(G, g0, CE, ce0, CI, ci0, coeffs);

		// Sanity check on results...
		double a=coeffs[0]/wr/wr, b=coeffs[1]/hr/hr, c=coeffs[2];
		if(a > 1e-5 || b > 1e-5)
		{
			printf("Warning: convex fitting result incorrect: %4.6f, %4.6f\n", a, b);
			if(a>0.0) a= -a;
			if(b>0.0) b= -b;

		}

		quadCoeffs[0+8*i] = (float)b;
		quadCoeffs[1+8*i] = 0;
		quadCoeffs[2+8*i] = -2*(float)b*centery;
		quadCoeffs[3+8*i] = (float)a;
		quadCoeffs[4+8*i] = -2*(float)a*centerx;
		quadCoeffs[5+8*i] = (float)a*centerx*centerx + (float)b*centery*centery + (float)c;
		quadCoeffs[6+8*i] = centerx;
		quadCoeffs[7+8*i] = centery;

		cvReleaseMat(&Response);

		// just to visualize
		float w0 = (float)(SEARCH_REG_X + 1)/2;
		float x_pad = patchModel->m_patchSize[0]/2.0;
		float y_pad = patchModel->m_patchSize[1]/2.0;
		/*croppedImg.convertTo(croppedImg, CV_8UC1);
		circle(croppedImg, Point(centerx + x_pad, centery + y_pad), 2, Scalar(255,255, 0), 3, 8, 0);
		std::stringstream ss;
		ss << i;
		putText(croppedImg, ss.str(), Point(1, hr-2), FONT_HERSHEY_SCRIPT_SIMPLEX, 1,
										Scalar::all(255), 2, 2);
		imshow("xyz", croppedImg);
		waitKey();*/
		//cout << "max resp ( " << i << " ): " << maxv  << ", mean: " << respAtMean << endl;

		// for visualizing all points on face
		visxy[2*i] = (float) (centerx - w0 + alignedxy[2*i]);
		visxy[2*i + 1] = (float) (centery - w0 + alignedxy[2*i + 1]);

		//cout << "Time for rest point: " << float( clock () - begin_rest_time )  << endl;
		//cout << "Time for one point: " << float( clock () - begin_point_time ) << endl;
	}

	// for debugging only
	// align back to image coordinate.
	//Utils::alignDataInverse(visxy, this->m_transform, shapeModel->m_numFacePoints, this->m_initialXY->data.fl);

	cout << "Time for one round: " << float( clock () - begin_time ) << endl;
}

void PoseEstimator::svmMatch(CvMat& matNr, CvMat* weights, float rho, CvMat *Response) {
	int w = matNr.cols;
	int h = matNr.rows;
	int Ww = weights->cols;
	int Wh = weights->rows;

	float *subImg = (float*)malloc(Ww*Wh*sizeof(float));
	int respI = 0;

	for(int x = 0; (x + Ww) <= w; x++ ) {
		for(int y = 0; (y + Wh) <= h; y++) {
			int cnt = 0;
			float sum = 0.0;
			float maxx = -100000.0;
			float minx = 1000000.0;
			for(int i = 0; i < Ww; i++ ) {
				for(int j = 0; j < Wh; j++) {
					float val = matNr.data.fl[ w*(y+j) + (x+i) ];
					subImg[ cnt ++ ] = val;
					maxx = MAX( val, maxx );
					minx = MIN( val, minx );
				}
			}

			for(int i = 0; i < Ww*Wh; i++ ) {
				subImg[ i ] = (subImg[ i ] - minx)/(maxx - minx);
			}

			// svm, w.x + b
			float resp = 0.0;
			for(int i = 0; i < Ww*Wh; i++) {
				resp += weights->data.fl[ i ]*subImg[ i ];
			}
			resp += rho;

			Response->data.fl[ respI ++ ] = resp;
		}
	}

	// I really do not know why response has to be transposed here but it is needed
	cvT( Response, Response );
}

void PoseEstimator::jointOptimize(float* coeffs) {

	ShapeModel* shapeModel = m_poseModel->m_shapeModel;
	PatchModel* patchModel = m_poseModel->m_patchModel;
	SvmWorkspace* svmW = m_poseModel->m_pSvmWs;
	int numPatches = patchModel->m_numPatches * 2;

	const clock_t begin_time = clock();

	// Prepare
	int i;  

	// Prepare -2H and -F in R:(0.5*x_t*2H*x+F*x)
	// H = [-2a 0; 0 -2b], F = [2ax0 2by0]
	float *p_2Hdat = shapeModel->m_2H->data.fl;
	float *p_Fdat = shapeModel->m_F->data.fl;

	for(i = 0;i < numPatches/2; i++) {
		p_2Hdat[i*2] = -coeffs[3]*2;	// -2a
		p_2Hdat[i*2+1] = -coeffs[1];	// 0
		
		p_2Hdat += numPatches;					
		
		p_2Hdat[i*2] = -coeffs[1];		// 0
		p_2Hdat[i*2+1] = -coeffs[0]*2;	// -2b

		*p_Fdat++ = -coeffs[4];			// 2ax0
		*p_Fdat++ = -coeffs[2];			// 2by0

		p_2Hdat += numPatches;
		coeffs += 8;
	}

	// Prepare norm2((x+basexy) - Evec*(Evec'*(x+basexy))));
	// basexy:
	float newxy[256];
	// assumes SEARCH_REG_X == SEARCH_REG_Y, otherwise use two diff w0
	float w0 = (float)(SEARCH_REG_X + 1)/2;

	CvMat *pMeanShape = shapeModel->m_meanShape;
	float *pMeanxy = pMeanShape->data.fl;
	float *pxy0 = this->m_initialXY->data.fl;
	
	float *alignedxy = this->m_alignedXY->data.fl;
	float *tform = this->m_transform;
	
	float basexy[256];

	for(i = 0; i < numPatches; i++)
	{
		basexy[i] = alignedxy[i] - pMeanxy[i] - w0; // this gets base with respect to mean shape as ref
	}

	CvMat BasexyMat = cvMat(numPatches, 1, CV_32FC1, basexy);


	// norm2((x+basexy)-PP_t'*(x+basexy)) = x'*W*x + 2*basexy'*W*x + basexy'*W*basexy;
	//			where W = (I-PP')'*(I-PP')
	cvGEMM(shapeModel->m_2alphaI_PPtSq, &BasexyMat, 1, 0, 0, shapeModel->m_2alphaWtB, CV_GEMM_A_T);

	// Prepare G (quad term), g0 (linear term):
	// Eq = x'Hx - F'x -2BW(x+basexy)'(x+basexy)
	// G = (H + 2BW), g0 = 2BW - F
	float *psrc1 = shapeModel->m_2H->data.fl;
	float *psrc2 = shapeModel->m_2alphaI_PPtSq->data.fl;
	
	Matrix<double> CE;
	Vector<double> g0, ce0, ci0, coeffs, xout;

	for(i = 0; i < numPatches; i++)
	{
		for(int j = 0; j < numPatches; j++)
		{
			m_G[i][j] = (*psrc1++) + (*psrc2++);
		}
	}

	psrc1 = shapeModel->m_F->data.fl;
	psrc2 = shapeModel->m_2alphaWtB->data.fl;

	g0.resize( numPatches );

	for(i = 0; i < numPatches; i++) {
		g0[i] = (*psrc1++) + (*psrc2++);
	}

	// Prepare constraint matrices:
	//		-x + ub > 0;
	//		 x - lb > 0;
	//		-Pnr'x - Pnr'*basexy + sub > 0;
	//		Pnr'x + Pnr'*basexy + sub > 0; where Pnr = eigen/sqrt(lambda)
	// Calculate Pnr'*basexy
	cvGEMM(shapeModel->m_eigenNr, &BasexyMat, 1, 0, 0, shapeModel->m_BBase, CV_GEMM_A_T);
	
	// CE
	//  ce0
	//ce0.resize( 0, 2*numPatches );

	// Create CI:
	//memset(CI, 0, sizeof(CI));

	float *pBMatDat = shapeModel->m_eigenNr->data.fl;
	
	for(i=0;i<numPatches;i++)
	{
		// Lower and upper bound.
		m_CI[i][i] = -1;
		m_CI[i][numPatches+i] = 1;

		// Shape constraint: -3<bj/sqrt(lambdaj)<3
		for(int k=0;k<shapeModel->m_numEigenVals;k++)
		{
			m_CI[i][numPatches*2 + k] = -(*pBMatDat);
			m_CI[i][numPatches*2 + shapeModel->m_numEigenVals + k] = *pBMatDat++;
		}
	
	}

	// Create ci0:
	ci0.resize( 2*(shapeModel->m_numEigenVals + numPatches) );

	for(i=0;i<numPatches;i++)
	{
		// Lower and upper bound.
		ci0[i] = 2*w0;
		ci0[numPatches + i] = 0;
	}

	float *pBBaseMatDat = shapeModel->m_BBase->data.fl;
	
	for(i=0;i<shapeModel->m_numEigenVals;i++)
	{
		// Shape constraint:
		ci0[2*numPatches + i] = 3 - *pBBaseMatDat;
		ci0[2*numPatches + shapeModel->m_numEigenVals + i] = 3 + *pBBaseMatDat++;
	}
	
	// Do quadprog:
	xout.resize(numPatches);
	double err = solve_quadprog(m_G, g0, m_CE, ce0, m_CI, ci0, xout);
	cout << "function value: " << err << endl;
	
	// Reconstruct x:
	for(i=0;i<numPatches;i++)
	{
		newxy[i] =(float) (xout[i] - w0 + alignedxy[i]);
	}
	
	// align back to image coordinate.
	// 
	Utils::alignDataInverse(newxy, this->m_transform, shapeModel->m_numFacePoints, this->m_initialXY->data.fl);
	cout << "Time for one round in joint optm: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

}

void PoseEstimator::loadModel() {
	cout << "Loading Trained Pose Model.. " << endl;
	if( m_poseModel != NULL ) {
		m_poseModel->loadModel(m_strModelFileName);
		// init some matrixs
		int numPatches = m_poseModel->m_patchModel->m_numPatches;
		int numEigens = m_poseModel->m_shapeModel->m_numEigenVals;
		m_G.resize( 0, 2*numPatches, 2*numPatches );
		m_CE.resize( 0, 2*numPatches, 0 );

		m_CI.resize( 0, 2*numPatches, 
						2*(numEigens + 2*numPatches) );

		this->threadInit();
	}
}

void PoseEstimator::makeInitialGuess(cv::Rect& faceRect, cv::Rect& eyeRect) {
	int width = m_testMat.cols;
	int height = m_testMat.rows;
	int faceWid = faceRect.width;
	int faceHt = faceRect.height;
	int fcenterX = faceRect.x + faceWid/2;
	int fcenterY = faceRect.y + faceHt/2;

	ShapeModel* shapeModel = this->m_poseModel->m_shapeModel;

	CvMat* pMeanShape = shapeModel->m_meanShape;
	int numFacePts = shapeModel->m_numFacePoints;
	
	float* pSrc = NULL;
	float* pDst = NULL;
	int step = 0;
	float maxx = -1000, minx = 1000, maxy = -1000, miny = 1000;

	CvMat* pHomo = cvCreateMat(3, numFacePts, CV_32FC1);
	// has final x,y positions for the mean shape, scaled to fit face
	CvMat* pOutXY = cvCreateMat(3, numFacePts, CV_32FC1);


	// calculate mean x,y for mean shape
	float meanx=0.0, meany=0.0;
	pSrc = pMeanShape->data.fl;

	for(int i = 0; i < numFacePts; i++ ) {
		meanx += *pSrc++;
		meany += *pSrc++;
	}
	meanx = meanx/numFacePts;
	meany = meany/numFacePts;

	// fill the homogenous matrix
	pSrc = pMeanShape->data.fl;
	pDst = pHomo->data.fl;
	step = pHomo->step/sizeof(float);
	for(int i = 0; i < numFacePts; i++) {
		float x = *pSrc - meanx;
		float y = *(pSrc + 1) - meany;
		
		*pDst = x;
		*(pDst + step) = y;
		*(pDst + 2*step) = 1;

		if( x > maxx ) maxx = x;
		if( x < minx ) minx = x;
		if( y > maxy ) maxy = y;
		if( y < miny ) miny = y;

		cout << "y: " << (*(pSrc + step)) << "miny: " << y << endl;

		pSrc += 2;
		pDst ++;
	}

	// calculate scale factor
	float meanWidth = maxx - minx;
	float meanHeight = maxy - miny;

	float scale = (float)faceWid/meanWidth;
	if(scale*meanHeight > faceHt)
		scale = (float)faceHt/meanHeight;

	// scale using eye information
	// get mean extreme eye x pos (pt 27 and pt 32)
	/*float meanLeftEye = pMeanShape->data.fl[ 27*2 ];
	float meanRhtEye = pMeanShape->data.fl[ 32*2 ];
	scale = (float)eyeRect.width/(fabs(meanRhtEye - meanLeftEye));
	scale = scale*0.8;*/

	//TODO rotation matrix --- TODO
	float rotmdat[9] = {1,0,0,0,1,0,1,1,1};
	CvMat rotMat = cvMat(3, 3, CV_32FC1, rotmdat);

	// multiply everything to get final x,y
	cvGEMM(&rotMat, pHomo, scale, 0, 0, pOutXY, 0);

	m_initialXY = cvCreateMat(numFacePts, 2, CV_32FC1);
	m_alignedXY = cvCreateMat(numFacePts, 2, CV_32FC1);

	pSrc = pOutXY->data.fl;
	pDst = m_initialXY->data.fl;
	step = pOutXY->step/sizeof(float);

	for (int i=0; i<numFacePts;i++) {
		*pDst++ = *pSrc + fcenterX;
		*pDst++ = *(pSrc+step) + fcenterY;

		pSrc++;
	}

	// save transform matrix
	m_transform[0] = scale;
	m_transform[1] = scale;
	m_transform[2] = fcenterX;
	m_transform[3] = fcenterY;

	cvReleaseMat(&pHomo);
	cvReleaseMat(&pOutXY);
}

int main() {
	PoseEstimator* pEst = new PoseEstimator();
	pEst->loadModel();
	pEst->estimatePose(TEST_IMAGE);
	return 0;
}