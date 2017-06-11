#include "stdafx.h"
#include "Utils.h"

namespace Utils {

int ReadVecFromString(const char *str, float *f, int count)
{
	const char *ptr = str;
	int i;

	for(i=0;i<count;i++)
	{
		f[i] = (float)atoi(ptr);

		while(*ptr!=' ' && *ptr)
			ptr++;

		if(*ptr ==0) 
			return i+1;
		if(*ptr == ' ')
			ptr++;
		
	}

	return -1;
}

int ReadMatFromString(const char *str, CvMat *Mat)
{
	const char *ptr = str;
	int i;
	float *pdat;
	

	pdat = Mat->data.fl;
	
	for(i=0;i<Mat->cols*Mat->rows;i++)
	{
		*pdat++ = (float)atof(ptr);	

		while(*ptr!=' ' && *ptr)
			ptr++;
			
		if(*ptr ==0) 
			return 0;

		if(*ptr == ' ')
			ptr++;
	}

	return 0;
}

int ReadFloatArrayFromString(const char *str, float *arr, int num)
{
	const char *ptr = str;
	int i;
	
	for(i=0;i<num;i++)
	{
		*arr++ = (float)atof(ptr);	

		while(*ptr!=' ' && *ptr)
			ptr++;
			
		if(*ptr ==0) 
			return 0;

		if(*ptr == ' ')
			ptr++;
	}

	return 0;
}


void IncreaseFaceRectSize( cv::Rect& face, int imgWid, int imgHt ) {
	int wid = face.width;
	int ht = face.height;

	int x = face.x;
	int y = face.y;

	x = MAX( x - ceil(wid/10.0), 1 );
	y = MAX( y - ceil(ht/6.0), 1 );

	int xm = face.x+wid;
	int ym = face.y+ht;

	xm = MIN( xm + floor(wid/10.0), imgWid - 1 );
	ym = MIN( ym + floor(ht/6.0), imgHt - 1 );

	face.width = xm - x;
	face.height = ym - y;
	face.x = x;
	face.y = y;
}

void AdjustFaceRectSize( cv::Rect& face, cv::Rect& eyes, int imgWid, int imgHt ) {
	int wid = face.width;
	int ht = face.height;

	int x = face.x;
	int y = face.y;

	int xm = x + face.width;
	int ym = y + face.height;

	int eye_x = eyes.x;
	int eye_y = eyes.y;

	int wid_eye = eyes.width;
	int ht_eye = eyes.height;

	if( wid_eye != 0 ) {
		x = MAX( eye_x - ceil(wid_eye/5.0), 1 );
		y = MAX( eye_y - ceil(ht_eye/1.5), 1 );

		xm = MIN( eye_x + wid_eye + ceil(wid_eye/5.0), imgWid - 1 );
		ym = face.y+ht;
	} else {
		/*x = MAX( x - ceil(wid/10.0), 1 );
		y = MAX( y - ceil(ht/6.0), 1 );

		xm = MIN( xm + floor(wid/10.0), imgWid - 1 );
		ym = MIN( ym + floor(ht/6.0), imgHt - 1 );*/
	}

	/*xm = MIN( xm + floor(wid/10.0), imgWid - 1 );
	ym = MIN( ym + floor(ht/6.0), imgHt - 1 );*/

	face.width = xm - x;
	face.height = ym - y;
	//face.height = MAX(ym - y, face.width);
	face.x = x;
	face.y = y;
}

void procrustes(float* pdat, float* pbase, int numPts, float* pout, float* tform) {
	int i;
	float *pdat0 = pdat, *pbase0 = pbase;
	
	float ux=0, uy=0, uxp=0, uyp=0, s=0, w1=0, w2=0;
	float tx, ty, txp, typ;
	for(i=0;i<numPts;i++)
	{
		tx = *pdat++;
		ty = *pdat++;

		txp = *pbase++;
		typ = *pbase++;

		ux+=tx;
		uy+=ty;
		uxp+=txp;
		uyp+=typ;

		s+=tx*tx+ty*ty;

		w1+=txp*ty - typ*tx;
		w2+=tx*txp + ty*typ;
		
	}

	ux /= numPts;
	uy /= numPts;
	uxp /= numPts;
	uyp /= numPts;

	float pmat[16], invp[16];
	pmat[0] = ux; pmat[1] = uy; pmat[2] = 1; pmat[3] = 0;
	pmat[4] = uy; pmat[5] =-ux; pmat[6]=0; pmat[7]=1;
	pmat[8] = 0; pmat[9] = s; pmat[10]=numPts*uy; pmat[11] = -numPts*ux;
	pmat[12] = s; pmat[13] = 0; pmat[14]=numPts*ux; pmat[15] = numPts*uy;

	float rmat[4];
	rmat[0] = uxp; rmat[1] = uyp; rmat[2] = w1; rmat[3] = w2;

	// pmat^-1 * rmat;
	CvMat MatP = cvMat(4, 4, CV_32FC1, pmat);
	CvMat MatR = cvMat(4, 1, CV_32FC1, rmat);

	CvMat MatPinv = cvMat(4, 4, CV_32FC1, invp);
	cvInvert(&MatP, &MatPinv);

	CvMat MatRes = cvMat(4, 1, CV_32FC1, tform);

	cvGEMM(&MatPinv, &MatR, 1, 0, 0, &MatRes);

	//	assemble output:
	pdat = pdat0;
	for(i=0;i<numPts;i++)
	{
		tx = *pdat++;
		ty = *pdat++;

		*pout++ = tform[0]*tx + tform[1]*ty + tform[2];
		*pout++ = -tform[1]*tx + tform[0]*ty + tform[3];
	}
}

void CopyImageToMat(IplImage *pImg, CvMat *pMat)
{
	uchar *pidat = (uchar*) pImg->imageData;
	float *pmdat = pMat->data.fl;

	for(int j=0;j<pImg->height;j++)
	{
		for(int i=0;i<pImg->width;i++)
		{
			pmdat[i] = (float)pidat[i];
		}
		pidat+=pImg->widthStep;
		pmdat+=pMat->cols;
	}
}

// aligns the data with transform inverse
int alignDataInverse(float *pdat, float *tform, int numPts, float *pout)
{
	// transform inverse of matrix sc*[cos -sin tx; sin cos ty; 1 1 1] is:
	// 1/sc*[cos sin -tx; -sin cos -ty; 1 1 1]
	// as inv(R) = R'
	int i;
	float tx, ty;
	float sc = tform[0]*tform[0]+tform[1]*tform[1];
	//	assemble output:
	for(i=0;i<numPts;i++)
	{
		tx = *pdat++ - tform[2];	
		ty = *pdat++ - tform[3];
		
		*pout++ = (tform[0]*tx - tform[1]*ty)/sc;
		*pout++ = (tform[1]*tx + tform[0]*ty)/sc;
	}

	return 0;
}

}

