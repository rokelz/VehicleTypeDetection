

#include <sys/resource.h> // memory management.
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "struct.h"
#include <unistd.h>
#include <ctype.h>


using namespace std;

static int frame_count = 0;
static int ID = 1;
static int StopReflesh = 0;
static double averagethreshold = 0;

int kalman_num = 0;

IplImage *frame, *frame_now, *frame_gray_now, *frame_gray_temp, *frame_bkg,
		*frame_gray_pass = NULL; 

IplImage *background(IplImage *frame_gray_now, IplImage *frame_bkg,
		IplImage *frame_gray_pass, int frame_count, int StopReflesh) {
	CvMat *temp = 0;
	temp = cvCreateMat(frame_gray_now->height, frame_gray_now->width, CV_32FC1);

	if (!frame_bkg) {
		frame_bkg = cvCreateImage(
				cvSize(frame_gray_now->width, frame_gray_now->height),
				IPL_DEPTH_8U, frame_gray_now->nChannels);
		frame_bkg->origin = frame_gray_now->origin;
		cvCopy(frame_gray_now, frame_bkg, 0);
	}
	else if (!StopReflesh) {
		IplImage *frame_mask = 0;

		frame_mask = cvCreateImage(
				cvSize(frame_gray_now->width, frame_gray_now->height),
				IPL_DEPTH_8U, 1);
		frame_mask->origin = frame_gray_now->origin;

		cvAbsDiff(frame_gray_now, frame_gray_pass, frame_mask);

		cvThreshold(frame_mask, frame_mask, 0, 255, CV_THRESH_BINARY_INV);

		cvConvert(frame_bkg, temp);

		if (frame_count < 20) {		
			cvRunningAvg(frame_gray_now, temp, (0.503 - (frame_count) / 110.0),
					frame_mask);
			if (frame_count < 10)
				cvSmooth(temp, temp, CV_GAUSSIAN, 3, 0, 0);
		} else
			cvRunningAvg(frame_gray_now, temp, 0.003, frame_mask);

		cvConvert(temp, frame_bkg);

		cvReleaseImage(&frame_mask);
		cvReleaseMat(&temp);
	}

	return frame_bkg;
}

PointSeqList foreground(IplImage *frame_gray_now, IplImage *frame_bkg, IplImage *final,
		int frame_count, double *averagethreshold) {
	const int CONTOUR_MAX_AREA = 4900; 
	const double SPEED_LIMIT = 80.00;
	CvScalar *mean, *std_dev;
	CvMemStorage *storage;
	CvSeq *cont;
	PointSeq *Centroids;
	PointSeqList Head = NULL;
	double threshold_now;

	IplImage *pyr = cvCreateImage(
			cvSize((frame_gray_now->width & -2) / 2,
					(frame_gray_now->height & -2) / 2), 8, 1);
	IplImage *temp = cvCreateImage(
			cvSize(frame_gray_now->width, frame_gray_now->height), 8, 1);

	temp->origin = 1;
	mean = (CvScalar *) malloc(sizeof(CvScalar));
	std_dev = (CvScalar *) malloc(sizeof(CvScalar));

	cvAbsDiff(frame_gray_now, frame_bkg, frame_gray_now);

	cvAvgSdv(frame_gray_now, mean, std_dev, NULL);

	threshold_now = 2.3 * std_dev->val[0];

	if (frame_count < 55) { 
		if (*averagethreshold < threshold_now) {
			*averagethreshold = threshold_now;
		}
	} else if (threshold_now < *averagethreshold) {
		threshold_now = *averagethreshold;
	} else {
		*averagethreshold = ((frame_count - 1) * (*averagethreshold)
				+ threshold_now) / frame_count;
	}
	cvThreshold(frame_gray_now, frame_gray_now, threshold_now, 255,
			CV_THRESH_BINARY);

	cvErode(frame_gray_now, frame_gray_now, NULL, 1);
	cvDilate(frame_gray_now, frame_gray_now, NULL, 3);
	cvErode(frame_gray_now, frame_gray_now, NULL, 1);

	cvSmooth(frame_gray_now, temp, CV_GAUSSIAN, 5, 3, 0);
	cvThreshold(temp, temp, 0, 255, CV_THRESH_BINARY);


	cvPyrDown(temp, pyr, CV_GAUSSIAN_5x5);
	cvDilate(pyr, pyr, 0, 1);
	cvPyrUp(pyr, temp, CV_GAUSSIAN_5x5);
	cvSmooth(temp, temp, CV_GAUSSIAN, 5, 3, 0);
	cvAvgSdv(temp, mean, std_dev, NULL);
	cvThreshold(temp, temp, mean->val[0], 255, CV_THRESH_BINARY);

	cvDilate(temp, temp, NULL, 1);
	cvErode(temp, temp, NULL, 1);

	cvShowImage("temp", temp);
	storage = cvCreateMemStorage(0);
	cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint),
			storage);

	cvFindContours(temp, storage, &cont, sizeof(CvContour), CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	cvZero(frame_gray_now);

	for (; cont; cont = cont->h_next) {
		if (cont->total < 20) //the number of sequence elements
			continue;

		CvRect r = ((CvContour*) cont)->rect;
		if (r.height * r.width > CONTOUR_MAX_AREA){
			CvMoments moments;
			cvMoments(cont, &moments, 0);
			CvPoint Center = cvPoint(cvRound(moments.m10 / moments.m00),
					cvRound(moments.m01 / moments.m00));
			if (Center.x > 0 && Center.x <= 640 && Center.y > 210
					&& Center.y <= 320)
			{
				cvRectangleR(final, r, CV_RGB(255,255,0), 2);
				cvCircle(final, Center, 2, CV_RGB(255,255,0), -1, 4, 0);
				Centroids = (PointSeq *) malloc(sizeof(PointSeq));
				Centroids->Point = Center;
				Centroids->ID = 0;
				Centroids->contourArea = r.height * r.width;
				Centroids->next = Head;
				Head = Centroids;
				if (Head->next) {
					Head->next->pre = Head;
				}
			}
		}
	}

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&pyr);
	cvReleaseImage(&temp);
	delete[] mean;
	delete[] std_dev;
	return Head;
}
KalmanPoint *NewKalman(PointSeq *Point_New, int ID, int frame_count, int contourArea) {
	const float A[] = { 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1 };
	const float H[] = { 1, 0, 0, 0, 0, 1, 0, 0 };

	KalmanPoint *Kalmanfilter;
	Kalmanfilter = (KalmanPoint *) malloc(sizeof(KalmanPoint));
	CvKalman *Kalman = cvCreateKalman(4, 2, 0);

	float measure[2] =
			{ (float) Point_New->Point.x, (float) Point_New->Point.y };
	CvMat measurement = cvMat(2, 1, CV_32FC1, measure);
	memcpy(Kalman->transition_matrix->data.fl, A, sizeof(A));
	memcpy(Kalman->measurement_matrix->data.fl, H, sizeof(H));
	cvSetIdentity(Kalman->error_cov_post, cvRealScalar(10));
	cvSetIdentity(Kalman->process_noise_cov, cvRealScalar(1e-5));
	cvSetIdentity(Kalman->measurement_noise_cov, cvRealScalar(1e-1));
	cvZero(Kalman->state_post);
	cvmSet(Kalman->state_post, 0, 0, Point_New->Point.x);
	cvmSet(Kalman->state_post, 1, 0, Point_New->Point.y);
	const CvMat *prediction = cvKalmanPredict(Kalman, 0);
	const CvMat *correction = cvKalmanCorrect(Kalman, &measurement);
	Kalmanfilter->Point_now = Point_New->Point;
	Kalmanfilter->Point_pre = Point_New->Point;
	Kalmanfilter->firstPoint = Point_New->Point;
	Kalmanfilter->firstFrame = frame_count; 
	Kalmanfilter->lastFrame = frame_count; 
	Kalmanfilter->Kalman = Kalman;
	Kalmanfilter->ID = Point_New->ID = ID;
	Kalmanfilter->contourArea=contourArea;
	//vehicle =2; 0 = undefined;
	if (Kalmanfilter->contourArea >= 4000){
		Kalmanfilter->jenis = 2;
	}  else {
		Kalmanfilter->jenis = 0;
	}
	Kalmanfilter->Loss = 0;

	return Kalmanfilter;
}
PointSeqList KalmanProcess(KalmanPoint *Kalmanfilter, PointSeqList Points,
		IplImage *temp, IplImage *final, int *StopReflesh, int frame_count) {
	CvPoint Centroid;
	CvPoint Point_Find;
	const CvMat *Prediction, *correction;
	CvMat measurement;
	int delta =10;

	Point_Find.x = Point_Find.y = 0;

	const CvMat *state_ptMat_pre = Kalmanfilter->Kalman->state_post;
	Prediction = cvKalmanPredict(Kalmanfilter->Kalman, 0);


	const CvMat *Prediction_error = Kalmanfilter->Kalman->error_cov_pre;

	double error_x_pre = cvmGet(Prediction_error, 0, 0);
	double error_y_pre = cvmGet(Prediction_error, 1, 1);


	Centroid = cvPoint(cvRound(cvmGet(Prediction, 0, 0)),
			cvRound(cvmGet(Prediction, 1, 0)));

	if (Centroid.x > temp->width + delta || Centroid.x < 0 - delta
			|| Centroid.y > temp->height + delta || Centroid.y < 0 - delta) {
		cvReleaseKalman(&(Kalmanfilter->Kalman));
		Kalmanfilter->Kalman = NULL;
		return Points;
	}
	PointSeq *find = NULL, *pt = Points;
	int direction = 640 * 480;//webcam logitech
	while (pt) {
		int x = pt->Point.x - Centroid.x;
		int y = pt->Point.y - Centroid.y;
		int t = x * x + y * y;
		if (t < direction) {
			direction = t;

			find = pt;
			Point_Find = pt->Point;
		}
		pt = pt->next;
	}

	if (sqrt(direction) > 30) {
		Point_Find = Centroid;
		find = NULL;
		Kalmanfilter->Loss++;
	}

	if (find) {
		if (find == Points) {
			Points = Points->next;
			delete[] find;
			find = Points;
		} else {
			if (find->next == NULL) {
				find = find->pre;
				delete[] find->next;
				find->next = NULL;
			} else {
				PointSeq *s;
				s = find->next;
				find->pre->next = find->next;
				find->next->pre = find->pre;
				delete[] find;
				find = s;
			}
		}
	} else {
		Point_Find = Centroid;
		Kalmanfilter->Loss++;
	}

	Kalmanfilter->Point_pre = Kalmanfilter->Point_now;
	Kalmanfilter->Point_now = Point_Find;
	Kalmanfilter->lastFrame = frame_count;

	int delta_x = abs(Kalmanfilter->Point_now.x - Kalmanfilter->Point_pre.x);
	int delta_y = abs(Kalmanfilter->Point_now.y - Kalmanfilter->Point_now.y);
	if (delta_x <= 2 && delta_y <= 2) {
		*StopReflesh = 1;
	} else
		*StopReflesh = 0;

	float measure[2] = { (float) Point_Find.x, (float) Point_Find.y };
	measurement = cvMat(2, 1, CV_32FC1, measure);
	correction = cvKalmanCorrect(Kalmanfilter->Kalman, &measurement);
	const CvMat *state_ptMat_now = Kalmanfilter->Kalman->state_post;
	const CvPoint state_pt_pre = cvPoint(cvRound(cvmGet(state_ptMat_pre, 0, 0)),
			cvRound(cvmGet(state_ptMat_pre, 1, 0)));
	const CvPoint state_pt_now = cvPoint(cvRound(cvmGet(state_ptMat_now, 0, 0)),
			cvRound(cvmGet(state_ptMat_now, 1, 0)));
	cvLine(final, Kalmanfilter->firstPoint, state_pt_now, CV_RGB(255, 255, 0),
			1, CV_AA, 0);
	char text[10];
	
	CvFont font;
	cvInitFont(&font, CV_FONT_VECTOR0, 0.5f, 0.5f, 0, 2);
	cvPutText(final, text, state_pt_now, &font, CV_RGB(255, 0, 0));

	

	return Points;
}
double GetTickCount(void) {
	struct timespec now;
	if (clock_gettime(CLOCK_MONOTONIC, &now))
		return 0;
	return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

int main() {
	filterList List = NULL;
	int key;
	int time = GetTickCount();
	int vehiCounter = 0;
	
	double sumspeed1 = 0.0;
	double avgspeed1 = 0.0;
	int n1 = 0;
	int i = 0;

	CvCapture *input_camera = cvCaptureFromAVI(
			"/home/ruwiniopatha/Desktop/kalmanTest2/src/clip.mp4");
	frame = cvQueryFrame(input_camera);

	cvNamedWindow("Capturing Image ...", 0);

	cvResizeWindow("Capturing Image ...",
			(int) cvGetCaptureProperty(input_camera, CV_CAP_PROP_FRAME_HEIGHT),
			(int) cvGetCaptureProperty(input_camera, CV_CAP_PROP_FRAME_WIDTH));

	CvSize size = cvSize(
			(int) cvGetCaptureProperty(input_camera, CV_CAP_PROP_FRAME_HEIGHT),
			(int) cvGetCaptureProperty(input_camera, CV_CAP_PROP_FRAME_WIDTH));

	int fps = cvGetCaptureProperty(input_camera, CV_CAP_PROP_FPS);

	//CvVideoWriter *writer = cvCreateVideoWriter(
	//		"/home/ruwiniopatha/Desktop/kalmanTest2/src/out.mp4",
	//		CV_FOURCC('I', 'Y', 'U', 'V'), fps, size, -1);

	struct rusage usage;
	while (frame != NULL) {
		cout<<frame_count<<endl;
		frame_count++;
		i++;
		frame = cvQueryFrame(input_camera);
		frame_now = cvCreateImage(cvSize(frame->width, frame->height),
		IPL_DEPTH_8U, frame->nChannels);
		cvCopy(frame, frame_now, 0);
		frame_now->origin = frame->origin;
		frame_gray_now = cvCreateImage(cvSize(frame->width, frame->height),
		IPL_DEPTH_8U, 1);
		frame_gray_now->origin = frame->origin;
		cvCvtColor(frame, frame_gray_now, CV_BGR2GRAY);
		frame_gray_now->origin = frame->origin;
		frame_gray_temp = cvCreateImage(cvSize(frame->width, frame->height),
		IPL_DEPTH_8U, 1);
		frame_gray_temp->origin = frame->origin;
		cvCopy(frame_gray_now, frame_gray_temp, 0);
		frame_gray_temp->origin = frame_gray_now->origin;

		frame_bkg = background(frame_gray_now, frame_bkg, frame_gray_pass,
				frame_count, StopReflesh);
		frame_bkg->origin = 1;

		PointSeqList Centroids = foreground(frame_gray_now, frame_bkg,
				frame_now, frame_count, &averagethreshold);
		if (!List) {
			PointSeq *s, *q = Centroids;
			while (q) {

				KalmanPoint *Kalman_Now;
				Kalman_Now = NewKalman(q, ID, frame_count, q->contourArea);

				kalman_num++;
				ID++;
				cout << ID << endl;
				Kalman_Now->next = List;
				List = Kalman_Now;
				if (List->next) {
					List->next->pre = List;
				}

				if (q = Centroids) {
					Centroids = Centroids->next;
				}
				s = q->next;
				delete[] q;
				q = s;
			}
		}

		else {
			KalmanPoint *k = List;
			while (k) {
				PointSeq *q = Centroids;

				Centroids = KalmanProcess(k, q, frame_gray_now, frame_now,
						&StopReflesh, frame_count);

				if (k->Kalman == NULL || k->Loss > 3) { //k-Loss > 3 original
					kalman_num--;
					double dx = abs(k->Point_now.x - k->firstPoint.x) * 20;
					double dy = abs(k->Point_now.y - k->firstPoint.y) * 20;
					double displacement = sqrt(dx * dx + dy * dy);
					double timeTaken = (k->lastFrame - k->firstFrame)
							* (1000 / fps);
					double speed = (displacement / timeTaken) * 36;
					if (speed > 0) {
						if (k->Point_now.x <= 640) {
							n1++;
							sumspeed1 += speed;
							if (k->jenis == 2) {
								vehiCounter++;
							}  else {
								cout << "undefined" << endl;
							}
							avgspeed1 = sumspeed1 / n1;
						}
					}
					cvReleaseKalman(&k->Kalman);
					if (k == List) {
						List = List->next;
						delete[] k;
						k = List;
					} else {
						if (k->next == NULL) {
							k = k->pre;
							delete[] k->next;
							k->next = NULL;
						} else {
							KalmanPoint *s;
							s = k->next;
							k->pre->next = k->next;
							k->next->pre = k->pre;
							delete[] k;
							k = s;
						}
					}
				} else {
					k = k->next;
				}
			}

			if (Centroids) {
				PointSeq *s, *p = Centroids;
				while (p) {
					if (p->ID == 0) {
						KalmanPoint *Kalman_Now;
						Kalman_Now = NewKalman(p, ID, frame_count,
								p->contourArea);

						kalman_num++;
						ID++;
						Kalman_Now->next = List;
						List = Kalman_Now;
						if (List->next) {
							List->next->pre = List;
						}
					}

					if (p = Centroids) {
						Centroids = Centroids->next;
					}
					s = p->next;
					delete[] p;
					p = s;
				}
			}
		}

		char text[10];

		char avgspeed[20];
		char countMot[10];
		char countMob[10];
		char countTruksed[10];
		char countTrukbes[10];

		CvFont font;
		cvInitFont(&font, CV_FONT_VECTOR0, 0.7f, 0.7f, 0, 2);
		cvPutText(frame_now, text, cvPoint(420, 380), &font,
				CV_RGB(255, 255, 255));

		
		sprintf(countMob, "Number of vehicles %d ", vehiCounter);
		cvPutText(frame_now, countMob, cvPoint(380, 420), &font,
				CV_RGB(255, 255, 255));

		
		cvLine(frame_now, cvPoint(0, 420), cvPoint(640, 420), cvScalar(255), 2,
		CV_AA, 0);	
		cvLine(frame_now, cvPoint(0, 210), cvPoint(640, 210), cvScalar(255), 2,
		CV_AA, 0);	

		cvReleaseImage(&frame_gray_pass);
		if (!frame_gray_pass)
			frame_gray_pass = cvCreateImage(cvSize(frame->width, frame->height),
			IPL_DEPTH_8U, 1);
		frame_gray_pass->origin = 1;
		cvCopy(frame_gray_temp, frame_gray_pass, NULL);

		
		cvShowImage("Capturing Image 2...", frame_gray_now);
		cvShowImage("Capturing Image 1...", frame_bkg);
		cvShowImage("Capturing Image ...", frame_now);

		
		//cvWriteFrame(writer, frame_now);
		char path[255];
		strcpy(path, "/home/ruwiniopatha/Desktop/kalmanTest2/src/clip.jpg");
		cvSaveImage(path, frame_now);

		if (i == fps * 60 * 1) {
			
			vehiCounter = 0;
			
			i = 0;
			sumspeed1 = 0.0;
			avgspeed1 = 0.0;
			n1 = 0;
		
		}

		cvReleaseImage(&frame_now);
		cvReleaseImage(&frame_gray_now);
		cvReleaseImage(&frame_gray_temp);

		

		key = cvWaitKey(10);
		if (key == 27)
			break;

	}


	cvReleaseCapture(&input_camera);

	//cvReleaseVideoWriter(&writer);
	cvDestroyWindow("Capturing Image ...");

	return 0;
}

