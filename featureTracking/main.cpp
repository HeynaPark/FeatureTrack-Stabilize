#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <thread>

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <time.h>

using namespace std;
using namespace cv;
using std::thread;


double PROCESS_NOISE = 4e-3;
double measurementNoise = 0.25;


struct Trajectory
{
	Trajectory() {}
	Trajectory(double _x, double _y, double _a) {
		x = _x;
		y = _y;
		a = _a;
	}
	// "+"
	friend Trajectory operator+(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x * c2.x, c1.y * c2.y, c1.a * c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory& c1, const Trajectory& c2) {
		return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory& rx) {
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x, y, a);
	}

	double x;
	double y;
	double a; 
};

struct TransformParam
{
	TransformParam() {}
	TransformParam(double _dx, double _dy, double _da) {
		dx = _dx;
		dy = _dy;
		da = _da;
	}

	double dx;
	double dy;
	double da; 
};


int cnt = 0;	//for debuging
int k = 1;		//for kalman filter cnt first loop
int newLeft = 0;
int newTop = 0;
int newWidth = 1920;
int newHeight = 1080;
double border = 0.005;

Mat last_T;


cuda::GpuMat curGray;
cuda::GpuMat prevGray;
Ptr<cuda::CornersDetector> detector;
Ptr<cuda::SparsePyrLKOpticalFlow> pryLK_sparse;
Ptr<cuda::FastFeatureDetector> gpuFastDetector;

double a = 0, x = 0, y = 0;
vector <Trajectory> trajectory;
vector <TransformParam> prev_to_cur_transform;
vector <Trajectory> smoothed_trajectory;
vector <TransformParam> new_prev_to_cur_transform;


cuda::GpuMat preimg;
cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
vector<Point2f> prev_corner, cur_corner;
vector<Point2f> prev_corner2, cur_corner2;


cuda::GpuMat gpuErr;
vector <float> err;

cuda::GpuMat prev_;
Mat mask;

Trajectory X;
Trajectory X_;
Trajectory P;
Trajectory P_;
Trajectory K;
Trajectory z;
Trajectory Q;
Trajectory R;



void StabilizeFrameNonUpdate(cuda::GpuMat prev, cuda::GpuMat& cur, cuda::GpuMat& draw);
void StablizeFrame(cuda::GpuMat prev, cuda::GpuMat& cur, cuda::GpuMat& draw);
void MakeMask(Mat img, cuda::GpuMat& mask);
void ReclassifyInlier(const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier);
static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, Scalar line_color);
static void DrawArrows_new(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color);

/*
//image to json
String filename = "ImageToJson.json";
void writeData(Mat img) {
	FileStorage fs(filename, FileStorage::WRITE);

	if (!fs.isOpened())
		cout << "File open failed." << endl;

	fs << "data" << img;

	fs.release();
}
*/

cuda::GpuMat prevImgCuda;

cuda::GpuMat mask_gpu;
Mat dst;
Mat src;
Mat img;
Mat manualMask;
Mat maskImg;

bool bFirstForSave = true;
bool bFirst = true;
bool bMoveStabil = true;

int MASK_BRUSH_SIZE = 100;


//VideoCapture cap("D:/test/stabil/ncaa1.mp4");
VideoCapture cap;
clock_t start, end;
double result;

void dummy() {
	Mat dummy = imread("dummy.png", IMREAD_GRAYSCALE);
	cuda::GpuMat dummy_gpu, out;
	dummy_gpu.upload(dummy);
	StablizeFrame(dummy_gpu, dummy_gpu, out); 
	StablizeFrame(dummy_gpu, dummy_gpu, out);

	bFirst = true;

}

void onMouseEvent(int event, int x, int y, int flags, void* dstImage) {
	
	Mat mouseImage = *(Mat*)dstImage;

	switch (event) {
		case EVENT_MOUSEMOVE:
			if (flags & EVENT_LBUTTONDOWN) {
				circle(mouseImage, Point(x, y), MASK_BRUSH_SIZE, Scalar::all(0), -1);
				circle(manualMask, Point(x, y), MASK_BRUSH_SIZE, Scalar::all(0), -1);
			}
			break;
	}
	imshow("Mouse event", mouseImage);
	imshow("mask img", manualMask);
}


double THRESHOLD = 2;
int RESIZE_FACTOR = 3;
int minDistance = 40;
int winSize = 21;
int maxCorner = 300;

string Filename = "KBL/kbl124";



int main(int argc, char** argv)
{
	cuda::GpuMat prevImgGpu, curImgGpu, outputSrcGpu;

	cap = VideoCapture("D:/test/stabil/" + Filename + ".mp4");
	int fps = (int)cap.get(CAP_PROP_FPS);
	int width = cap.get(CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CAP_PROP_FRAME_HEIGHT);

	if (width == 3840)
		THRESHOLD = 4 / (double)RESIZE_FACTOR;
	if (width == 1920)
		THRESHOLD = 2 / (double)RESIZE_FACTOR;

	VideoWriter dst_output(Filename + "(dst)_size"+to_string(RESIZE_FACTOR)+"_thresh" +to_string(THRESHOLD)+"_dist30.mp4", VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, Size(width, height));
	VideoWriter src_output(Filename + "(src)_size"+to_string(RESIZE_FACTOR)+"_thresh" +to_string(THRESHOLD)+"_dist30.mp4", VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, Size(width / RESIZE_FACTOR, height / RESIZE_FACTOR));
	//VideoWriter cmp_output(Filename+"(cmp)_thresh_0.5.mp4", VideoWriter::fourcc('D', 'I', 'V', 'X'), fps, Size(1920,540));

	if (!cap.isOpened()) {
		printf("Can't open the file.");
		return -1;
		}

		cap >> img;

	//	MakeMask(img, mask_gpu);


	cout << "image size: " << width << endl;
	cout << "threshold: " << THRESHOLD << endl;


	while (1) {
	
		cap >> img;

		if (img.empty()) {
			printf("img empty.");
			break;
		}

		start = clock();
		prevImgGpu.upload(img);

		StablizeFrame(prevImgGpu, curImgGpu, outputSrcGpu);		// prev 가 매번 업데이트(직전 프레임)
		//stab_live_keepFirstframe(prev_gpu, cur_gpu, cur_gpu2);	// prev가 첫 프레임으로 고정
		
		if (bFirstForSave)
		{
			prevImgGpu.download(dst);
			prevImgGpu.download(src);

			resize(dst, dst, img.size());
			bFirstForSave = false;
		}
		else
		{
			curImgGpu.download(dst);
			outputSrcGpu.download(src);
		}


		
		if (dst.cols == 3840)
			resize(dst, dst, Size(1920, 1080), INTER_AREA);
		namedWindow("dst", WINDOW_FULLSCREEN);
		moveWindow("src", 3840,540);
		moveWindow("dst", 0, 0);
		imshow("dst", dst);
		imshow("src", src);
		//waitKey(1);

	/*		if (src.cols == width / resize_factor)
			src_output.write(src);
		if(dst.cols == width)
			dst_output.write(dst);*/
		if (waitKey(1) == 27) {
			break;
		}


	}

	
	cap.release();
	dst_output.release();
	src_output.release();



	waitKey(0);


	return 0;
}





void StabilizeFrameNonUpdate(cuda::GpuMat img, cuda::GpuMat& cur2, cuda::GpuMat& draw)
{

	Mat T(2, 3, CV_64F);


	if (bFirst) {
		bFirst = false;

		prev_ = img.clone();
		cuda::resize(prev_, prevGray, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
		cuda::cvtColor(prevGray, prevGray, COLOR_BGR2GRAY);

		//mask
		//mask = imread("mask2.png",IMREAD_GRAYSCALE);
		//mask_gpu.upload(mask);
		//cuda::resize(mask_gpu, mask_gpu, Size(img.cols / resize_factor, img.rows / resize_factor));
		

		
		double pstd = 4e-3;
		double cstd = 0.25;
		Q.x = pstd;
		Q.y = pstd;
		Q.a = pstd;
		R.x = cstd;
		R.y = cstd;
		R.a = cstd;

		newLeft = img.rows * border;
		newTop = img.cols * border;

		detector = cuda::createGoodFeaturesToTrackDetector(prevGray.type(), 1000, 0.01, 20);
		pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
		
	}

	else {

		clock_t  start, end;
		double  result;

		start = clock();


		cuda::resize(img, curGray, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
		cuda::cvtColor(curGray, curGray, COLOR_BGR2GRAY);


		cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
		vector<Point2f> prev_corner, cur_corner;
		vector<Point2f> prev_corner2, cur_corner2;


		cuda::GpuMat gpuStatus;
		cuda::GpuMat gpuErr;
		vector <uchar> status;
		vector <float> err;

		detector->detect(prevGray, prev_corner_gpu,mask_gpu);
		pryLK_sparse->calc(prevGray, curGray, prev_corner_gpu, cur_corner_gpu, gpuStatus, gpuErr);	


		//예외처리 필요
		prev_corner_gpu.download(prev_corner);
		cur_corner_gpu.download(cur_corner);
		gpuStatus.download(status);
		gpuErr.download(err);


		for (size_t i = 0; i < prev_corner.size(); i++) {
			if (status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}
		}

		vector<uchar> inlier;
		T = estimateAffinePartial2D(prev_corner2, cur_corner2, inlier, RANSAC, THRESHOLD); //similar to rigidtransform.
	//	T = getAffineTransform(prev_corner2, cur_corner2);


		T = T * RESIZE_FACTOR;  


		if (T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));



		/*x += dx;
		y += dy;
		a += da;

		z = Trajectory(x, y, a);

		if (k == 1) {
			X = Trajectory(0, 0, 0);
			P = Trajectory(1, 1, 1);
		}
		else
		{
			X_ = X;
			P_ = P + Q;
			K = P_ / (P_ + R);
			X = X_ + K * (z - X_);
			P = (Trajectory(1, 1, 1) - K) * P_;
		}

		double diff_x = (X.x - x);
		double diff_y = (X.y - y);
		double diff_a = (X.a - a);


		//dx = dx + diff_x;
		//dy = dy + diff_y;
		//da = da + diff_a;*/


		Mat T_;
		T.copyTo(T_);


		T_.at<double>(0, 2) = -dx;
		T_.at<double>(1, 2) = -dy;

		cout << cnt << "      dx: " << dx << "   dy: " << dy << endl << endl;

		end = clock();
		result = (double)(end - start);
		cout << "calc time : " << result << endl;

		T_.at<double>(0, 0) = 1;
		T_.at<double>(0, 1) = 0;
		T_.at<double>(1, 0) = 0;
		T_.at<double>(1, 1) = 1;


		if (cur_corner2.size() < 10 || dx > 10 || dy > 10)
			cur2 = img;
		else
		{
			cuda::warpAffine(img, cur2, T_, img.size());
		}

		cur2 = cur2(Range(newLeft, cur2.rows - newLeft), Range(newTop, cur2.cols - newTop));
		cuda::resize(cur2, cur2, img.size());

		prevImgCuda = img.clone();

		Mat gray;
		prevImgCuda.download(gray);
		resize(gray, gray, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
	//	drawArrows_new(gray, prev_corner2, cur_corner2, status, inlier, Scalar(50, 200, 255));
		draw.upload(gray);

		k++;
		cnt++;
	}
	preimg = img.clone();

}


void MakeMask(Mat img, cuda::GpuMat& mask){
	
	cuda::GpuMat prevGpu;

	manualMask = Mat(img.rows, img.cols, CV_8UC1, Scalar::all(255));
	prevGpu.upload(img);
	namedWindow("Mouse event", WINDOW_NORMAL);
	//resizeWindow("Mouse event", 600, 100);
	imshow("mask img", manualMask);
	imshow("Mouse event", img);
	setMouseCallback("Mouse event", onMouseEvent, (void*)&img);


	if (waitKey(0) == 32) {

		imwrite(Filename + "_mask.png",manualMask);
		mask.upload(manualMask);
		cuda::resize(mask, mask, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));

	}
	else if (waitKey(0) == 114 || waitKey(0) == 82) {
		Mat temp_mask = imread(Filename + "_mask.png",IMREAD_GRAYSCALE);
		mask.upload(temp_mask);
		cuda::resize(mask, mask, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
	}


	destroyWindow("Mouse event");
}


Mat newT(2, 3, CV_64F);
Mat T(2, 3, CV_64F);
Mat _T(2, 3, CV_64F);

void StablizeFrame(cuda::GpuMat img, cuda::GpuMat& cur2, cuda::GpuMat& draw)
{

	if (bFirst) {
		bFirst = false;

		cuda::resize(img, prevImgCuda, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
		cuda::cvtColor(prevImgCuda, prevGray, COLOR_BGR2GRAY);
	
	
		Q.x = PROCESS_NOISE;			//시스템 노이즈
		Q.y = PROCESS_NOISE;
		Q.a = PROCESS_NOISE;
		R.x = measurementNoise;			//측정 노이즈
		R.y = measurementNoise;
		R.a = measurementNoise;

		if (img.cols == 1920) {
			newLeft = 16;
			newTop = 9;
		}
		else if (img.cols == 3840) {
			newLeft = 16;
			newTop = 9;
		}
		else {
			cout << "[Exception] : image size is wrong" << endl;
			return;
		}

		newWidth = img.rows - newLeft;
		newHeight = img.cols - newTop;


		detector = cuda::createGoodFeaturesToTrackDetector(prevGray.type(), maxCorner, 0.01, minDistance);
		pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(winSize, winSize), 3, 10);

	}

	else {

		clock_t  start, end;
		double  result;

		start = clock();

		cuda::resize(prevImgCuda, prevGray, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
		cuda::resize(img, curGray, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
		cuda::cvtColor(prevGray, prevGray, COLOR_BGR2GRAY);
		cuda::cvtColor(curGray, curGray, COLOR_BGR2GRAY);


		cuda::GpuMat prevCornerGpu, curCornerGpu;
		vector<Point2f> prevCorner, curCorner;
		vector<Point2f> prevCorner2, curCorner2;
		vector<KeyPoint> keypoints;

		cuda::GpuMat gpuStatus;
		cuda::GpuMat gpuErr;
		vector <uchar> status;
		vector <float> err;

		detector->detect(prevGray, prevCornerGpu);
		pryLK_sparse->calc(prevGray, curGray, prevCornerGpu, curCornerGpu, gpuStatus, gpuErr);	//cold start issue
	

		if (prevCornerGpu.empty() || curCornerGpu.empty() || gpuStatus.empty() || gpuErr.empty()) {
			cout << "[Exception] : corner point is empty." << endl;
		}


		prevCornerGpu.download(prevCorner);
		curCornerGpu.download(curCorner);
		gpuStatus.download(status);
		gpuErr.download(err);
		


	//	cout << "corner size : " << prev_corner.size() << endl;

		for (size_t i = 0; i < prevCorner.size(); i++) {
			if (status[i]) {
				prevCorner2.push_back(prevCorner[i]);
				curCorner2.push_back(curCorner[i]);
			}
		}

		vector<uchar> inlier;
	
		T = estimateAffinePartial2D(prevCorner2, curCorner2, inlier , RANSAC, THRESHOLD); //similar to rigidtransform
	


		T = T * RESIZE_FACTOR;  //corner size 1/2

		
		if (T.data == NULL) {
			last_T.copyTo(T);
		}
		T.copyTo(last_T);



		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

	
		x += dx;
		y += dy;
		a += da;



		z = Trajectory(x, y, a);

		if (k == 1) {
			X = Trajectory(0, 0, 0);
			P = Trajectory(1, 1, 1);
		}
		else
		{
			X_ = X;
			P_ = P + Q;
			K = P_ / (P_ + R);
			X = X_ + K * (z - X_);
			P = (Trajectory(1, 1, 1) - K) * P_;
		}

		double diff_x = (X.x - x);
		double diff_y = (X.y - y);
		double diff_a = (X.a - a);


		Mat T_;
		T.copyTo(T_);


		T_.at<double>(0, 2) = diff_x;
		T_.at<double>(1, 2) = diff_y;

		T_.at<double>(0, 0) = 1;
		T_.at<double>(0, 1) = 0;
		T_.at<double>(1, 0) = 0;
		T_.at<double>(1, 1) = 1;


		end = clock();
		result = (double)(end - start);
	//	cout << "calc time : " << result << endl;


		if (curCorner2.size() < 10 || dx > 10 || dy > 10)
		{
			cur2 = img;
			cout << "			dx, dy is too big!			" << endl;
		}
		else
			cuda::warpAffine(img, cur2, T_, img.size());
			//cuda::warpPerspective(img, cur2, T_, img.size());

		
		cur2 = cur2(Range(newLeft, newWidth), Range(newTop, newHeight));
		cuda::resize(cur2, cur2, img.size());


		prevImgCuda = img.clone();
		curGray.copyTo(prevGray);


		Mat gray;
		prevImgCuda.download(gray);
		resize(gray, gray, Size(img.cols / RESIZE_FACTOR, img.rows / RESIZE_FACTOR));
		DrawArrows_new(gray, prevCorner2, curCorner2, status, inlier, Scalar(50, 200, 255));
		draw.upload(gray);



		k++;
		cnt++;
	}
	preimg = img.clone();

}

void ReclassifyInlier(const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier) {

	int green = 0;
	int yellow = 0;

	for (size_t i = 0; i < nextPts.size(); ++i)
	{
		if (status[i])
		{
			int line_thickness = 3;

			Point p = prevPts[i];
			Point q = nextPts[i];

			double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (double)(p.x - q.x) * (p.x - q.x));
			double angle = atan2((double)p.y - q.y, (double)p.x - q.x);

			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			if (inlier[i] == 1)
			{
				if (hypotenuse < 0.1) {
					green++;
					continue;
				}
				else {
					p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
					p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
	
					p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
					p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
	
					yellow++;
				}
			}
			else
			{
				p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
	
				p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
			}

		}
	}



	//if (nextPts.size() * 0.5 > yellow) {
	//	bMoveStabil = false;
	//}
	//else {
	//	bMoveStabil = true;
	//	
	//}
	//for (size_t i = 0; i < nextPts.size(); ++i)
	//{
	//	if (status[i])
	//	{
	//		Point p = prevPts[i];
	//		Point q = nextPts[i];

	//		double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (double)(p.x - q.x) * (p.x - q.x));
	//		if (inlier[i] == 1)
	//		{
	//			if (hypotenuse < 0.1) {

	//				green++;
	//				continue;
	//			}
	//	}
	//}




}


static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, Scalar line_color)
{
	int green = 0;
	int yellow = 0;


	for (size_t i = 0; i < nextPts.size(); ++i)
	{
		if (status[i])
		{
			int line_thickness = 3;

			Point p = prevPts[i];
			Point q = nextPts[i];

			double angle = atan2((double)p.y - q.y, (double)p.x - q.x);

			double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (double)(p.x - q.x) * (p.x - q.x));

			if (hypotenuse < 1.0) {
				circle(frame, p, 2, Scalar(50, 200, 40), 2, 8);
				green++;
				continue;
			}
			
			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			if (1)
			{  
				line(frame, p, q, line_color, line_thickness);

				p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
				line(frame, p, q, line_color, line_thickness);

				p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
				line(frame, p, q, line_color, line_thickness);
				yellow++;
			}
			else
			{
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);

				p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);

				p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);
			}
			
		}
	}
	cout << "nextPts size: " << nextPts.size() << "	  yellow : " << yellow << "		green : " << green << "		 //		green percent : " << ((double)green/ nextPts.size())*100<< "		yellow percent : "<< ((double)yellow / nextPts.size())*100 << endl;
}



static void DrawArrows_new(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color)
{
	int green = 0;
	int yellow = 0;

	for (size_t i = 0; i < nextPts.size(); ++i)
	{
		if (status[i])
		{
			int line_thickness = 3;

			Point p = prevPts[i];
			Point q = nextPts[i];

			double angle = atan2((double)p.y - q.y, (double)p.x - q.x);
			double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (double)(p.x - q.x) * (p.x - q.x));


			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			if (inlier[i] == 1)
			{
				if (hypotenuse < 1.0) {
					circle(frame, p, 2, Scalar(50, 200, 40), 2, 8);
					green++;
					continue;
				}

				else {
					line(frame, p, q, line_color, line_thickness);

					p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
					p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
					line(frame, p, q, line_color, line_thickness);

					p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
					p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
					line(frame, p, q, line_color, line_thickness);
					yellow++;
				}
			}
			else
			{
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);

				p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);

				p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);
			}
		}
	}
	
	double green_per = ((double)green / nextPts.size()) * 100;
	double yellow_per = ((double)yellow / nextPts.size()) * 100;

	//cout << "nextPts size: " << nextPts.size() << " green  : " << green << "		yellow  : " << yellow << "	yellow / green  " << (double)yellow / green << endl;
	//cout << "nextPts size: " << nextPts.size() <<" green percent : " << green_per<< "		yellow percent : " << yellow_per  <<"	Total score: " << abs(green_per-yellow_per)/100.0<< endl;
}