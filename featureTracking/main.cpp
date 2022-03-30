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

#include <time.h>

using namespace std;
using namespace cv;
using std::thread;

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
	double da; // angle
};

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
	double a; // angle
};


void stab_live_frame(cuda::GpuMat prev, cuda::GpuMat& cur, cuda::GpuMat& draw);
void stab_live_size(cuda::GpuMat prev, cuda::GpuMat& cur, cuda::GpuMat& draw);
static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color);
static void drawArrows_new(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color);


//String filename = "ImageToJson.json";
//
//void writeData(Mat img) {
//	FileStorage fs(filename, FileStorage::WRITE);
//
//	if (!fs.isOpened())
//		cout << "File open failed." << endl;
//
//	fs << "data" << img;
//
//	fs.release();
//}

int resize_factor = 2;

cuda::GpuMat prevImgCuda;
cuda::GpuMat prev_gpu, cur_gpu, cur_gpu2;
Mat dst;
Mat src;
Mat img;
bool first = true;
bool bDummy = false;
bool _first = true;



//VideoCapture cap("D:/test/stabil/ncaa1.mp4");
VideoCapture cap;
clock_t start, end;
double result;

void dummy() {
	Mat dummy = imread("dummy.png", IMREAD_GRAYSCALE);
	cuda::GpuMat dummy_gpu, out;
	dummy_gpu.upload(dummy);
	//t1 = thread(stab_live_size, dummy_gpu, dummy_gpu, out);
	stab_live_size(dummy_gpu, dummy_gpu, out);
	stab_live_size(dummy_gpu, dummy_gpu, out);
	bDummy = false;
	_first = true;

}

int main(int argc, char** argv)
{

	string Filename = "merge_crop_short";

	for (int i = 0; i < 1; i++) {
		cuda::GpuMat prev_gpu, cur_gpu, cur_gpu2;

		cap = VideoCapture("D:/test/stabil/" + Filename + ".mp4");
		int fps = (int)cap.get(CAP_PROP_FPS);
		int width = cap.get(CAP_PROP_FRAME_WIDTH);
		int height = cap.get(CAP_PROP_FRAME_HEIGHT);
		VideoWriter dst_output(Filename+"(dst)_after.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
		VideoWriter src_output(Filename+"(src)_after.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width / resize_factor, height / resize_factor));


		if (!cap.isOpened()) {
			printf("Can't open the file.");
			return -1;
		}


		//thread t1(dummy);
		//t1.join();
		////	t1.detach();
		//thread t2(loop);
		//t2.join();

		while (1) {
			 

			cap >> img;

			if (img.empty()) {
				printf("img empty.");
				break;
			}

			start = clock();
			prev_gpu.upload(img);

			stab_live_size(prev_gpu, cur_gpu, cur_gpu2);
			if (first)
			{
				prev_gpu.download(dst);
				prev_gpu.download(src);

				resize(dst, dst, img.size());
				first = false;
			}
			else
			{
				cur_gpu.download(dst);
				cur_gpu2.download(src);
			}

	

			//moveWindow("dst", 0, 0);
			imshow("src", src);
			resize(dst, dst, Size(1920, 1080), INTER_AREA);
			imshow("dst", dst);



			if (src.cols == width / resize_factor)
				//src_output.write(src);
		//	dst_output.write(dst);
			if (waitKey(1) == 27) {
				break;
			}


		}

		
		cap.release();
		dst_output.release();
		src_output.release();


	}


	waitKey(0);


	return 0;
}


Mat last_T;
int cnt = 0;
Trajectory X;
Trajectory X_;
Trajectory P;
Trajectory P_;
Trajectory K;
Trajectory z;


cuda::GpuMat cur_grey;
cuda::GpuMat prev_grey;
Ptr<cuda::CornersDetector> detector;
Ptr<cuda::SparsePyrLKOpticalFlow> pryLK_sparse;


double a = 0, x = 0, y = 0;
int k = 1;
vector <Trajectory> trajectory;
vector <TransformParam> prev_to_cur_transform;
vector <Trajectory> smoothed_trajectory;
vector <TransformParam> new_prev_to_cur_transform;
Trajectory Q;
Trajectory R;

int frame_cnt = 0;
cuda::GpuMat preimg;
cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
vector<Point2f> prev_corner, cur_corner;
vector<Point2f> prev_corner2, cur_corner2;


cuda::GpuMat gpuErr;
vector <float> err;


double maxPixel_x = 0;
double maxPixel_y = 0;
double border = 0.01;
cuda::GpuMat prev_;

Mat mask;
cuda::GpuMat mask_gpu;


void stab_live_frame(cuda::GpuMat img, cuda::GpuMat& cur2, cuda::GpuMat& draw)
{


	Mat T(2, 3, CV_64F);


	if (_first) {


		if (bDummy)
		{
			prev_grey = img.clone();
		}
		else {

			Mat frame0 = imread("frame2.png", IMREAD_GRAYSCALE);
			
			prev_.upload(frame0);
	
			cuda::resize(prev_, prev_grey, Size(img.cols / resize_factor, img.rows / resize_factor));
	

			//mask
			mask = imread("mask2.png",IMREAD_GRAYSCALE);
			mask_gpu.upload(mask);
			cuda::resize(mask_gpu, mask_gpu, Size(img.cols / resize_factor, img.rows / resize_factor));
			cuda::resize(img, prevImgCuda, Size(img.cols / resize_factor, img.rows / resize_factor));
			cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);
		}


		_first = false;
		double pstd = 4e-3;
		double cstd = 0.25;
		Q.x = pstd;
		Q.y = pstd;
		Q.a = pstd;
		R.x = cstd;
		R.y = cstd;
		R.a = cstd;

		maxPixel_x = img.rows * border;
		maxPixel_y = img.cols * border;

		if (!bDummy) {

			detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30);
			pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
		}
	}

	else {

		clock_t  start, end;
		double  result;

		start = clock();

		if (bDummy) {
			cur_grey = img.clone();
		}
		else {
			prev_grey = prev_.clone();
			cuda::resize(img, cur_grey, Size(img.cols / resize_factor, img.rows / resize_factor));
			cuda::cvtColor(cur_grey, cur_grey, COLOR_BGR2GRAY);

		}

		cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
		vector<Point2f> prev_corner, cur_corner;
		vector<Point2f> prev_corner2, cur_corner2;


		cuda::GpuMat gpuStatus;
		cuda::GpuMat gpuErr;
		vector <uchar> status;
		vector <float> err;

		detector->detect(prev_grey, prev_corner_gpu, mask_gpu);
		pryLK_sparse->calc(prev_grey, cur_grey, prev_corner_gpu, cur_corner_gpu, gpuStatus, gpuErr);	//cold start issue


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
		T = estimateAffinePartial2D(prev_corner2, cur_corner2, inlier, RANSAC, 3); //similar to rigidtransform.


		T = T * resize_factor;  //corner size 1/2


		if (T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));


		Mat T_;
		T.copyTo(T_);


		T_.at<double>(0, 2) = -dx;
		T_.at<double>(1, 2) = -dy;


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

		cur2 = cur2(Range(maxPixel_x, cur2.rows - maxPixel_x), Range(maxPixel_y, cur2.cols - maxPixel_y));


		cuda::resize(cur2, cur2, img.size());

		prevImgCuda = img.clone();
		cur_grey.copyTo(prev_grey);

		Mat gray;
		prevImgCuda.download(gray);
		resize(gray, gray, Size(img.cols / resize_factor, img.rows / resize_factor));
		drawArrows_new(gray, prev_corner2, cur_corner2, status, inlier, Scalar(50, 200, 255));
		draw.upload(gray);

		k++;
		cnt++;
	}
	preimg = img.clone();

}


void stab_live_size(cuda::GpuMat img, cuda::GpuMat& cur2, cuda::GpuMat& draw)
{
	//if (img.cols == 3840)
	//	resize_factor = 4;


	Mat T(2, 3, CV_64F);


	if (_first) {


		if (bDummy)
		{
			prev_grey = img.clone();
		}
		else {

			Mat frame0 = imread("frame2.png", IMREAD_GRAYSCALE);

			prev_.upload(frame0);

			cuda::resize(prev_, prev_grey, Size(img.cols / resize_factor, img.rows / resize_factor));
			//cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY); 

			mask = imread("mask2.png", IMREAD_GRAYSCALE);
			mask_gpu.upload(mask);
			cuda::resize(mask_gpu, mask_gpu, Size(img.cols / resize_factor, img.rows / resize_factor));
			cuda::resize(img, prevImgCuda, Size(img.cols / resize_factor, img.rows / resize_factor));
			cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);
		}


		_first = false;
		double pstd = 4e-3;
		double cstd = 0.25;
		Q.x = pstd;
		Q.y = pstd;
		Q.a = pstd;
		R.x = cstd;
		R.y = cstd;
		R.a = cstd;

		maxPixel_x = img.rows * border;
		maxPixel_y = img.cols * border;

		if (!bDummy) {

			detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30);
			pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
		}
	}

	else {

		clock_t  start, end;
		double  result;

		start = clock();

		if (bDummy) {
			cur_grey = img.clone();
		}
		else {

			cuda::resize(prev_, prev_grey, Size(img.cols / resize_factor, img.rows / resize_factor));
			//cuda::resize(prevImgCuda, prev_grey, Size(img.cols / resize_factor, img.rows / resize_factor));
			cuda::resize(img, cur_grey, Size(img.cols / resize_factor, img.rows / resize_factor));
			//cuda::cvtColor(prev_grey, prev_grey, COLOR_BGR2GRAY);
			cuda::cvtColor(cur_grey, cur_grey, COLOR_BGR2GRAY);

		}

		cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
		vector<Point2f> prev_corner, cur_corner;
		vector<Point2f> prev_corner2, cur_corner2;


		cuda::GpuMat gpuStatus;
		cuda::GpuMat gpuErr;
		vector <uchar> status;
		vector <float> err;

		detector->detect(prev_grey, prev_corner_gpu);
		pryLK_sparse->calc(prev_grey, cur_grey, prev_corner_gpu, cur_corner_gpu, gpuStatus, gpuErr);	//cold start issue



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
		// Mat T_cur;
		// T_cur = estimateAffinePartial2D(cur_corner2, prev_corner2,  inlier, RANSAC, 3);
		T = estimateAffinePartial2D(prev_corner2, cur_corner2, inlier, RANSAC, 3); //similar to rigidtransform.
	   // T = estimateAffinePartial2D( cur_corner2, prev_corner2, inlier, RANSAC, 3); //similar to rigidtransform.



	   // T_cur = T_cur * 2;
		T = T * resize_factor;  //corner size 1/2
		// 

		//  T.at<double>(0, 2) = -T_cur.at<double>(0, 2);
	   // T.at<double>(1, 2) = -T_cur.at<double>(1, 2);


		if (T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		// writeData(last_T);

		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

		/*    double dx = T.at<double>(0, 2);
			  double dy = T.at<double>(1, 2);
			  double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));*/

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

		// calculate difference in smoothed_trajectory and trajectory
		double diff_x = (X.x - x);
		double diff_y = (X.y - y);
		double diff_a = (X.a - a);

		// calculate newer transformation array
		//dx = dx + diff_x;
		//dy = dy + diff_y;
		//da = da + diff_a;


		/*   T.at<double>(0, 0) = cos(da);
		   T.at<double>(0, 1) = -sin(da);
		   T.at<double>(1, 0) = sin(da);
		   T.at<double>(1, 1) = cos(da);*/

		Mat T_;
		T.copyTo(T_);

		//T_.at<double>(0, 2) = diff_x;
		//T_.at<double>(1, 2) = diff_y;

		// T_cur.copyTo(T_);  

		 //T_.at<double>(0, 2) = diff_x;
		 //T_.at<double>(1, 2) = diff_y;

		T_.at<double>(0, 2) = -dx;
		T_.at<double>(1, 2) = -dy;


		end = clock();
		result = (double)(end - start);
		cout << "calc time : " << result << endl;

		//
		T_.at<double>(0, 0) = 1;
		T_.at<double>(0, 1) = 0;
		T_.at<double>(1, 0) = 0;
		T_.at<double>(1, 1) = 1;

		//cout << cnt  << "      dx: " << dx <<  "   dy: " << dy << endl << endl;

		//T = T * 0.5;

		if (cur_corner2.size() < 10 || dx > 10 || dy > 10)
			cur2 = img;
		else
		{

			cuda::warpAffine(img, cur2, T_, img.size());

		}

		cur2 = cur2(Range(maxPixel_x, cur2.rows - maxPixel_x), Range(maxPixel_y, cur2.cols - maxPixel_y));


		cuda::resize(cur2, cur2, img.size());

		prevImgCuda = img.clone();
		cur_grey.copyTo(prev_grey);

		Mat gray;
		prevImgCuda.download(gray);
		resize(gray, gray, Size(img.cols / resize_factor, img.rows / resize_factor));
		drawArrows_new(gray, prev_corner2, cur_corner2, status, inlier, Scalar(50, 200, 255));
		draw.upload(gray);

		k++;
		cnt++;
	}
	preimg = img.clone();

}




static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color)
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
			

			// Here we lengthen the arrow by a factor of three.
			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			// Now we draw the main line of the arrow.
			if (inlier[i] == 1)
			{  
				line(frame, p, q, line_color, line_thickness);

				// Now draw the tips of the arrow. I do some scaling so that the
				// tips look proportional to the main line of the arrow.

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

				// Now draw the tips of the arrow. I do some scaling so that the
				// tips look proportional to the main line of the arrow.

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





static void drawArrows_new(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color)
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

		


			// Here we lengthen the arrow by a factor of three.
			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			// Now we draw the main line of the arrow.
			if (inlier[i] == 1)
			{
				if (hypotenuse < 1.0) {
					circle(frame, p, 2, Scalar(50, 200, 40), 2, 8);
					green++;
					continue;
				}

				else {
					line(frame, p, q, line_color, line_thickness);

					// Now draw the tips of the arrow. I do some scaling so that the
					// tips look proportional to the main line of the arrow.

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

				// Now draw the tips of the arrow. I do some scaling so that the
				// tips look proportional to the main line of the arrow.

				p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);

				p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
				p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
				line(frame, p, q, Scalar(20, 20, 200), line_thickness - 1);
			}

		}
	}
	//cout << "nextPts size: " << nextPts.size() << "	  yellow : " << yellow << "		green : " << green << "		 //		green percent : " << ((double)green / nextPts.size()) * 100 << "		yellow percent : " << ((double)yellow / nextPts.size()) * 100 << endl;
}