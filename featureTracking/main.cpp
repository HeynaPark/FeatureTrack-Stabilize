#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"


#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


using namespace std;
using namespace cv;

const int HORIZTAL_BORDER_CROP = 20;

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


void stab_live(cuda::GpuMat prev, cuda::GpuMat& cur);
void stab_live_size(cuda::GpuMat prev, cuda::GpuMat& cur, cuda::GpuMat& draw);
void stab_live_frame(cuda::GpuMat prev, cuda::GpuMat& cur, cuda::GpuMat& draw);
void stab_live_2(cuda::GpuMat prev, cuda::GpuMat& cur, cuda::GpuMat& draw);
static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color);


int first = 1;

cuda::GpuMat prevImgCuda;
//cuda::GpuMat prev_gpu, cur_gpu;




int main(int argc, char ** argv)
{
        
        
  

    for (int i = 0; i < 1; i++) {
        //cuda::GpuMat img;
        cuda::GpuMat prev_gpu, cur_gpu, cur_gpu2;
        Mat img;
        first = 1;
     
        //VideoCapture cap("C:/Users/4Dreplay/Downloads/003060_join3.mp4");
        VideoCapture cap("D:/test/stabil/ncaa1.mp4");
        int fps = (int)cap.get(CAP_PROP_FPS);
        VideoWriter src_output("ncaa1-every frame update(src).avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(1920/2, 1080/2));
        VideoWriter dst_output("ncaa1-every frame update(dst).avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(1920 , 1080));
        
       
        if (!cap.isOpened()) {
            printf("Can't open the file.");
            return -1;
        }

       
        Mat dst;
        Mat src;
        Mat dst2;

        while (1) {
           
         
            
            cap >> img;

            if (img.empty()) {
                printf("img empty.");
                break;
            }

            prev_gpu.upload(img);
            stab_live_size(prev_gpu, cur_gpu, cur_gpu2);
          //  stab_live_frame(prev_gpu, cur_gpu, cur_gpu2);
           // prev_gpu.download(src);
            if (first)
            {
                prev_gpu.download(dst);
                prev_gpu.download(src);
                first = false;
            }
            else
            {
                cur_gpu.download(dst);
                cur_gpu2.download(src);
            }

            imshow("src", src);
            imshow("dst", dst);

         
            
            if(src.cols==1920/2)
                src_output.write(src);
            dst_output.write(dst);
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

//cuda::GpuMat last_T;
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
double vert_border = 0;
double horz_border = 0;
bool _first = false;
double a = 0, x = 0, y = 0;
int k = 1;
vector <Trajectory> trajectory;
vector <TransformParam> prev_to_cur_transform;
vector <Trajectory> smoothed_trajectory;
vector <TransformParam> new_prev_to_cur_transform;
Trajectory Q;
Trajectory R;
double CROP_PER = 0.01;
int frame_cnt = 0;
//cuda::GpuMat last_T;
cuda::GpuMat preimg;
cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
vector<Point2f> prev_corner, cur_corner;
vector<Point2f> prev_corner2, cur_corner2;


cuda::GpuMat gpuErr;

vector <float> err;



void stab_live(cuda::GpuMat img, cuda::GpuMat& cur2)
{
    Mat T;
 
    //Mat T(2, 3, CV_64F);

    
    if (!_first) {
       prevImgCuda = img.clone();
       cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);

       _first = true;
       double pstd = 4e-3;
       double cstd =0.25;
       Q.x = pstd;
       Q.y = pstd;
       Q.a = pstd;
       R.x = cstd;
       R.y = cstd;
       R.a = cstd;
       
       vert_border = HORIZTAL_BORDER_CROP * prevImgCuda.rows / prevImgCuda.cols;

       detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30);
       pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
       
     
       //return 0;
    }

    else {
        cuda::cvtColor(img, cur_grey, COLOR_BGR2GRAY);


        cuda::GpuMat prev_corner, cur_corner;
        cuda::GpuMat prev_corner2, cur_corner2;

        cuda::GpuMat gpuStatus;
        vector <uchar> status;
        vector <float> err;



        detector->detect(prev_grey, prev_corner);
        pryLK_sparse->calc(prev_grey, cur_grey, prev_corner, cur_corner, gpuStatus);


        vector<Point2f> prevPts(prev_corner.cols);
        prev_corner.download(prevPts);

        vector<Point2f> curPts(cur_corner.cols);
        cur_corner.download(curPts);

        vector<uchar> stat(gpuStatus.cols);
        gpuStatus.download(stat);



        Mat prev_corner_c, cur_corner_c;
        prev_corner.download(prev_corner_c);
        cur_corner.download(cur_corner_c);

        T = estimateAffine2D(prev_corner_c, cur_corner_c);
        cout << "normal T " << T << endl;

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

        double diff_x = X.x - x;
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = dx + diff_x;
        dy = dy + diff_y;
        da = da + diff_a;

        T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = -sin(da);
        T.at<double>(1, 0) = sin(da);
        T.at<double>(1, 1) = cos(da);

        T.at<double>(0, 2) = dx;
        T.at<double>(1, 2) = dy;

        //   cuda::GpuMat cur2;
           //Mat cur2;

        cuda::warpAffine(prevImgCuda, cur2, T, img.size());
        //warpAffine(prev, cur2, T, cur.size());

        cur2 = cur2(Range(vert_border, cur2.rows - vert_border), Range(HORIZTAL_BORDER_CROP, cur2.cols - HORIZTAL_BORDER_CROP));

        cuda::resize(cur2, cur2, img.size());



        prevImgCuda = img.clone();
        cur_grey.copyTo(prev_grey);

        Mat gray;
        img.download(gray);
      //  drawArrows(gray, prevPts, curPts, stat, Scalar(50, 200, 255));
        img.upload(gray);


        k++;

    }

}

Mat mask_cpu = Mat::zeros(1080 / 2, 1920 / 2, CV_8UC1);
cuda::GpuMat mask;

void stab_live_size(cuda::GpuMat img, cuda::GpuMat& cur2, cuda::GpuMat& draw)
{
   // Mat T;

    Mat T(2, 3, CV_64F);


    if (!_first) {
       // prevImgCuda = img.clone();
        cuda::resize(img, prevImgCuda, Size(img.cols / 2, img.rows / 2));
        cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);

        _first = true;
        double pstd = 4e-3;
        double cstd = 0.25;
        Q.x = pstd;
        Q.y = pstd;
        Q.a = pstd;
        R.x = cstd;
        R.y = cstd;
        R.a = cstd;

        //vert_border = HORIZTAL_BORDER_CROP * prevImgCuda.rows / prevImgCuda.cols;
        vert_border = prevImgCuda.rows * CROP_PER;
        horz_border = prevImgCuda.cols * CROP_PER;

        detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30);
        //detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30,3,true,0.02);
        pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
       // Ptr<cuda::SparseOpticalFlow> sparse = cuda::SparseOpticalFlow::Algorithm();

        //return 0;
    }

    else {
        cuda::cvtColor(img, cur_grey, COLOR_BGR2GRAY);
        cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);
        cuda::resize(prev_grey, prev_grey, Size(img.cols / 2, img.rows / 2));
        cuda::resize(cur_grey, cur_grey, Size(img.cols / 2, img.rows / 2));


        cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
        vector<Point2f> prev_corner, cur_corner;
        vector<Point2f> prev_corner2, cur_corner2;
        //vector<cuda::GpuMat> prev_corner2, cur_corner2;

        cuda::GpuMat gpuStatus;
        cuda::GpuMat gpuErr;
        vector <uchar> status;
        vector <float> err;

        //mask
     //   mask_cpu(Rect(50, 50, prev_grey.cols - 100, prev_grey.rows - 100)) = 1;
        // imshow("mask", mask_cpu); 
         //waitKey(1);
      //  mask.upload(mask_cpu);


        detector->detect(prev_grey, prev_corner_gpu);
        //detector->detect(prev_grey, prev_corner_gpu,mask);
        pryLK_sparse->calc(prev_grey, cur_grey, prev_corner_gpu, cur_corner_gpu, gpuStatus, gpuErr);



        prev_corner_gpu.download(prev_corner);
        cur_corner_gpu.download(cur_corner);
        gpuStatus.download(status);
        gpuErr.download(err);


        for (size_t i = 0; i < prev_corner.size(); i++) {
            //if (status[i]   && err[i] < 5) {
            if (status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }

        /* vector<Point2f> prevPts;
         prev_corner2.download(prevPts);

         vector<Point2f> curPts(cur_corner.size());
         cur_corner.download(curPts);*/

         /* vector<uchar> stat(gpuStatus.cols);
          gpuStatus.download(stat);*/



          /*    Mat prev_corner_c, cur_corner_c;
              prev_corner2.download(prev_corner_c);
              cur_corner2.download(cur_corner_c);*/

              //   T = estimateAffine2D(prev_corner_c, cur_corner_c);
       // Mat inlier;
        vector<uchar> inlier;
        //
       // T = estimateAffine2D(prev_corner2, cur_corner2, inlier, RANSAC, 3);
      
        T = estimateAffinePartial2D(prev_corner2, cur_corner2,inlier,RANSAC,3); //similar to rigidtransform.
       // T = estimateAffinePartial2D( cur_corner2, prev_corner2, inlier, RANSAC, 3); //similar to rigidtransform.
        
        T = T*2;    //corner size 1/2


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

        double diff_x = X.x - x;
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = dx + diff_x;
        dy = dy + diff_y;
        da = da + diff_a;

     /*   T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = -sin(da);
        T.at<double>(1, 0) = sin(da);
        T.at<double>(1, 1) = cos(da);*/

        T.at<double>(0, 0) = 1;
        T.at<double>(0, 1) = 0;
        T.at<double>(1, 0) = 0;
        T.at<double>(1, 1) = 1;


        T.at<double>(0, 2) = dx;
        T.at<double>(1, 2) = dy;

        cout << "dx: " << dx << endl;
        cout << "dy: " << dy << endl;

        //   cuda::GpuMat cur2;
           //Mat cur2;

        if (cur_corner2.size() < 10 || dx > 10 || dy > 10)
            cur2 = img;

        //if (cur_corner2.size() > 10)
        else
        {
         //   cuda::warpAffine(img, cur2, T, img.size());
           cuda::warpAffine(preimg, cur2, T, img.size());
            //warpAffine(prev, cur2, T, cur.size());
            
        }
        cur2 = cur2(Range(vert_border, cur2.rows - vert_border), Range(horz_border, cur2.cols - horz_border));


        cuda::resize(cur2, cur2, img.size());
    

        
        prevImgCuda = img.clone();
        cur_grey.copyTo(prev_grey);

        Mat gray;
        prevImgCuda.download(gray);
        resize(gray, gray, Size(img.cols / 2, img.rows / 2));
      //  drawArrows(gray, prevPts, curPts, stat, Scalar(50, 200, 255));
        drawArrows(gray,prev_corner2 , cur_corner2, status, inlier, Scalar(50, 200, 255));
        draw.upload(gray);


        k++;

    }
    preimg = img.clone();

}



void stab_live_frame(cuda::GpuMat img, cuda::GpuMat& cur2, cuda::GpuMat& draw)
{
    Mat T(2, 3, CV_64F);


    if (!_first) {
        // prevImgCuda = img.clone();
        cuda::resize(img, prevImgCuda, Size(img.cols / 2, img.rows / 2));
        cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);

        _first = true;
        double pstd = 4e-3;
        double cstd = 0.25;
        Q.x = pstd;
        Q.y = pstd;
        Q.a = pstd;
        R.x = cstd;
        R.y = cstd;
        R.a = cstd;

   
        vert_border = prevImgCuda.rows * CROP_PER;
        horz_border = prevImgCuda.cols * CROP_PER;

        detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30);
        pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
  

         //return 0;
    }

    else {
        if (frame_cnt % 30 == 0)
        {

            cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);
            cuda::resize(prev_grey, prev_grey, Size(img.cols / 2, img.rows / 2));
        }
        cuda::cvtColor(img, cur_grey, COLOR_BGR2GRAY);
        cuda::resize(cur_grey, cur_grey, Size(img.cols / 2, img.rows / 2));


      //  vector<Point2f> prev_corner, cur_corner;
       // vector<Point2f> prev_corner2, cur_corner2;
     


        cuda::GpuMat gpuStatus;
        cuda::GpuMat gpuErr;
        vector <uchar> status;
        vector <float> err;

        if (frame_cnt % 30 == 0)
        {

            detector->detect(prev_grey, prev_corner_gpu);
            prev_corner_gpu.download(prev_corner);
        }//
        pryLK_sparse->calc(prev_grey, cur_grey, prev_corner_gpu, cur_corner_gpu, gpuStatus, gpuErr);



      
        cur_corner_gpu.download(cur_corner);
        gpuStatus.download(status);
        gpuErr.download(err);

        cur_corner2.clear();
        prev_corner2.clear();
        for (size_t i = 0; i < prev_corner.size(); i++) {
            //if (status[i]   && err[i] < 5) {
            if (status[i]) {
                //if (frame_cnt % 5 == 0)
                prev_corner2.push_back(prev_corner[i]);
                //vector<Point2f> cur_corner2;
               
                cur_corner2.push_back(cur_corner[i]);
            }
        }

        vector<uchar> inlier;

       // T = estimateAffine2D(prev_corner2, cur_corner2, inlier, RANSAC, 3);

        T = estimateAffinePartial2D(prev_corner2, cur_corner2, inlier, RANSAC, 3); //similar to rigidtransform.
 
        T = T * 2;    //corner size 1/2


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

        double diff_x = X.x - x;
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = dx + diff_x;
        dy = dy + diff_y;
        da = da + diff_a;

        /*   T.at<double>(0, 0) = cos(da);
           T.at<double>(0, 1) = -sin(da);
           T.at<double>(1, 0) = sin(da);
           T.at<double>(1, 1) = cos(da);*/

        T.at<double>(0, 0) = 1;
        T.at<double>(0, 1) = 0;
        T.at<double>(1, 0) = 0;
        T.at<double>(1, 1) = 1;


        T.at<double>(0, 2) = dx;
        T.at<double>(1, 2) = dy;

        cout << "dx: " << dx << endl;
        cout << "dy: " << dy << endl;

        //   cuda::GpuMat cur2;
           //Mat cur2;

        if (cur_corner2.size() < 10 || dx > 10 || dy > 10)
            cur2 = img;

        else
        {
            //   cuda::warpAffine(img, cur2, T, img.size());
            cuda::warpAffine(preimg, cur2, T, img.size());


        }
        cur2 = cur2(Range(vert_border, cur2.rows - vert_border), Range(horz_border, cur2.cols - horz_border));


        cuda::resize(cur2, cur2, img.size());



        prevImgCuda = img.clone();
        cur_grey.copyTo(prev_grey);

        Mat gray;
        prevImgCuda.download(gray);
        resize(gray, gray, Size(img.cols / 2, img.rows / 2));
        drawArrows(gray, prev_corner2, cur_corner2, status, inlier, Scalar(50, 200, 255));
        draw.upload(gray);


        k++;

        frame_cnt++;
    }
    preimg = img.clone();
}


void stab_live_2(cuda::GpuMat img, cuda::GpuMat& cur2, cuda::GpuMat& draw)
{
    // Mat T;

    Mat T(2, 3, CV_64F);


    if (!_first) {
        // prevImgCuda = img.clone();
        cuda::resize(img, prevImgCuda, Size(img.cols / 2, img.rows / 2));
        cuda::cvtColor(prevImgCuda, prev_grey, COLOR_BGR2GRAY);

        _first = true;
        double pstd = 4e-3;
        double cstd = 0.25;
        Q.x = pstd;
        Q.y = pstd;
        Q.a = pstd;
        R.x = cstd;
        R.y = cstd;
        R.a = cstd;

        //vert_border = HORIZTAL_BORDER_CROP * prevImgCuda.rows / prevImgCuda.cols;
        vert_border = prevImgCuda.rows * CROP_PER;
        horz_border = prevImgCuda.cols * CROP_PER;

        detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30);
        //detector = cuda::createGoodFeaturesToTrackDetector(prev_grey.type(), 200, 0.01, 30,3,true,0.02);
        pryLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
        // Ptr<cuda::SparseOpticalFlow> sparse = cuda::SparseOpticalFlow::Algorithm();

         //return 0;
    }

    else {
        cuda::cvtColor(img, cur_grey, COLOR_BGR2GRAY);
        cuda::resize(cur_grey, cur_grey, Size(img.cols / 2, img.rows / 2));


        cuda::GpuMat prev_corner_gpu, cur_corner_gpu;
        vector<Point2f> prev_corner, cur_corner;
        vector<Point2f> prev_corner2, cur_corner2;
        //vector<cuda::GpuMat> prev_corner2, cur_corner2;

        cuda::GpuMat gpuStatus;
        cuda::GpuMat gpuErr;
        vector <uchar> status;
        vector <float> err;

        //mask
        mask_cpu(Rect(50, 50, prev_grey.cols - 100, prev_grey.rows - 100)) = 1;
        // imshow("mask", mask_cpu); 
         //waitKey(1);
        mask.upload(mask_cpu);


        detector->detect(prev_grey, prev_corner_gpu);
        //detector->detect(prev_grey, prev_corner_gpu,mask);
        pryLK_sparse->calc(prev_grey, cur_grey, prev_corner_gpu, cur_corner_gpu, gpuStatus, gpuErr);



        prev_corner_gpu.download(prev_corner);
        cur_corner_gpu.download(cur_corner);
        gpuStatus.download(status);
        gpuErr.download(err);


        for (size_t i = 0; i < prev_corner.size(); i++) {
            //if (status[i]   && err[i] < 10) {
            if (status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }

  
       // Mat inlier;
        vector<uchar> inlier;
        //
        T = estimateAffine2D(prev_corner2, cur_corner2, inlier, RANSAC, 3);
        //   T = estimateRigidTransform(prev_corner2, cur_corner2);

        T = T * 2;


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

        double diff_x = X.x - x;
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = dx + diff_x;
        dy = dy + diff_y;
        da = da + diff_a;

        T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = -sin(da);
        T.at<double>(1, 0) = sin(da);
        T.at<double>(1, 1) = cos(da);



        T.at<double>(0, 2) = dx;
        T.at<double>(1, 2) = dy;
        cout << "dx: " << dx << endl;
        cout << "dy: " << dy << endl;

        //   cuda::GpuMat cur2;
           //Mat cur2;

        if (cur_corner2.size() > 10)
        {
            cuda::warpAffine(prevImgCuda, cur2, T, img.size());
            //warpAffine(prev, cur2, T, cur.size());

            cur2 = cur2(Range(vert_border, cur2.rows - vert_border), Range(horz_border, cur2.cols - horz_border));


            cuda::resize(cur2, cur2, img.size());
        }
        else
            cur2 = img;


        prevImgCuda = img.clone();
        cur_grey.copyTo(prev_grey);

        Mat gray;
        prevImgCuda.download(gray);
        resize(gray, gray, Size(img.cols / 2, img.rows / 2));
        //  drawArrows(gray, prevPts, curPts, stat, Scalar(50, 200, 255));
        drawArrows(gray, prev_corner2, cur_corner2, status, inlier, Scalar(50, 200, 255));
        draw.upload(gray);


        k++;

    }

}


static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, const vector<uchar>& inlier, Scalar line_color)
{
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
            }
            else
            {
                line(frame, p, q, Scalar(20,20,200), line_thickness-1);

                // Now draw the tips of the arrow. I do some scaling so that the
                // tips look proportional to the main line of the arrow.

                p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
                p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
                line(frame, p, q, Scalar(20, 20, 200), line_thickness-1);

                p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
                p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
                line(frame, p, q, Scalar(20, 20, 200), line_thickness-1);
            }
        }
    }
}