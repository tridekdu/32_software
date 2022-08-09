#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fmt/core.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "ketopt.h"

#define PI 3.14159265

// - Editable Params --------------------------------

#define EMAT_H 8
#define EMAT_W 10

typedef unsigned int uint;

static int input_width = 320;
static int input_height = 240;
static int input_fps = 60;

float x = 0;
float y = 0;
float w = input_width;
float h = input_height;
float xt = 0;
float yt = 0;
float r = 0;
float z = 0;
float k1 = 0.5f;
float k2 = 0.5f;
float k3 = 0.5f;
float p1 = 0.5f;
float p2 = 0.5f;

// --------------------------------------------------

static int proc_f = 0;
static uint32_t brmask = 0b00000000000000000000000010000111;
static uint32_t color = 0xAA6464;
static uint32_t eye_color = 0xAA6464;

int pupil_min_x = input_width/2;
int pupil_min_y = input_height/2;
int pupil_max_x = input_width/2;
int pupil_max_y = input_height/2;

cv::Mat cam_input;
cv::Mat cam_processed;
cv::Mat eye_output;

cv::Size frameSize(cv::Size(input_width, input_height));
cv::Mat cameraMatrix(3, 3, cv::DataType<float>::type);
cv::Mat distCoeffs(5, 1, cv::DataType<float>::type);
cv::Mat mapx, mapy;

uint32_t flip(uint32_t v){
	v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
	v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
	v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
	v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
	v = ( v >> 16             ) | ( v               << 16);
	return v;
}

void driver_write(uint32_t num){
	char data[4] = {(num & 0xFF), ((num>>8) & 0xFF), ((num>>16) & 0xFF), ((num>>24) & 0xFF)};
	write(proc_f, data, 2);
}

void driver_init(){
	proc_f = open("/proc/ledpanels", O_RDWR);
}

std::string gstreamer_pipeline(int width, int height, int framerate) {
	return
    	" libcamerasrc ! video/x-raw, "
		" width=(int)" + std::to_string(width) + ","
		" height=(int)" + std::to_string(height) + ","
		" framerate=(fraction)" + std::to_string(framerate) +"/1 ! appsink";
}

void updateMatrices(){

	cv::Point2f center((input_width)*0.5f, (input_height)*0.5f);
    cv::Mat newRotMat = cv::getRotationMatrix2D(center, r, 1.0 + z);

	float rr = r*PI/180.0f;
	float a = cos(rr);
	float b = sin(rr);
	float cx = input_width * 0.5f;
	float cy = input_height * 0.5f;

	cameraMatrix.at<float>(0, 0) = input_width * newRotMat.at<double>(0,0);
	cameraMatrix.at<float>(0, 1) = input_width * newRotMat.at<double>(0,1);
	cameraMatrix.at<float>(0, 2) = input_width * 0.5f * z;

 	cameraMatrix.at<float>(1, 0) = input_height * newRotMat.at<double>(1,0);
	cameraMatrix.at<float>(1, 1) = input_height * newRotMat.at<double>(1,1);
	cameraMatrix.at<float>(1, 2) = input_height * 0.5f * z;

	cameraMatrix.at<float>(2, 0) = 0.0f;
	cameraMatrix.at<float>(2, 1) = 0.0f;
	cameraMatrix.at<float>(2, 2) = 1.0f;

    distCoeffs = (cv::Mat1d(1, 5) << k1, k2, p1, p2, k3);

//  cv::Mat newCamMat = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, frameSize, 1, frameSize, 0);
//	cameraMatrix.at<float>(0, 0) *= newRotMat.at<float>(0, 0);
//	cameraMatrix.at<float>(0, 1) *= newRotMat.at<float>(0, 1);
//	cameraMatrix.at<float>(0, 2) = input_width*0.5f;
// 	cameraMatrix.at<float>(1, 0) = newRotMat.at<float>(1, 0);
//	cameraMatrix.at<float>(1, 1) = newRotMat.at<float>(1, 1);
//	cameraMatrix.at<float>(1, 2) = input_height*0.5f;
//	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCamMat, frameSize, CV_16SC2, mapx, mapy);

	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, frameSize, CV_16SC2, mapx, mapy);

}

// - OpenCV GUI Things ------------------------------

void callback_ROI_X( int i, void* ) { x = i; updateMatrices(); }
void callback_ROI_Y( int i, void* ) { y = i; updateMatrices(); }
void callback_ROI_W( int i, void* ) { w = i; updateMatrices(); }
void callback_ROI_H( int i, void* ) { h = i; updateMatrices(); }
void callback_ROI_R( int i, void* ) { r = (float)i; updateMatrices(); }
void callback_ROI_Z( int i, void* ) { z = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_K1( int i, void* ) { k1 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_K2( int i, void* ) { k2 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_K3( int i, void* ) { k3 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_P1( int i, void* ) { p1 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_P2( int i, void* ) { p2 = i/1000.0f - .5f; updateMatrices(); }

void callback( int i, void* ) {}

void makeGUI(){
    cv::namedWindow( "adjustments", cv::WINDOW_AUTOSIZE );
	cv::createTrackbar("ROI X", "adjustments", NULL, input_width, callback_ROI_X);
	cv::createTrackbar("ROI Y", "adjustments", NULL, input_height, callback_ROI_Y);
	cv::createTrackbar("ROI W", "adjustments", NULL, input_width, callback_ROI_W);
	cv::setTrackbarPos("ROI W", "adjustments", input_width);
	cv::createTrackbar("ROI H", "adjustments", NULL, input_height, callback_ROI_H);
	cv::setTrackbarPos("ROI H", "adjustments", input_height);
	cv::createTrackbar("r", "adjustments", NULL, 360, callback_ROI_R);
	cv::createTrackbar("z", "adjustments", NULL, 1000, callback_ROI_Z);
	cv::setTrackbarPos("z", "adjustments", 500);
	cv::createTrackbar("k1", "adjustments", NULL, 1000, callback_CAM_K1);
	cv::setTrackbarPos("k1", "adjustments", 500);
	cv::createTrackbar("k2", "adjustments", NULL, 1000, callback_CAM_K2);
	cv::setTrackbarPos("k2", "adjustments", 500);
	cv::createTrackbar("k3", "adjustments", NULL, 1000, callback_CAM_K3);
	cv::setTrackbarPos("k3", "adjustments", 500);
	cv::createTrackbar("p1", "adjustments", NULL, 1000, callback_CAM_P1);
	cv::setTrackbarPos("p1", "adjustments", 500);
	cv::createTrackbar("p2", "adjustments", NULL, 1000, callback_CAM_P2);
	cv::setTrackbarPos("p2", "adjustments", 500);
    cv::resizeWindow("adjustments", 240, 480);
    cv::moveWindow("adjustments", 240, 0);
}
// --------------------------------------------------

int main(int argc, char *argv[]) {

	static ko_longopt_t longopts[] = {
		{ "gst", ko_required_argument, 301 },
		{ NULL, 0, 0 }
	};

	std::string pipeline = gstreamer_pipeline(input_width, input_height, input_fps);

	ketopt_t opt = KETOPT_INIT;
	int i, c;
	while ((c = ketopt(&opt, argc, argv, 1, "xy:", longopts)) >= 0) {
		if (c == 301){
			pipeline = opt.arg;
		} else if (c == '?') printf("unknown opt: -%c\n", opt.opt? opt.opt : ':');
		else if (c == ':') printf("missing arg: -%c\n", opt.opt? opt.opt : ':');
	}

    std::cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }

	updateMatrices();
	driver_init();
	makeGUI();

	while(cap.isOpened()){

    	if (!cap.read(cam_input)) {
            std::cout << "Capture read error" << std::endl;
            continue;
        }

		//cv::cvtColor(cam_input, cam_input, cv::COLOR_BGR2GRAY);

		cv::remap(cam_input, cam_processed, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

		

		cv::resize(cam_processed, eye_output, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);

		cv::imshow("adjustments", eye_output);
		cv::imshow("cam_processed", cam_processed);

		uint8_t* pp = (uint8_t*)eye_output.data;

		driver_write(0b00000000000000000000000000000000);

		for(int e=0; e < 2; e++)
			for(int y=0; y < EMAT_H; y++) {
				for(int x=0; x < EMAT_W; x++) {
					uint dx = ((y < EMAT_H ? y % 2 == 0 : y % 2 != 0 ) ? x : (EMAT_W-1)-x);
					uint dy = (y < EMAT_H ? 7-y%EMAT_H : y%EMAT_H );
                    if (e) {
						dx = (EMAT_W - dx - 1) % EMAT_W;
						dy = (EMAT_H - dy - 1) % EMAT_H;
					}
					uint32_t color = ( (pp[dy*EMAT_W*3 + dx*3 + 0]/2) << 16) | ((pp[dy*EMAT_W*3 + dx*3 + 1]/2) << 8) | (pp[dy*EMAT_W*3 + dx*3 + 2]/2);
					driver_write(brmask | (flip(color)));
				}
			}

		driver_write(0b11111111111111111111111111111111);
		usleep(1000);

        if (cv::waitKey(10) == 27) { break; }
	}

    cap.release();

}
