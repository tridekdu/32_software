#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fmt/core.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <fstream>
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

static int mode = 0;

static bool passthrough = false;

float x = 0;
float y = 0;
float w = input_width;
float h = input_height;
float r = 0;
float q = 0.5f;
float zx = 0.5f;
float zy = 0.5f;
float k1 = 0.5f;
float k2 = 0.5f;
float k3 = 0.5f;
float p1 = 0.5f;
float p2 = 0.5f;

float cut = 0.5f;
float d = 0.5f;
float e = 0.5f;
float c = 0.5f;

// --------------------------------------------------

static int proc_f = 0;
static uint32_t brmask = 0b00000000000000000000000010000111;
//static uint32_t color = 0xAA6464;
//static uint32_t eye_color = 0xAA6464;
cv::Scalar eye_color(39,39,67);

int pupil_min_x = input_width/2;
int pupil_min_y = input_height/2;
int pupil_max_x = input_width/2;
int pupil_max_y = input_height/2;

cv::Rect roi_rect(x, y, w, h);

cv::Mat cam_input;
cv::Mat cam_gray;
cv::Mat cam_remapped;
cv::Mat cam_procbuf_1;
cv::Mat cam_procbuf_2;
cv::Mat cam_procbuf_3;
cv::Mat cam_procbuf_4;
cv::Mat cam_procbuf_5;

cv::Mat pupil(64, 64, CV_8UC3);
cv::Mat eye_buffer_L(input_height, input_width, CV_32FC3, cv::Scalar(0));
cv::Mat eye_buffer_R(input_height, input_width, CV_32FC3, cv::Scalar(0));
//cv::Mat buffer_L(EMAT_W * 3, EMAT_H * 3, CV_8UC3);
//cv::Mat buffer_R(EMAT_W * 3, EMAT_H * 3, CV_8UC3);


cv::Mat eye_out_L = cv::Mat(cv::Size(EMAT_W, EMAT_H), CV_8UC3, cv::Scalar(0)); //eye_out.colRange(0, EMAT_W-1);
cv::Mat eye_out_R = cv::Mat(cv::Size(EMAT_W, EMAT_H), CV_8UC3, cv::Scalar(0)); //eye_out.colRange(EMAT_W-1, EMAT_W-1);

cv::Mat eye_out_preview;

uint8_t* eoL = (uint8_t*)eye_out_L.data;
uint8_t* eoR = (uint8_t*)eye_out_R.data;

cv::Size frameSize(cv::Size(input_width, input_height));
cv::Mat cameraMatrix(3, 3, cv::DataType<float>::type);
cv::Mat distCoeffs(5, 1, cv::DataType<float>::type);
cv::Mat mapx, mapy, dilate_element, erode_element;



// Setup SimpleBlobDetector parameters.
cv::SimpleBlobDetector::Params params;

std::vector<cv::KeyPoint> keypoints;
cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
// SimpleBlobDetector::create creates a smart pointer.
// So you need to use arrow ( ->) instead of dot ( . )
// detector->detect( im, keypoints);


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
		" framerate=(fraction)" + std::to_string(framerate) +"/1 ! videoconvert ! appsink sync=false";
}

void updateMatrices(){

	cv::Point2f center((input_width)*0.5f, (input_height)*0.5f);
    cv::Mat newRotMat = cv::getRotationMatrix2D(center, r, 1.0);

	float rr = r*PI/180.0f;
	float a = cos(rr);
	float b = sin(rr);
	float cx = input_width * 0.5f;
	float cy = input_height * 0.5f;

	cameraMatrix.at<float>(0, 0) = input_width * newRotMat.at<double>(0,0);
	cameraMatrix.at<float>(0, 1) = input_width * newRotMat.at<double>(0,1);
	cameraMatrix.at<float>(0, 2) = input_width * (1.0f - 2.0f*(0.5f-zx));

 	cameraMatrix.at<float>(1, 0) = input_height * newRotMat.at<double>(1,0);
	cameraMatrix.at<float>(1, 1) = input_height * newRotMat.at<double>(1,1);
	cameraMatrix.at<float>(1, 2) = input_height * (1.0f - 2.0f*(0.5f-zy));

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

	if(x + w >= input_width) w = input_width - x;
	if(y + h >= input_height) h = input_height - y;

	roi_rect = cv::Rect(x,y,w,h);

	int ds = (int)(d*6);
	int es = (int)(e*6);

    erode_element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 1+es*2, 1+es*2 ), cv::Point(es,es));
    dilate_element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 1+ds*2, 1+ds*2 ), cv::Point(ds,ds));

	//params.minThreshold = 10;
	//params.maxThreshold = 200;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = false;
	//params.minInertiaRatio = 0.01;
	params.filterByConvexity = false;
	//params.minConvexity = 0.87;
	params.filterByColor = false;
	params.filterByCircularity = false;
	//params.minCircularity = 0.1;
	params.filterByArea = true;
	params.minArea = c*3000.0f;
	params.maxArea = q*3000.0f;


}

// - OpenCV GUI Things ------------------------------

void callback_ROI_X( int i, void* ) { x = i; updateMatrices(); }
void callback_ROI_Y( int i, void* ) { y = i; updateMatrices(); }
void callback_ROI_W( int i, void* ) { w = i; updateMatrices(); }
void callback_ROI_H( int i, void* ) { h = i; updateMatrices(); }
void callback_ROI_R( int i, void* ) { r = (float)i; updateMatrices(); }
void callback_ROI_Q( int i, void* ) { q = i/1000.0f; updateMatrices(); }
void callback_ROI_ZX( int i, void* ) { zx = i/1000.0f - .5f; updateMatrices(); }
void callback_ROI_ZY( int i, void* ) { zy = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_K1( int i, void* ) { k1 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_K2( int i, void* ) { k2 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_K3( int i, void* ) { k3 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_P1( int i, void* ) { p1 = i/1000.0f - .5f; updateMatrices(); }
void callback_CAM_P2( int i, void* ) { p2 = i/1000.0f - .5f; updateMatrices(); }
void callback_E_CUT( int i, void* ) { cut = i/1000.0f; updateMatrices(); }
void callback_E_D( int i, void* ) { d = i/1000.0f; updateMatrices(); }
void callback_E_E( int i, void* ) { e = i/1000.0f; updateMatrices(); }
void callback_E_C( int i, void* ) { c = i/1000.0f; updateMatrices(); }


void callback( int i, void* ) {}

std::vector<int> trackbarPos = {0, 0, input_width, input_height, 180, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500};

std::string statefilePath("trackbarStates.dat");

void saveTrackbars(){
	std::cout << "Saving Trackbars" << std::endl;

	//std::ofstream file(statefilePath, std::ios::out | std::ofstream::binary);
	//std::copy(trackbarPos.begin(), trackbarPos.end(), std::ostreambuf_iterator<char>(file));
    // Write out a list to a disk file

	trackbarPos.at(0) = cv::getTrackbarPos("ROI_X", "adjustments");
	trackbarPos.at(1) = cv::getTrackbarPos("ROI_Y", "adjustments");
	trackbarPos.at(2) = cv::getTrackbarPos("ROI_W", "adjustments");
	trackbarPos.at(3) = cv::getTrackbarPos("ROI_H", "adjustments");
	trackbarPos.at(4) = cv::getTrackbarPos("r", "adjustments");
	trackbarPos.at(5) = cv::getTrackbarPos("q", "adjustments");
	trackbarPos.at(6) = cv::getTrackbarPos("zx", "adjustments");
	trackbarPos.at(7) = cv::getTrackbarPos("zy", "adjustments");
	trackbarPos.at(8) = cv::getTrackbarPos("k1", "adjustments");
	trackbarPos.at(9) = cv::getTrackbarPos("k2", "adjustments");
	trackbarPos.at(10) = cv::getTrackbarPos("k3", "adjustments");
	trackbarPos.at(11) = cv::getTrackbarPos("p1", "adjustments");
	trackbarPos.at(12) = cv::getTrackbarPos("p2", "adjustments");
	trackbarPos.at(13) = cv::getTrackbarPos("cut", "adjustments");
	trackbarPos.at(14) = cv::getTrackbarPos("d", "adjustments");
	trackbarPos.at(15) = cv::getTrackbarPos("e", "adjustments");
	trackbarPos.at(16) = cv::getTrackbarPos("c", "adjustments");

    std::ofstream os (statefilePath, std::ios::binary);
    int s = trackbarPos.size();
    os.write((const char*)&s, sizeof(s));
    os.write((const char*)&trackbarPos[0], s * sizeof(int));
    os.close();
}

void loadTrackbars(){
	std::cout << "Loadint Trackbars" << std::endl;

	//std::ifstream file(statefilePath, std::ios::in | std::ifstream::binary);
	//std::istreambuf_iterator iter(file);
	//std::copy(iter.begin(), iter.end(), std::back_inserter(trackbarPos));

    std::ifstream is(statefilePath, std::ios::binary);

	if(!is.good()) return;

    int s;
    is.read((char*)&s, sizeof(s));
    trackbarPos.resize(s);

     // Is it safe to read a whole array of structures directly into the vector?
    is.read((char*)&trackbarPos[0], s * sizeof(int));
}

void makeGUI(){

	loadTrackbars();

    cv::namedWindow( "adjustments", cv::WINDOW_AUTOSIZE );

	cv::createTrackbar("ROI_X", "adjustments", NULL, input_width, callback_ROI_X);
	cv::setTrackbarPos("ROI_X", "adjustments", trackbarPos.at(0));

	cv::createTrackbar("ROI_Y", "adjustments", NULL, input_height, callback_ROI_Y);
	cv::setTrackbarPos("ROI_Y", "adjustments", trackbarPos.at(1));

	cv::createTrackbar("ROI_W", "adjustments", NULL, input_width, callback_ROI_W);
	cv::setTrackbarPos("ROI_W", "adjustments", trackbarPos.at(2));

	cv::createTrackbar("ROI_H", "adjustments", NULL, input_height, callback_ROI_H);
	cv::setTrackbarPos("ROI_H", "adjustments", trackbarPos.at(3));

	cv::createTrackbar("r", "adjustments", NULL, 360, callback_ROI_R);
	cv::setTrackbarPos("r", "adjustments", trackbarPos.at(4));

	cv::createTrackbar("zx", "adjustments", NULL, 1000, callback_ROI_ZX);
	cv::setTrackbarPos("zx", "adjustments", trackbarPos.at(6));

	cv::createTrackbar("zy", "adjustments", NULL, 1000, callback_ROI_ZY);
	cv::setTrackbarPos("zy", "adjustments", trackbarPos.at(7));

	cv::createTrackbar("k1", "adjustments", NULL, 1000, callback_CAM_K1);
	cv::setTrackbarPos("k1", "adjustments", trackbarPos.at(8));

	cv::createTrackbar("k2", "adjustments", NULL, 1000, callback_CAM_K2);
	cv::setTrackbarPos("k2", "adjustments", trackbarPos.at(9));

	cv::createTrackbar("k3", "adjustments", NULL, 1000, callback_CAM_K3);
	cv::setTrackbarPos("k3", "adjustments", trackbarPos.at(10));

	cv::createTrackbar("p1", "adjustments", NULL, 1000, callback_CAM_P1);
	cv::setTrackbarPos("p1", "adjustments", trackbarPos.at(11));

	cv::createTrackbar("p2", "adjustments", NULL, 1000, callback_CAM_P2);
	cv::setTrackbarPos("p2", "adjustments", trackbarPos.at(12));

	cv::createTrackbar("cut", "adjustments", NULL, 1000, callback_E_CUT);
	cv::setTrackbarPos("cut", "adjustments", trackbarPos.at(13));

	cv::createTrackbar("d", "adjustments", NULL, 1000, callback_E_D);
	cv::setTrackbarPos("d", "adjustments", trackbarPos.at(14));

	cv::createTrackbar("e", "adjustments", NULL, 1000, callback_E_E);
	cv::setTrackbarPos("e", "adjustments", trackbarPos.at(15));

	cv::createTrackbar("c", "adjustments", NULL, 1000, callback_E_C);
	cv::setTrackbarPos("c", "adjustments", trackbarPos.at(16));

	cv::createTrackbar("q", "adjustments", NULL, 1000, callback_ROI_Q);
	cv::setTrackbarPos("q", "adjustments", trackbarPos.at(5));

    cv::resizeWindow("adjustments", 240, 480);
    cv::moveWindow("adjustments", 240, 0);
}
// --------------------------------------------------

int main(int argc, char *argv[]) {

	static ko_longopt_t longopts[] = {
		{ "gst", ko_required_argument, 301 },
		{ "passthrough", ko_required_argument, 302 },
		{ NULL, 0, 0 }
	};

	std::string pipeline = gstreamer_pipeline(input_width, input_height, input_fps);

	ketopt_t opt = KETOPT_INIT;
	int i, c;
	while ((c = ketopt(&opt, argc, argv, 1, "xy:", longopts)) >= 0) {
		if (c == 301){
			pipeline = opt.arg;
		} else if (c == 302){
			passthrough = true;
		} else if (c == '?') printf("unknown opt: -%c\n", opt.opt? opt.opt : ':');
		else if (c == ':') printf("missing arg: -%c\n", opt.opt? opt.opt : ':');
	}

    std::cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if(!cap.isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        return (-1);
    }

//	cv::CvMoments *moments = (cv::CvMoments*)malloc(sizeof(cv::CvMoments));

	updateMatrices();
	driver_init();
	makeGUI();

	while(cap.isOpened()){

//        std::cout << "Capture?" << std::endl;

    	if (!cap.read(cam_input)) {
            std::cout << "Capture read error" << std::endl;
            continue;
        }

//        std::cout << "Capture!" << std::endl;

		if(passthrough){

			cv::resize(cam_input, eye_out_R, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);
			cv::resize(cam_input, eye_out_L, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);

			cv::imshow("cam_input", cam_input);
			cv::hconcat(eye_out_L, eye_out_R, eye_out_preview);
			cv::imshow("adjustments", eye_out_preview);

		} else {
			cv::cvtColor(cam_input, cam_gray, cv::COLOR_BGR2GRAY);
			cv::remap(cam_gray, cam_remapped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
			cv::rectangle(cam_remapped, roi_rect, cv::Scalar(255));
//		cv::cvtColor(cam_input(roi_rect), cam_procbuf_1, cv::COLOR_BGR2GRAY );

			cam_procbuf_1 = cam_remapped(roi_rect);


/*
        cv::cvMoments(cam_proc_gray, moments, 1);

        double moment10 = cv::getSpatialMoment(moments, 1, 0);
        double moment01 = cv::getSpatialMoment(moments, 0, 1);
        double area = cv::getCentralMoment(moments, 0, 0);

        static int posX = moment10/area;
        static int posY = moment01/area;
*/

			cv::threshold(cam_procbuf_1, cam_procbuf_2, (int)(cut*254), 255, cv::THRESH_BINARY);
//		cv::Canny(cam_procbuf_1, cam_procbuf_2, (int)(cut*254), 254, 3);

//		cv::erode(cam_procbuf_2, cam_procbuf_3, erode_element, cv::Point(-1,-1), (int)(e*10));
//		cv::dilate(cam_procbuf_3, cam_procbuf_5, dilate_element, cv::Point(-1,-1), (int)(d*10));

			cv::bitwise_not(cam_procbuf_2, cam_procbuf_3);
			cv::erode(cam_procbuf_3, cam_procbuf_4, erode_element);

			cv::dilate(cam_procbuf_4, cam_procbuf_5, dilate_element);

//		cv::accumulateWeighted(cam_procbuf_5, eye_buffer_R, 0.2);
//		cv::accumulateWeighted(cam_procbuf_5, eye_buffer_L, 0.2);

		// Detect blobs.
			detector->detect( cam_procbuf_5, keypoints );
			//cv::drawKeypoints( cam_procbuf_5, keypoints, eye_buffer_R, eye_color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			//cv::drawKeypoints( cam_procbuf_5, keypoints, eye_buffer_L, eye_color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

			cv::resize(eye_buffer_R, eye_out_R, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);
			cv::resize(eye_buffer_L, eye_out_L, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);

			cv::imshow("cam_buf", cam_procbuf_5);
			cv::imshow("adjustments", eye_out_R);
			cv::imshow("cam_input", cam_remapped);
			cv::imshow("eyebuffer", eye_buffer_R);

		}

		//Make eye-images
//		eye_out_L = cv::Mat::zeros(cv::Size(EMAT_W,EMAT_H), CV_8UC1);
//		eye_out_R = cv::Mat::zeros(cv::Size(EMAT_W,EMAT_H), CV_8UC1);


//		eye_out_L.copyTo(eye_out_view);
//		eye_out_R.copyTo( eye_out_view( cv::Rect(0,EMAT_W-1,) ) );
//		cv::imshow("eye_output_L", eye_out_L);
//		cv::imshow("eye_output_R", eye_out_R);


		//- PANEL OUTPUT -----------------------------------------------------------------/
		driver_write(0b00000000000000000000000000000000);

		for(int y=0; y < EMAT_H; y++) {
			for(int x=0; x < EMAT_W; x++) {

				uint dx = ((y < EMAT_H ? y % 2 == 0 : y % 2 != 0 ) ? x : (EMAT_W-1)-x);
				uint dy = 7-y%EMAT_H;

				uint32_t color = ((eoR[dy*EMAT_W*3 + dx*3 + 0]/2) << 16) |
								 ((eoR[dy*EMAT_W*3 + dx*3 + 1]/2) << 8) |
								  (eoR[dy*EMAT_W*3 + dx*3 + 2]/2);

				driver_write(brmask | (flip(color)));
			}
		}

		for(int y=0; y < EMAT_H; y++) {
			for(int x=0; x < EMAT_W; x++) {

				uint dx = (y % 2 != 0 ? x : (EMAT_W-1)-x);
				uint dy = y%EMAT_H ;

				uint32_t color = ((eoL[dy*EMAT_W*3 + dx*3 + 0]/2) << 16) |
								 ((eoL[dy*EMAT_W*3 + dx*3 + 1]/2) << 8) |
								  (eoL[dy*EMAT_W*3 + dx*3 + 2]/2);

				driver_write(brmask | (flip(color)));
			}
		}

		driver_write(0b11111111111111111111111111111111);
		usleep(10000);
		//--------------------------------------------------------------------------------*/

		int key = cv::waitKey(16);
		if(key == 27 || key == 81) break;
		switch(key){
			case 115: saveTrackbars(); break;
			case 109:
				mode++;
				if(mode > 5) mode = 0;
				break;
			default: break;
		}
	}

    cap.release();

}
