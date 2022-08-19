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

static int mode = 0;

float x = 0;
float y = 0;
float w = input_width;
float h = input_height;
float r = 0;
float z = 0.5f;
float k1 = 0.5f;
float k2 = 0.5f;
float k3 = 0.5f;
float p1 = 0.5f;
float p2 = 0.5f;

float cut = 0.5f;
float d = 0.5f;
float e = 0.5f;

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
cv::Mat cam_procbuf_1;
cv::Mat cam_procbuf_2;
cv::Mat cam_procbuf_3;
cv::Mat cam_procbuf_4;
cv::Mat cam_procbuf_5;

cv::Mat pupil(64, 64, CV_8UC3);
cv::Mat eye_buffer_L(input_width, input_height, CV_32F, cv::Scalar(0));
cv::Mat eye_buffer_R(input_width, input_height, CV_32F, cv::Scalar(0));
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
cv::Mat mapx, mapy, dilate_element;


// Setup SimpleBlobDetector parameters.
cv::SimpleBlobDetector::Params params;
// Change thresholds
//params.minThreshold = 10;
//params.maxThreshold = 200;
// Filter by Area.
//params.filterByArea = true;
//params.minArea = 1500;
// Filter by Circularity
//params.filterByCircularity = true;
//params.minCircularity = 0.1;
// Filter by Convexity
//params.filterByConvexity = true;
//params.minConvexity = 0.87;
// Filter by Inertia
//params.filterByInertia = true;
//params.minInertiaRatio = 0.01;

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
	cameraMatrix.at<float>(0, 2) = input_width * (0.75f + z);

 	cameraMatrix.at<float>(1, 0) = input_height * newRotMat.at<double>(1,0);
	cameraMatrix.at<float>(1, 1) = input_height * newRotMat.at<double>(1,1);
	cameraMatrix.at<float>(1, 2) = input_height * (0.75f + z);

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

    dilate_element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( (int)(10.0f*d) + 1, (int)(10.0f*d) + 1 ), cv::Point( (int)(10.0f*d), (int)(10.0f*d)) );

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
void callback_E_CUT( int i, void* ) { cut = i/1000.0f - .5f; updateMatrices(); }
void callback_E_D( int i, void* ) { d = i/1000.0f - .5f; updateMatrices(); }
void callback_E_E( int i, void* ) { e = i/1000.0f - .5f; updateMatrices(); }


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

	cv::createTrackbar("p2", "adjustments", NULL, 1000, callback_CAM_P2);
	cv::setTrackbarPos("p2", "adjustments", 500);

	cv::createTrackbar("cut", "adjustments", NULL, 1000, callback_E_CUT);
	cv::setTrackbarPos("cut", "adjustments", 500);

	cv::createTrackbar("d", "adjustments", NULL, 1000, callback_E_D);
	cv::setTrackbarPos("d", "adjustments", 500);

	cv::createTrackbar("e", "adjustments", NULL, 1000, callback_E_E);
	cv::setTrackbarPos("e", "adjustments", 500);


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

//	cv::CvMoments *moments = (cv::CvMoments*)malloc(sizeof(cv::CvMoments));

	updateMatrices();
	driver_init();
	makeGUI();

	while(cap.isOpened()){

    	if (!cap.read(cam_input)) {
            std::cout << "Capture read error" << std::endl;
            continue;
        }

		cv::cvtColor(cam_input, cam_procbuf_1, cv::COLOR_BGR2GRAY);
		cv::rectangle(cam_input, roi_rect, cv::Scalar(255, 0, 255));

		cv::remap(cam_procbuf_1, cam_procbuf_2, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
//		cv::cvtColor(cam_input(roi_rect), cam_procbuf_1, cv::COLOR_BGR2GRAY );

		cam_procbuf_2 = cam_procbuf_2(roi_rect);


		// Detect blobs.
		//detector->detect( cam_processed, keypoints );
		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
		//cv::drawKeypoints( cam_processed, keypoints, cam_processed, eye_color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


/*
        cv::cvMoments(cam_proc_gray, moments, 1);

        double moment10 = cv::getSpatialMoment(moments, 1, 0);
        double moment01 = cv::getSpatialMoment(moments, 0, 1);
        double area = cv::getCentralMoment(moments, 0, 0);

        static int posX = moment10/area;
        static int posY = moment01/area;
*/

		cv::threshold(cam_procbuf_2, cam_procbuf_5, (int)(e*254), 255, cv::THRESH_BINARY_INV);
//		cv::Canny(cam_procbuf_1, cam_procbuf_5, (int)(cut*254), 254, 3);
//		cv::dilate(cam_procbuf_2, cam_procbuf_5, dilate_element);

//		cv::accumulateWeighted(cam_procbuf_1, eye_buffer_R, 0.2);

		cv::resize(eye_buffer_R, eye_out_R, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);
		cv::resize(eye_buffer_L, eye_out_L, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);


/*
		switch(mode){
			default:
			case 0:
				cv::resize(cam_processed, eye_out_tmp, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);
				//cv::resize(cam_processed, eye_out_R, cv::Size(EMAT_W, EMAT_H), 0, 0, cv::INTER_NEAREST);
				break;

		}
*/

		//Make eye-images
//		eye_out_L = cv::Mat::zeros(cv::Size(EMAT_W,EMAT_H), CV_8UC1);
//		eye_out_R = cv::Mat::zeros(cv::Size(EMAT_W,EMAT_H), CV_8UC1);


//		eye_out_L.copyTo(eye_out_view);
//		eye_out_R.copyTo( eye_out_view( cv::Rect(0,EMAT_W-1,) ) );
//		cv::imshow("eye_output_L", eye_out_L);
//		cv::imshow("eye_output_R", eye_out_R);

		cv::hconcat(eye_out_L, eye_out_R, eye_out_preview);
		cv::imshow("adjustments", eye_out_preview);
		cv::imshow("cam_input", cam_input);
		cv::imshow("cam_buf", cam_procbuf_5);

		//- PANEL OUTPUT -----------------------------------------------------------------
		/*
		driver_write(0b00000000000000000000000000000000);

		for(int y=0; y < EMAT_H; y++) {
			for(int x=0; x < EMAT_W; x++) {
				uint dx = ((y < EMAT_H ? y % 2 == 0 : y % 2 != 0 ) ? x : (EMAT_W-1)-x);
				uint dy = (y < EMAT_H ? 7-y%EMAT_H : y%EMAT_H );
				uint32_t color = ( (eoR[dy*EMAT_W*3 + dx*3 + 0]/2) << 16) | ((eoR[dy*EMAT_W*3 + dx*3 + 1]/2) << 8) | (eoR[dy*EMAT_W*3 + dx*3 + 2]/2);
				driver_write(brmask | (flip(color)));
			}
		}

		for(int y=0; y < EMAT_H; y++) {
			for(int x=0; x < EMAT_W; x++) {
				uint dx = ((y < EMAT_H ? y % 2 == 0 : y % 2 != 0 ) ? x : (EMAT_W-1)-x);
				uint dy = (y < EMAT_H ? 7-y%EMAT_H : y%EMAT_H );
				uint32_t color = ( (eoL[dy*EMAT_W*3 + dx*3 + 0]/2) << 16) | ((eoL[dy*EMAT_W*3 + dx*3 + 1]/2) << 8) | (eoL[dy*EMAT_W*3 + dx*3 + 2]/2);
				driver_write(brmask | (flip(color)));
			}
		}

		driver_write(0b11111111111111111111111111111111);
		usleep(1000);
		//--------------------------------------------------------------------------------*/
		int key = cv::waitKey(10);
		if(key == 27 || key == 81) break;
		switch(key){
			case 109:
				mode++;
				if(mode > 5) mode = 0;
				break;
			default: break;
		}
	}

    cap.release();

}
