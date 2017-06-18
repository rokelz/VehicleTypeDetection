#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame );
String vehicle_identifier_name = "cascade.xml";
//String nonvehicle_identifier_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier vehicle_identifier;
//CascadeClassifier nonvehicle_identifier;
String window_name = "Capture - Vehicle detection";
int frameCount;
int main( void )
{
    
    //Mat frame;
    //char fileName[100] = "/home/ruwiniopatha/Desktop/Milestone4/CVBD/HaarsCascade/clip.mp4"; //Gate1_175_p1.avi"; //mm2.avi"; //";//_p1.avi";
    //VideoCapture stream1(fileName);   //0 is the id of video device.0 if you have only one camera   
    if( !vehicle_identifier.load( vehicle_identifier_name ) ){ printf("--(!)Error loading vehicels cascade\n"); return -1; };
        frameCount = 0;
    Mat img1 = imread( "/home/ruwiniopatha/Desktop/Milestone4/CVBD/HaarsCascade/image1.jpg");
    detectAndDisplay( img1 );
 //unconditional loop  

//    while (true) {   
  //  Mat cameraFrame;   
    //if(!(stream1.read(frame))) //get one frame form video   
     //break;
    //else{
	//detectAndDisplay( frame );
        //char c = (char)waitKey(10);
        //if( c == 27 ) { break; }

	//}
//}
    //-- 1. Load the cascades
   
   // if( !nonvehicle_identifier.load( nonvehicle_identifier_name ) ){ printf("--(!)Error loading non vehicels cascade\n"); return -1; };
    //-- 2. Read the video stream
   // capture.open( -1 );
    //if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
   	
    /*while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        char c = (char)waitKey(10);
        if( c == 27 ) { break; } // escape
    }*/
	
    return 0;
}
void detectAndDisplay( Mat frame )
{
    int detectedVehicles=0,detectedNonVehicles=0;	
    std::vector<Rect> vehicles,nonvehicels;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect vehicles
    vehicle_identifier.detectMultiScale( frame_gray, vehicles, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    //nonvehicle_identifier.detectMultiScale(frame_gray,nonvehicles,1.1,2,0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( size_t i = 0; i < vehicles.size(); i++ )
    {
        Point center( vehicles[i].x + vehicles[i].width/2, vehicles[i].y + vehicles[i].height/2 );
        ellipse( frame, center, Size( vehicles[i].width/2, vehicles[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
	detectedVehicles++;    
	imwrite("abc.jpg",frame);
	//Mat vehiclesROI = frame_gray( vehicles[i] );
       // std::vector<Rect> nonvehicles;
        //-- In each face, detect eyes
        //nonvehicle_identifier.detectMultiScale( vehiclesROI, nonvehicles, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        
    }
     //-- Detect non vehicles
    /*
    for ( size_t j = 0; j < nonvehicles.size(); j++ )
        {
            Point nonvehicles_center( nonvehicles[j].x + nonvehicles[j].width/2, nonvehicles[j].y + nonvehicles[j].height/2 );
            int radius = cvRound( (nonvehicles[j].width + nonvehicles[j].height)*0.25 );
            circle( frame, nonvehicles_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
	    detectedNonVehicles++;
        }
	*/
    //cout<<"Frame number: "<<frameCount<<" detected vehicles: "<<detectedVehicles<<endl;
    //-- Show what you got

    imshow( window_name, frame );
}
