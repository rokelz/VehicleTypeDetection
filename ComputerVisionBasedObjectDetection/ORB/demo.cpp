#include<opencv2/features2d.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>      //for imshow
#include<vector>
#include<iostream>
#include<iomanip>
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    Mat gray_image1 = imread( "/home/ruwiniopatha/Desktop/Milestone4/CVBD/ORB/image2.jpg", IMREAD_GRAYSCALE );
    Mat gray_image2 = imread( "/home/ruwiniopatha/Desktop/Milestone4/CVBD/ORB/image1.jpg", IMREAD_GRAYSCALE );
 
    // Initiate ORB detector
    Ptr<FeatureDetector> detector = ORB::create();
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
// find the keypoints and descriptors with ORB
    detector->detect(gray_image1, keypoints_object);
    detector->detect(gray_image2, keypoints_scene);

    Ptr<DescriptorExtractor> extractor = ORB::create();
    extractor->compute(gray_image1, keypoints_object, descriptors_object );
    extractor->compute(gray_image2, keypoints_scene, descriptors_scene );

// Flann needs the descriptors to be of type CV_32F
    descriptors_scene.convertTo(descriptors_scene, CV_32F);
    descriptors_object.convertTo(descriptors_object, CV_32F);

    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        if( matches[i].distance < 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }


    vector< Point2f > obj;
    vector< Point2f > scene;


    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat img_matches;
    drawMatches( gray_image1, keypoints_object, gray_image2, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow( "Good Matches", img_matches );
    for( int i = 0; i < (int)good_matches.size(); i++ )
  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
    /*
    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, CV_RANSAC );
    // Use the Homography Matrix to warp the images
    cv::Mat result;
    warpPerspective(gray_image1,result,H,Size(gray_image1.cols+gray_image2.cols,gray_image1.rows));
    cv::Mat half(result,cv::Rect(0,0,gray_image2.cols,gray_image2.rows));
    gray_image2.copyTo(half);
    imshow( "Result", result );
    */
    waitKey(0);
    return 0;
}
