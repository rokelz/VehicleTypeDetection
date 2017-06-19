#include <iostream>

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// If you find this code useful, please add a reference to the following paper in your work:
// Gil Levi and Tal Hassner, "LATCH: Learned Arrangements of Three Patch Codes", arXiv preprint arXiv:1501.03719, 15 Jan. 2015

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main()
{
    Mat img1 = imread( "/home/ruwiniopatha/Desktop/Milestone4/CVBD/SIFT/image1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread( "/home/ruwiniopatha/Desktop/Milestone4/CVBD/SIFT/image2.jpg", IMREAD_GRAYSCALE);

    Ptr<xfeatures2d::SIFT> latch = xfeatures2d::SIFT::create();

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    latch->detect( img1, kpts1 );
    latch->detect( img2, kpts2 );
    //orb_detector->detect(img1, kpts1);
    latch->compute(img1, kpts1, desc1);

    //orb_detector->detect(img2, kpts2);
    latch->compute(img2, kpts2, desc2);

    //BFMatcher matcher(NORM_HAMMING);
	BFMatcher matcher;
  	vector< DMatch > matches;
  	matcher.match( desc1, desc2, matches );
//	vector< vector<DMatch> > nn_matches;
   // matcher.knnMatch(desc1, desc2, nn_matches, 2);

    /*
    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if (dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

    for (unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if (dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
*/
    double max_dist = 0; double min_dist = 100;
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < desc1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
/*
    Mat res= Mat::zeros( img1.size(), CV_8UC3 );
    drawMatches(img1, kpts1, img2, kpts2, matches, res);
    char path[255];
		strcpy(path, "/home/ruwiniopatha/Desktop/Milestone4/CVBT/SIFT/sampl.jpg");
		//cvSaveImage(path, res);
    imwrite(path, res);
*/
   std::vector< DMatch > good_matches;
   for( int i = 0; i < desc1.rows; i++ )
   { if( matches[i].distance <= max(2*min_dist, 0.02) )
     { good_matches.push_back( matches[i]); }
   }
  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img1, kpts1, img2, kpts2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow( "Good Matches", img_matches );
  for( int i = 0; i < (int)good_matches.size(); i++ )
  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
  waitKey(0);
  return 0;

/*
    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "LATCH Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;
    return 0;

    namedWindow("SIFT", CV_WINDOW_AUTOSIZE );
    imshow("SIFT", res);
    waitKey(0); //press any key to quit
 
    return 0;*/
}

#else

int main()
{
    std::cerr << "OpenCV was built without xfeatures2d module" << std::endl;
    return 0;
}
  
#endif
 
