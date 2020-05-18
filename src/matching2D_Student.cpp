#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
int matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{   
    int no_matched_points;
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {   
        //NOTE : This Line was modified from what was originally provided  ,Because we can't use HAMMING with SIFT Descriptor
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        std::cout << "BF matching" << std::endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
                if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::cout << "FLANN matching" << std::endl;
    }


    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        no_matched_points = matches.size();
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<vector<cv::DMatch>> knn_matches;
       double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2); 

        double minDescDistRatio = 0.8;

        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
                matches.push_back((*it)[0]);

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
        no_matched_points=matches.size();
        
    }

    return no_matched_points;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else  if (descriptorType.compare("ORB") == 0) 
        extractor = cv::ORB::create();
    else if (descriptorType.compare("FREAK") == 0) 
        extractor = cv::xfeatures2d::FREAK::create();
    else if (descriptorType.compare("AKAZE") == 0) 
        extractor = cv::AKAZE::create();
    else if (descriptorType.compare("SIFT") == 0) 
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    else if (descriptorType.compare("BRIEF") == 0) 
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

    int blockSize = 2 ;
    int apertureSize = 3 ;
    int minResponse=100 ;
    double k= 0.04;
    double maxOverlap = 0.0 ;
    int scaledApertureSize= apertureSize * 2 ;

    double t = (double)cv::getTickCount();

    cv::Mat dst, dstNorm, dstNormScaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    bool newKeypointProcessed = false ;

    for (int i=0 ; i < dstNorm.rows; i++) {
        for (int j= 0 ; j < dstNorm.cols; j++) {

            const int response{ static_cast<int>(dstNorm.at<float>(i, j)) };

            if (static_cast<int>(dstNorm.at<float>(i, j)) > minResponse) {
                cv::KeyPoint newKeypoint;
                newKeypoint.pt = cv::Point2f(j, i);
                newKeypoint.size = scaledApertureSize;
                newKeypoint.response = response;
                newKeypoint.class_id = 0;

                newKeypointProcessed = false;

                for (auto it= keypoints.begin(); it != keypoints.end(); it++) 
                {
                    if (cv::KeyPoint::overlap(newKeypoint, (*it)) > maxOverlap) 
                    {

                        newKeypointProcessed = true;

                        if (newKeypoint.response > (*it).response) 
                            *it = newKeypoint;

                    }
                }

                if (!newKeypointProcessed) 
                    keypoints.push_back(newKeypoint);
            }
        }
        
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) 
    {
        cv::Mat visImage= img.clone() ;
        string windowName = "Harris Detector Output" ;

        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);

        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return t;

}

double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{


    cv::Ptr<cv::FeatureDetector> detector;
    //FAST, BRISK, ORB, AKAZE, and SIFT - FAST DETECTORS MENTIONED IN TASK
    if (detectorType.compare("FAST") == 0) 
    {
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(30, true, type);
    } 
    else if (detectorType.compare("BRISK") == 0)
       detector = cv::BRISK::create();
    else if (detectorType.compare("ORB") == 0) 
       detector = cv::ORB::create();
    else if (detectorType.compare("AKAZE") == 0) 
       detector = cv::AKAZE::create();
    else if (detectorType.compare("SIFT") == 0) 
       detector = cv::xfeatures2d::SIFT::create();
    

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType <<" detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) 
    {
        cv::Mat visImage= img.clone() ;
        string windowName = detectorType + "Detector Output" ;

        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);

        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return t;

}