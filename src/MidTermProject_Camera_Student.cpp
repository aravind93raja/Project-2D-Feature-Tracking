/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

void write_csv(string filename, vector<string> col_name,vector<double> time1 ,vector<double> time2 ,vector<int> number)
{
    ofstream myFile(filename);
    
    for(int j = 0; j < col_name.size(); ++j)
    {
        myFile << col_name[j];
        if(j != col_name.size() - 1) 
            myFile << ","; 
    }
    myFile << "\n\n";
    
    
    for(int i = 0; i < time1.size(); ++i)
    {

        myFile << time1[i];
        myFile << ","; 
        myFile << time2[i];
        myFile << ","; 
        myFile << number[i];
        myFile << ","; 
        myFile << "\n";
    }
    
    
    myFile.close();
}


/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    string detectorType = "FAST";  //// HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT , BRISK
    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    string BinaryOrHog = "DES_BINARY"; // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    vector<double> DetectorTimeData_vector;
    double DetectorTimeData;
    vector<double> DescriptorTimeData_vector;
    double DescriptorTimeData;
    vector<int> MatchedKeypoints_vector;
    int MatchedKeypoints;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */
        cout << "________________________________________________________" << endl;

        cout << "IMAGE NO :"<< imgIndex << endl;
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        if(dataBuffer.size()==dataBufferSize)
            dataBuffer.erase(dataBuffer.begin());

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
        cout << endl;
        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
       

        if (detectorType.compare("SHITOMASI") == 0)
        {
            DetectorTimeData = detKeypointsShiTomasi(keypoints, imgGray, bVis);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            DetectorTimeData = detKeypointsHarris(keypoints, imgGray, bVis);
        }
        else
        {
            DetectorTimeData = detKeypointsModern(keypoints, img, detectorType, bVis);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            std::vector<cv::KeyPoint> neededPoints;
            for (auto point : keypoints) 
                    if (vehicleRect.contains(cv::Point2f(point.pt))) 
                        neededPoints.push_back(point); 
            
            std::cout << "The Keypoints On the preceding vehicle are : "<< neededPoints.size()<<std::endl;
            keypoints = neededPoints;
        }
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;
        cout << endl;
        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        
        DescriptorTimeData = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
        cout << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            MatchedKeypoints = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, BinaryOrHog, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            cout << endl;
            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

        if (dataBuffer.size() <= 1)
            MatchedKeypoints_vector.push_back(0);
        else
            MatchedKeypoints_vector.push_back(MatchedKeypoints);
        DescriptorTimeData_vector.push_back(DescriptorTimeData);
        DetectorTimeData_vector.push_back(DetectorTimeData);
        

    } // eof loop over all images

   
    string filename = detectorType + "_" + descriptorType;
    vector<string> col_name = {"Det_time","Des_time","Match_points"};
    write_csv(filename, col_name,DetectorTimeData_vector ,DescriptorTimeData_vector ,MatchedKeypoints_vector);

    return 0;
}
