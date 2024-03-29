#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
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
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
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
}
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{ 
    // cv::Ptrcv::FeatureDetector detector; 
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0) 
    { 
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel; 
        bool bNMS = true; // perform non-maxima suppression on keypoints 
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; 
        // TYPE_9_16, TYPE_7_12, TYPE_5_8 detector = cv::FastFeatureDetector::create(threshold, bNMS, type); 
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        vector<cv::KeyPoint> keypoints;
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "FAST";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }
    else if (detectorType.compare("BRISK") == 0)
    { 
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel; 
        int octaves = 3;
        float patternScale = 1.0f;	
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create(threshold, octaves, patternScale);
        vector<cv::KeyPoint> keypoints;
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "BRISK";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }
    // else if (detectorType.compare("ORB") == 0)
    // {   
    //     int nfeatures = 500;
    //     float scaleFactor = 1.2f;
    //     int nlevels = 8;
    //     int edgeThreshold = 31;
    //     int firstLevel = 0;
    //     int WTA_K = 2;
    //     int scoreType=ORB::HARRIS_SCORE;
    //     int patchSize = 31;
    //     int fastThreshold = 20; 
    //     cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create (nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    //     vector<cv::KeyPoint> keypoints;
    //     double t = (double)cv::getTickCount();
    //     detector->detect(img, keypoints);
    //     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //     cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    //     if (bVis)
    //     {
    //         cv::Mat visImage = img.clone();
    //         cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //         string windowName = "Shi-Tomasi Corner Detector Results";
    //         cv::namedWindow(windowName, 6);
    //         imshow(windowName, visImage);
    //         cv::waitKey(0);
    //     }
    // }
    // else if (detectorType.compare("KAZE") == 0)
    // { 
    //     cv::Ptr<cv::FeatureDetector> detector =  cv::KAZE::create(	
    //         bool 	extended = false,
    //         bool 	upright = false,
    //         float 	threshold = 0.001f,
    //         int 	nOctaves = 4,
    //         int 	nOctaveLayers = 4,
    //         int 	diffusivity = KAZE::DIFF_PM_G2 
    //     )	
    //     vector<cv::KeyPoint> keypoints;
    //     double t = (double)cv::getTickCount();
    //     detector->detect(img, keypoints);
    //     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //     cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    //     if (bVis)
    //     {
    //         cv::Mat visImage = img.clone();
    //         cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //         string windowName = "Shi-Tomasi Corner Detector Results";
    //         cv::namedWindow(windowName, 6);
    //         imshow(windowName, visImage);
    //         cv::waitKey(0);
    //     }
    // }
    // else if ("SIFT" == detectorType)
    // { 
    //     int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel; 
    //     bool bNMS = true; // perform non-maxima suppression on keypoints 
    //     cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; 
    //     // TYPE_9_16, TYPE_7_12, TYPE_5_8 detector = cv::FastFeatureDetector::create(threshold, bNMS, type); 
    //     cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    //     vector<cv::KeyPoint> keypoints;
    //     t = (double)cv::getTickCount();
    //     detector->detect(imgGray, keypoints);
    //     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //     cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // }

}
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // load image from file
    // cv::Mat img;
    // img = cv::imread("../images/img1.png");
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale
    // don't need to convert at this point

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
// visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);
    

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    // vector<cv::KeyPoint> keypoints;
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
    
    // EOF STUDENT CODE
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}