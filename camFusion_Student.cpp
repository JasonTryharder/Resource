
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // std::map: It stores only unique keys and that too in sorted order based on its assigned sorting criteria.
    //some helper parameters
    int c = 0;
    std::vector<BoundingBox> prevB = prevFrame.boundingBoxes;
    std::vector<BoundingBox> currB = currFrame.boundingBoxes;
    bool mapT = false;
    int id1;
    int id2;
    std::multimap<int, int> boxIdmap;//multimap before map
    std::vector<int> idV1; 
    std::vector<int> appearCount;
    std::vector<int> appearId;
    // std::vector idV2;
    for (auto it3 = matches.begin(); it3 != matches.end(); ++it3)//if we get Matches as match(descriptor_for_keypoints1, descriptor_for_keypoints2, matches)
    {                                                            //then queryIdx refers to keypoints1 and trainIdx refers to keypoints2, or vice versa.
        cv::KeyPoint prevP = prevFrame.keypoints.at(it3->queryIdx);
        cv::KeyPoint currP = currFrame.keypoints.at(it3->trainIdx);
        
        // cout <<prevP.pt<< endl;
        // cout <<currP.pt<< endl;
        for (auto itbox1 = prevB.begin(); itbox1 != prevB.end(); ++itbox1)
        {
            // cout<< (*itbox1).boxID << "the box size of each frame"<<endl;
            // cout<< (*itbox1).roi << "the box size of each frame"<<endl;
            if ((*itbox1).roi.contains(prevP.pt))
            {
                id1 = (*itbox1).boxID;
                mapT = true;                
            }
//A-------------------------------------- cout<< (itbox1).roi<<endl;// why this doesn't work, * vs. no *
        
            if (mapT)
            {
                for (auto itbox2 = currB.begin(); itbox2 != currB.end(); ++itbox2)
                {
                    mapT = false;
                    if ((*itbox2).roi.contains(currP.pt))
                    {
                        // cout<<"I am inside"<<endl;
                        id2 = (*itbox2).boxID;
                        // cout << '\t'<< id1 << '\t'<< id2 << "this is BoxID" << endl;  
                        mapT = true; 
                        boxIdmap.insert(pair<int, int>(id1,id2));
                    }
                }
            }
        }
 
    }
    // to count each matches
    for (auto itbox1 = prevB.begin(); itbox1 != prevB.end(); ++itbox1)
    {
        std::multimap<int, int>::iterator it;
        idV1.clear();// for different prev boxID clear container
        // extract matches from multimap into idV1 vector 
        // use boxIdmap.equal_range will get every key 
        for (it = boxIdmap.equal_range((*itbox1).boxID).first; it !=boxIdmap.equal_range((*itbox1).boxID).second; ++it)
        {
            idV1.push_back((*it).second);//stores all matches from curr ID for(*itbox1).boxID in prev
            // cout << "Curr boxID is" << (*it).second<< endl;
        }
        // cout << "Number for boxID" << (*itbox1).boxID  << "of prev is" << idV1.size() << endl;
        // count the highest matches in idV1 
        appearCount.clear();//for different prevID clear container
        for (auto itbox1dummy = currB.begin(); itbox1dummy != currB.end(); ++itbox1dummy)
        {
            if (appearCount.size() == 0)
            {
                appearCount.push_back(std::count (idV1.begin(),idV1.end(),(*itbox1dummy).boxID));
                appearId.push_back((*itbox1dummy).boxID);
                // cout << appearCount.back() << "for boxID in prev of 1111-------" << (*itbox1).boxID << endl;
            }
            else
            {
                if (std::count(idV1.begin(),idV1.end(),(*itbox1).boxID) > appearCount.back())
                {
                    appearCount.push_back(std::count (idV1.begin(),idV1.end(),(*itbox1dummy).boxID));
                    appearId.push_back((*itbox1dummy).boxID);
                    // cout << appearCount.back() << "for boxID in prev of 2222-------" << (*itbox1).boxID << endl;
                    
                } 
                else
                {
                    // cout << appearCount.back() << "VS" << std::count(idV1.begin(),idV1.end(),(*itbox1dummy).boxID)<< "for 3333--------"<<(*itbox1).boxID << "and" <<(*itbox1dummy).boxID<< endl;
                }
            }
        }
        id1 = (*itbox1).boxID;
        id2 = appearId.back();
        cout << id1 << "  with  " << id2 << "appears" << endl;
        bbBestMatches.insert(pair<int, int>(id1,id2));
        // cout << id1 <<"with" << id2 << "apperars "<< std::count (idV1.begin(),idV1.end(),(*itbox1).boxID) << "times" <<endl;
        // cout << boxIdmap.count((*itbox1).boxID) << "for boxID =" << (*itbox1).boxID << endl;// compare this with actual uniques appearances 
    }
    for (auto itbox2 = currB.begin(); itbox2 != currB.end(); ++itbox2)
    {
        // cout << boxIdmap.count((*itbox2).boxID) << " for boxID =" << (*itbox2).boxID << endl;
    }
}
