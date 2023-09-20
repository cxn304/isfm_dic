#include <iostream>
#include <iostream>
#include "common.h"
#include "dataset.h"
#include "inits.h"
#include "map.h"
#include "steps.h"
#include "dense.h"

using namespace ISfM;

int main(int argc, char **argv)
{
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles("../test_data/", color_image_files, poses_TWC, ref_depth);
    if (ret == false)
    {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // 第一张图
    Mat ref = imread(color_image_files[0], 0); // gray-scale image,0表示灰度图
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;                          // 深度初始值
    double init_cov2 = 3.0;                           // 方差初始值
    Mat depth(height, width, CV_64F, init_depth);     // 深度图
    Mat depth_cov2(height, width, CV_64F, init_cov2); // 深度图方差
    // Use the constructor cv::Mat (int rows, int cols, int type [, fillValue])
    // where type specifies the number of channels and the data type of each element,
    // and fillValue is an optional value to initialize the matrix with.

    for (int index = 1; index < color_image_files.size(); index++)
    {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr)
            continue;
        SE3d pose_curr_TWC = poses_TWC[index];                    // 下面的是reference转换到world,然后左乘world转换到camera,就是reference转换到camera
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // 坐标转换关系：T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}
