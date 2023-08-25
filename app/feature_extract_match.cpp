#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "dataset.h"
#include "inits.h"
#include "map.h"
#include "steps.h"

using namespace std;
using namespace ISfM;
int main(int argc, char **argv)
{
    // step1 : 提取特征, 并存储到数据库
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    ISfM::ImageLoader Cimage_loader("./imgs/Viking/");
    ISfM::Dataset Cdates;
    Cdates.establishDbo(Cimage_loader.filenames_);
    string feature_path = "./data/features.yml";
    string filename_path = "./data/feature_name.txt";
    string matrix_path = "./data/similarityMatrix.yml";
    // step2 : 读取特征及数据库文件,计算匹配并存储到数据库
    cv::Mat sMatrix = Cdates.readDateSet(matrix_path,feature_path,filename_path,Cimage_loader.filenames_);// 读取数据库并建立Cdates
    Cdates.computeAndSaveMatches();

    return 0;
}