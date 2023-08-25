#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "dataset.h"
#include "inits.h"
#include "map.h"
#include "steps.h"

using namespace ISfM;
using namespace std;

int main(int argc, char **argv)
{
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    ISfM::ImageLoader Cimage_loader("./imgs/Viking/");
    ISfM::Dataset Cdates;
    string feature_path = "./data/features.yml";
    string filename_path = "./data/feature_name.txt";
    string matrix_path = "./data/similarityMatrix.yml";
    // step2 : 读取特征及数据库文件,计算匹配并存储到数据库
    cv::Mat sMatrix = Cdates.readDateSet(matrix_path, feature_path, filename_path, Cimage_loader.filenames_); // 读取数据库并建立Cdates
    Cdates.loadMatchesFromFile("./data/match_info.yml");
    // ISfM::Map::Ptr map_ = nullptr;
    // map_ = ISfM::Map::Ptr(new Map); // 初始化地图点

    // ISfM::Steps::Ptr steps = nullptr;
    // steps = ISfM::Steps::Ptr(new Steps);
    // steps->SetMap(map_);

    ISfM::Initializer Cinitializer(Cimage_loader,Cdates);
    Cinitializer.Initialize();

    return 0;
}
