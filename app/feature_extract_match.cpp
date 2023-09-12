#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "dataset.h"
#include "inits.h"
#include "map.h"
#include "steps.h"

using namespace std;
using namespace ISfM;

void visualizeAndSaveMatches(const ISfM::ImageLoader &Cimage_loader, ISfM::Dataset &Cdates)
{
    for (int i = 0; i < Cimage_loader.filenames_.size() - 1; ++i)
    {
        vector<cv::DMatch> &this_match = Cdates.matchesMap_[std::make_pair(i, i + 1)];
        cv::Mat image1 = cv::imread(Cimage_loader.filenames_[i], cv::IMREAD_COLOR);
        cv::Mat image2 = cv::imread(Cimage_loader.filenames_[i + 1], cv::IMREAD_COLOR);
        cv::Mat matchImage;
        cv::drawMatches(image1, Cdates.kpoints_[i],
                        image2, Cdates.kpoints_[i + 1],
                        this_match, matchImage);
        std::string outputImagePath = "./tmp/" + std::to_string(i) + "_" + std::to_string(i + 1) + "_matches.jpg";
        cv::imwrite(outputImagePath, matchImage);
    }
};

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
    visualizeAndSaveMatches(Cimage_loader, Cdates);
    return 0;
}