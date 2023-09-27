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
    int rows = Cdates.similarityMatrix_.rows;
    int cols = Cdates.similarityMatrix_.cols;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = i + 1; j < cols; ++j)
        {
            if (Cdates.similarityMatrix_.at<int>(i, j) > 0)
            {
                vector<cv::DMatch> &this_match = Cdates.matchesMap_[std::make_pair(i, j)];
                cv::Mat image1 = cv::imread(Cimage_loader.filenames_[i], cv::IMREAD_COLOR);
                cv::Mat image2 = cv::imread(Cimage_loader.filenames_[j], cv::IMREAD_COLOR);
                cv::Mat matchImage;
                cv::drawMatches(image1, Cdates.kpoints_[i],
                                image2, Cdates.kpoints_[j],
                                this_match, matchImage);
                std::string outputImagePath = "./tmp/" + std::to_string(i) + "_" + std::to_string(j) + "_matches.jpg";
                cv::imwrite(outputImagePath, matchImage);
            }
        }
    }
};

void visualizeChess(const ISfM::ImageLoader &Cimage_loader, ISfM::Dataset &Cdates)
{
    int rows = Cimage_loader.filenames_.size();
    for (int i = 0; i < rows - 1; ++i)
    {
        int j = i + 1;
        vector<cv::DMatch> &this_match = Cdates.matchesMap_[std::make_pair(i, j)];
        cv::Mat image1 = cv::imread(Cimage_loader.filenames_[i], cv::IMREAD_COLOR);
        cv::Mat image2 = cv::imread(Cimage_loader.filenames_[j], cv::IMREAD_COLOR);
        cv::Mat matchImage;
        cv::drawMatches(image1, Cdates.kpoints_[i],
                        image2, Cdates.kpoints_[j],
                        this_match, matchImage);
        std::string outputImagePath = "./tmp/" + std::to_string(i) + "_" + std::to_string(j) + "_matches.jpg";
        cv::imwrite(outputImagePath, matchImage);
    }
};

int main(int argc, char **argv)
{
    bool read_chess = true;
    // step1 : 提取特征, 并存储到数据库
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    ISfM::ImageLoader Cimage_loader("./imgs/Viking/", "./config/default.yaml");
    string feature_path = "./data/features.yml";
    string matrix_path = "./data/similarityMatrix.yml";
    ISfM::Dataset Cdates(Cimage_loader);
    if (read_chess)
    {
        Cdates.readChess(read_chess, Cimage_loader.filenames_);
        Cdates.readDateSet(matrix_path, feature_path, Cimage_loader.filenames_); // 读取数据库并建立Cdates
        Cdates.computeChessMatches();
        visualizeChess(Cimage_loader, Cdates);
    }
    else
    {
        Cdates.establishDbo(Cimage_loader.filenames_);
        // step2 : 读取特征及数据库文件,计算匹配并存储到数据库
        Cdates.readDateSet(matrix_path, feature_path, Cimage_loader.filenames_); // 读取数据库并建立Cdates
        Cdates.computeAndSaveMatches();
        visualizeAndSaveMatches(Cimage_loader, Cdates);
    }

    return 0;
}