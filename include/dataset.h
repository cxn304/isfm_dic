#pragma once
#ifndef DATASET_H
#define DATASET_H
#include "DBoW3/DBoW3.h"
#include "files.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

using namespace std;
namespace ISfM
{
    class Dataset
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Dataset> Ptr;
        Dataset() {}
        void establishDbo(const std::vector<std::string> &filenames);
        const std::map<std::string, std::pair<std::vector<cv::KeyPoint>,
                                              cv::Mat>> &
        getFeatures() const { return features_; } // image name and their kpts, des
        cv::Mat getDboScore() const { return dbo_score_; }
        void setDboScore(const cv::Mat &score) { dbo_score_ = score; }
        void findImageSimilar(const string vocab_file_path,
         const string feature_path, const string filename_path);
        void readImageSave(const string feature_path, const string filename_path);
        
        vector<vector<cv::KeyPoint>> kpoints_;
        std::string vocab_file_path;
        vector<cv::Mat> descriptors_; // descriptor vectors
        vector<string> filenames_;
        Eigen::MatrixXd similarityMatrix_;
    private:
        std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> features_;
        cv::Mat dbo_score_;
    };
}
#endif
