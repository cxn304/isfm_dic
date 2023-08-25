#pragma once
#ifndef DATASET_H
#define DATASET_H
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include "DBoW3/DBoW3.h"
#include "files.h"
#include "common.h"
#include "frame.h"
#include "feature_utils.h"

using namespace std;
namespace ISfM
{
    class Dataset
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        struct Image
    {
        int id;
        string name;
    };
        Dataset() {}
        void establishDbo(const vector<string> &filenames);
        int DetectFeatures(const cv::Mat &image,vector<cv::KeyPoint> &keypoints,cv::Mat &descriptors);
        const map<string, pair<vector<cv::KeyPoint>,
                               cv::Mat>> &
        getFeatures() const { return features_; } // image name and their kpts, des
        cv::Mat getDboScore() const { return dbo_score_; }
        void setDboScore(const cv::Mat &score) { dbo_score_ = score; }
        void saveSimilarMatrix(DBoW3::Vocabulary &vocab,
                               const string &feature_path, const string &filename_path);
        void saveORBSimilar(const string feature_path, const string filename_path);
        void readImageSave(const string feature_path, const string filename_path);
        cv::Mat readDateSet(const string &matrixPath, const string &feature_path, const string &filename_path,
                            const vector<string> &filenames);
        void computeAndSaveMatches();
        void ComputeMatches(const cv::Mat &desc1,
                            const cv::Mat &desc2,
                            std::vector<cv::DMatch> &matches,
                            const float distance_ratio);
        void loadMatchesFromFile(const std::string &filename);

        string vocab_file_path;
        vector<vector<cv::KeyPoint>> kpoints_;
        vector<cv::Mat> descriptors_; // descriptor vectors
        vector<string> filenames_;
        Eigen::MatrixXd similarityMatrix_;
        std::map<std::pair<int, int>, std::vector<cv::DMatch>> matchesMap_; // 存储每对图像之间的匹配结果
    private:
        map<string, pair<vector<cv::KeyPoint>, cv::Mat>> features_;
        map<int,string> file_paths_;
        cv::Mat dbo_score_; 
    };
}
#endif
