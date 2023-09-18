#pragma once
#ifndef DATASET_H
#define DATASET_H
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
// #include "DBoW3/DBoW3.h"
#include "files.h"
#include "common.h"
#include "frame.h"
#include "map.h"
#include <boost/format.hpp>

using namespace std;
namespace ISfM
{
    inline void SavePLY(const std::string &filename, const Map::LandmarksType &landmarked)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cout << "Failed to open the file." << std::endl;
            return;
        }

        // 写入PLY头部信息
        file << "ply" << std::endl;
        file << "format ascii 1.0" << std::endl;
        file << "element vertex " << landmarked.size() / 3 << std::endl;
        file << "property float x" << std::endl;
        file << "property float y" << std::endl;
        file << "property float z" << std::endl;
        file << "end_header" << std::endl;
        // 写入顶点数据
        for (auto &landmark : landmarked)
        {
            cv::Vec3d pos = landmark.second->pos_;
            file << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
        }
        file.close();
        std::cout << "PLY file saved successfully." << std::endl;
    };

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
        int DetectFeatures(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

        cv::Mat getDboScore() const { return dbo_score_; }
        void setDboScore(const cv::Mat &score) { dbo_score_ = score; }
        // void saveSimilarMatrix(DBoW3::Vocabulary &vocab,
        //                        const string &feature_path, const string &filename_path);
        void saveORBSimilar(const string feature_path, const string filename_path);
        void readImageSave(const string feature_path, const string filename_path);
        cv::Mat readDateSet(const string &matrixPath, const string &feature_path, const string &filename_path,
                            const vector<string> &filenames);
        void computeAndSaveMatches();
        void ComputeMatches(vector<cv::KeyPoint> &kp01, vector<cv::KeyPoint> &kp02,
                            cv::Mat &desc1, cv::Mat &desc2,
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
        map<int, string> file_paths_;
        cv::Mat dbo_score_;
    };
}
#endif
