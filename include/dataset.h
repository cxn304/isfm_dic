#pragma once
#ifndef DATASET_H
#define DATASET_H
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include "DBoW3/DBoW3.h"
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
        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "element vertex " << landmarked.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
        file << "end_header\n";
        // 写入顶点数据
        for (auto &landmark : landmarked)
        {
            cv::Vec3d pos = landmark.second->pos_;
            Eigen::Matrix<uchar, 3, 1> color = landmark.second->color_;
            file << pos[0] << " " << pos[1] << " " << pos[2] << " "
                 << static_cast<int>(color(0)) << " "
                 << static_cast<int>(color(1)) << " "
                 << static_cast<int>(color(2)) << "\n";
        }
        file.close();
        std::cout << "PLY file saved successfully." << std::endl;
    };

    inline void SavePose(const std::string &filename, const vector<Frame::Ptr> &frames)
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);

        for (int i = 0; i < frames.size(); ++i)
        {
            // 创建一个帧的命名空间
            fs << "frame"
               << "{";
            // 保存图像名称
            fs << "img_name" << frames[i]->img_name;
            // 保存内参
            cv::Mat intrinsics;
            cv::eigen2cv(frames[i]->intrix_, intrinsics);
            fs << "intrinsics" << intrinsics;
            // 保存位姿
            cv::Mat pose;
            cv::eigen2cv(frames[i]->pose_.matrix(), pose);
            fs << "pose" << pose;
            // 关闭帧的命名空间
            fs << "}";
        }
        fs.release();
        std::cout << "pose file saved successfully." << std::endl;
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
        int ComputeMatches(vector<cv::KeyPoint> &kp01, vector<cv::KeyPoint> &kp02,
                            cv::Mat &desc1, cv::Mat &desc2,
                            std::vector<cv::DMatch> &matches,
                            const float distance_ratio);
        void loadMatchesFromFile(const std::string &filename);

        string vocab_file_path;
        vector<vector<cv::KeyPoint>> kpoints_;
        vector<cv::Mat> descriptors_; // descriptor vectors
        vector<string> filenames_;
        Eigen::MatrixXi similarityMatrix_;
        std::map<std::pair<int, int>, std::vector<cv::DMatch>> matchesMap_; // 存储每对图像之间的匹配结果

    private:
        map<int, string> file_paths_;
        cv::Mat dbo_score_;
    };
}
#endif
