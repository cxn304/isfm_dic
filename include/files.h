#pragma once
#ifndef FILES_H
#define FILES_H
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include "common.h"

using namespace std;
namespace ISfM
{
    class ImageLoader
    {
    public:
        vector<string> filenames_;
        string dataset_dir_;
        int num_images_;
        int width_;
        int height_;
        double fx_, fy_;
        int board_size_;
        int leastMatchNum_, numFeatures_, numFeaturesInit_, numFeaturesTracking_;

        ImageLoader(){};
        ImageLoader(const string &img_path, const string &config_path) : img_path_(img_path), num_images_(0)
        {
            // 获取文件名列表
            filenames_ = get_filenames_(img_path);
            std::sort(filenames_.begin(), filenames_.end());
            // 统计图片数量
            for (const string &filename : filenames_)
            {
                if (is_image_file(filename))
                {
                    num_images_++;
                }
            }
            // 获取图像信息
            get_images_info(img_path, filenames_, width_, height_);

            cv::FileStorage fs(config_path, cv::FileStorage::READ);
            fs["dataset_dir"] >> dataset_dir_;
            // 读取其他参数
            fs["least_match_num"] >> leastMatchNum_;
            fs["num_features"] >> numFeatures_;
            fs["num_features_init"] >> numFeaturesInit_;
            fs["num_features_tracking"] >> numFeaturesTracking_;
            fs["fx"] >> fx_;
            fs["fy"] >> fy_;
            fs["boardSize"] >> board_size_;
        }

        void get_images_info(const string &path, vector<string> &filenames_, int &width, int &height)
        {
            for (const string &filename : filenames_)
            {
                if (is_image_file(filename))
                {
                    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
                    // 获取图片的长和宽
                    width = image.cols;
                    height = image.rows;
                    break;
                }
            }
        }

    private:
        string img_path_;

        vector<string> get_filenames_(const string &path)
        {
            vector<string> filenames_;
            DIR *dirp = opendir(path.c_str());
            struct dirent *dp;
            while ((dp = readdir(dirp)) != nullptr)
            {
                if (dp->d_type == DT_REG)
                { // 如果是普通文件
                    filenames_.push_back(path + dp->d_name);
                }
            }
            closedir(dirp);
            return filenames_;
        }

        bool is_image_file(const string &filename)
        {
            // 检查文件扩展名是否为jpg、png或bmp
            string ext = filename.substr(filename.find_last_of(".") + 1);
            return (ext == "jpg" || ext == "png" || ext == "bmp");
        }
    };
}
#endif
