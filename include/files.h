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
        vector<string> filenames;
        int num_images_;
        int width_;
        int height_;
        ImageLoader(){};
        ImageLoader(const string &path) : path_(path), num_images_(0)
        {
            // 获取文件名列表
            filenames = get_filenames(path);

            // 统计图片数量
            for (const string &filename : filenames)
            {
                if (is_image_file(filename))
                {
                    num_images_++;
                }
            }
            // 获取图像信息
            get_images_info(path, filenames, width_, height_);
        }

        void get_images_info(const string &path, vector<string> &filenames, int &width, int &height)
        {
            for (const string &filename : filenames)
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
        string path_;

        vector<string> get_filenames(const string &path)
        {
            vector<string> filenames;
            DIR *dirp = opendir(path.c_str());
            struct dirent *dp;
            while ((dp = readdir(dirp)) != nullptr)
            {
                if (dp->d_type == DT_REG)
                { // 如果是普通文件
                    filenames.push_back(path + dp->d_name);
                }
            }
            closedir(dirp);
            return filenames;
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
