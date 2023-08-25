#pragma once
#ifndef MAPBUILDER_H
#define MAPBUILDER_H
#include "common.h"
#include "inits.h"

using namespace std;
namespace ISfM
{
    class MapBuilder
    {
    public:
        struct Parameters
        {
            // 相机内参
            double fx;
            double fy;
            double cx;
            double cy;

            // 畸变参数, 默认参数无效
            double k1 = 0.0;
            double k2 = 0.0;
            double p1 = 0.0;
            double p2 = 0.0;

            Initializer::Parameters init_params;        // 初始化时，所需要用到的参数

            size_t min_num_matches = 10; // 数据库中匹配数大于该阈值的图像对才会被加载进scene graph
            size_t max_num_init_trials = 100;

            double complete_max_reproj_error = 4.0; // 补全track时，最大的重投影误差
            double merge_max_reproj_error = 4.0;    // 合并track时，最大的重投影误差
            double filtered_max_reproj_error = 4.0; // 过滤track时，最大的重投影误差
            double filtered_min_tri_angle = 1.5;    // 过滤track时，最小要满足的角度

            double global_ba_ratio = 1.07; // 当图像增加了该比率时，才会进行global BA

            bool is_visualization = true; // 是否开启重建时， 点云、相机的可视化
        };
        struct Statistics
        {
            // TOOD
        };

    public:
        MapBuilder(const string &database_path, const MapBuilder::Parameters &params);
        // 重建时，需要调用的函数
        // 调用SetUp() 设置重建时需要加载的数据
        // 调用DoBuild() 来进行重建
        // 调用Summary() 输出重建结果的统计信息
        
        void SetUp();
        void doBuild();

        //
        
        // 将重建结果（相机参数、图片参数、3D点）写到文件中
        
        void WriteCOLMAP(const string &directory);
        void WriteOpenMVS(const string &directory);
        void WritePLY(const string &path);
        void WritePLYBinary(const string &path);
        void Write(const string &path);
        void WriteCamera(const string &path);
        void WriteImages(const string &path);
        void WritePoints3D(const string &path);

    private:
        
        // 寻找用于初始化的图像对
        
        vector<int> FindFirstInitialImage() const;
        vector<int> FindSecondInitialImage(int image_id) const;

        
        // 尝试进行初始化，直至成功，或者得到限定的初始化次数
        bool TryInitialize();

        // 尝试注册下一张图片
        bool TryRegisterNextImage(const int &image_id);
        size_t Triangulate(const vector<vector<Map::CorrData>> &points2D_corr_datas,
                           double &ave_residual);

        
        // 如果进行Local BA， 所需要进行的操作
        
        void LocalBA();
        void MergeTracks();
        void CompleteTracks();
        void FilterTracks();

        
        // 如果进行Global BA， 所需要进行的操作
        
        void GlobalBA();
        void FilterAllTracks();
        // TODO : 对图像（或图像对）中没有3D点的2D点进行重建三角测量
        void retriangulate();

        string database_path_;
        Parameters params_;

        cv::Ptr<Initializer> initailizer_;

        int width_;
        int height_;
        cv::Mat K_;
        cv::Mat dist_coef_;
    };

} // namespace MonocularSfM

#endif 