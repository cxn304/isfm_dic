#pragma once
#ifndef INITIALIZER_H
#define INITIALIZER_H
#include "common.h"
#include "projection.h"
#include "files.h"

using namespace std;
namespace ISfM
{
    class Initializer
    {
    public:
        struct Parameters
        {
            unsigned rel_pose_min_num_inlier = 100;     // 2D-2D点对应的内点数量的阈值
            double rel_pose_ransac_confidence = 0.9999; // 求矩阵(H,E)时ransac的置信度
            double rel_pose_essential_error = 4.0;      // 求解决矩阵E的误差阈值
            double rel_pose_homography_error = 12.0;    // 求解决矩阵H的误差阈值
            double init_tri_max_error = 2.0;            // 三角测量时,重投影误差阈值
            double init_tri_min_angle = 4.0;            // 三角测量时, 角度阈值
        };
        struct Statistics
        {
            bool is_succeed = false; // 初始化是否成功
            string method = "None";  // 初始化使用了何种方法
            string fail_reason = "None";

            size_t num_inliers_H = 0; // 估计单应矩阵时，符合单应矩阵的内点的数量
            size_t num_inliers_F = 0; // 估计基础矩阵时，符合基础矩阵的内点的数量
            double H_F_ratio = 0;     // 单应矩阵的内点的数量 除以基础矩阵的内点的数量

            size_t num_inliers = 0;      // 成功三角测量的3D点数(重投影误差小于阈值)
            double median_tri_angle = 0; // 成功三角测量的3D点角度的中位数
            double ave_tri_angle = 0;    // 成功三角测量的3D点角度的平均值
            double ave_residual = 0;     // 平均重投影误差
            cv::Mat Rwto1;                  // 旋转矩阵1(单位矩阵)
            cv::Mat t1;                  // 平移向量1(零向量)
            cv::Mat Rwto2;                  // 旋转矩阵2
            cv::Mat t2;                  // 平移向量2
            vector<cv::Vec3d> points3D;  // 所有2D点所测量出来的3D点,包含了inlier和outlier
            vector<double> tri_angles;   // 每个3D点的角度
            vector<double> residuals;    // 每个3D点的重投影误差
            vector<bool> inlier_mask;    // 标记哪个3D点是内点
        };

    public:
        Initializer() {}
        Initializer(const Parameters &params, const cv::Mat &K);
        Initializer(const ImageLoader &image_loader);
        // 读取相似矩阵
        
        // 找到图像间的相似特征, 最大相关度的两张图片的id, 返回pts1和pts2,要以&取值的方式将pts1传入
        void featureMatching(cv::Mat &similarityMatrix_,
                             vector<vector<cv::KeyPoint>> &kpoints_,
                             vector<cv::Mat> &descriptors_,
                             vector<cv::Point2f> &pts1,
                             vector<cv::Point2f> &pts2);
        // 初始化主函数
        Statistics Initialize(const vector<cv::Point2f> &points2D1,
                              const vector<cv::Point2f> &points2D2);
        void PrintStatistics(const Statistics &statistics);// 打印初始化参数
        string GetFailReason();

    private:
        // 使用自带参数寻找Homo矩阵,inlier_mask用于标记哪些点对被认为是内点
        void FindHomography(const vector<cv::Point2f> &points2D1,
                            const vector<cv::Point2f> &points2D2,
                            cv::Mat &H,
                            vector<bool> &inlier_mask,
                            size_t &num_inliers);
        // 使用自带参数寻找Fundemental矩阵
        void FindFundanmental(const vector<cv::Point2f> &points2D1,
                              const vector<cv::Point2f> &points2D2,
                              cv::Mat &F,
                              vector<bool> &inlier_mask,
                              size_t &num_inliers);
        // 恢复相机外参从Homography矩阵
        bool RecoverPoseFromHomography(const cv::Mat &H,
                                       const vector<cv::Point2f> &points2D1,
                                       const vector<cv::Point2f> &points2D2,
                                       const vector<bool> &inlier_mask_H);
        //
        bool RecoverPoseFromFundanmental(const cv::Mat &F,
                                         const std::vector<cv::Point2f> &points2D1,
                                         const std::vector<cv::Point2f> &points2D2,
                                         const std::vector<bool> &inlier_mask_F);
        // 初始化中的三角化,P1,P2: K*[R,t] (3x3*3x4), return: 3d point
        cv::Vec3d Triangulate(const cv::Mat &P1,
                              const cv::Mat &P2,
                              const cv::Point2f &point2D1,
                              const cv::Point2f &point2D2);

    private:
        Parameters params_;
        Statistics statistics_;
        cv::Mat K_;
        ImageLoader image_loader_;
    };
}
#endif
