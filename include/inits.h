#pragma once
#ifndef INITIALIZER_H
#define INITIALIZER_H
#include "common.h"
#include "projection.h"
#include "files.h"
#include "point3d.h"
#include "dataset.h"
#include "map.h"
#include "feature.h"
#include "triangulate.h"
#include "camera.h"

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
        struct Returns
        {
            cv::Mat K_;
            Map::Ptr map_ = nullptr;                                                          // 初始化时的map,要传递到step里面的
            vector<Frame::Ptr> frames_;                                                       // 初始化时就搞定了所有frame
            std::vector<std::pair<std::pair<int, int>, std::vector<cv::DMatch>>> matchesMap_; // 存储每对图像之间的匹配结果,传递到step里面
            cv::Mat similar_matrix_;
            int id1, id2;
        };
        struct Statistics
        {
            bool is_succeed = false; // 初始化是否成功
            string method = "None";  // 初始化使用了何种方法
            string fail_reason = "None";

            size_t num_inliers_H = 0; // 估计单应矩阵时,符合单应矩阵的内点的数量
            size_t num_inliers_F = 0; // 估计基础矩阵时,符合基础矩阵的内点的数量
            double H_F_ratio = 0;     // 单应矩阵的内点的数量 除以基础矩阵的内点的数量

            size_t num_inliers = 0;      // 成功三角测量的3D点数(重投影误差小于阈值)
            double median_tri_angle = 0; // 成功三角测量的3D点角度的中位数
            double ave_tri_angle = 0;    // 成功三角测量的3D点角度的平均值
            double ave_residual = 0;     // 平均重投影误差
            cv::Mat Rwto1;               // 旋转矩阵1(单位矩阵)
            cv::Mat t1;                  // 平移向量1(零向量)
            cv::Mat Rwto2;               // 旋转矩阵2
            cv::Mat t2;                  // 平移向量2
        };

    public:
        Initializer(const Parameters &params, const cv::Mat &K);
        Initializer(const ImageLoader &image_loader, const Dataset::Ptr &Cdate, const cv::Mat &sMatrix);
        // 读取相似矩阵

        // 找到图像间的相似特征, 最大相关度的两张图片的id, 返回pts1和pts2,要以&取值的方式将pts1传入
        void featureMatching(double img_index1, double img_index2,
                             vector<vector<cv::KeyPoint>> &kpoints_,
                             vector<cv::Mat> &descriptors_,
                             vector<Feature::Ptr> &pts1,
                             vector<Feature::Ptr> &pts2);
        // 初始化主函数
        Returns Initialize();
        void PrintStatistics(const Statistics &statistics); // 打印初始化参数
        string GetFailReason();

    private:
        // 计算feature数量
        void coutFeaturePoint(const vector<Feature::Ptr> &feature2D1,
                              const vector<Feature::Ptr> &feature2D2);
        // 使用自带参数寻找Fundemental矩阵
        void FindFundanmental(const vector<Feature::Ptr> &points2D1,
                              const vector<Feature::Ptr> &points2D2,
                              cv::Mat &F,
                              vector<bool> &inlier_mask,
                              size_t &num_inliers);
        // 三角化点,包括将三角化后的点加入map和各个frame中
        void TriangulateInitPoints(Frame::Ptr &frame_one, Frame::Ptr &frame_two);

        bool RecoverPoseFromFundanmental(const cv::Mat &F,
                                         const vector<Feature::Ptr> &points2D1,
                                         const vector<Feature::Ptr> &points2D2,
                                         const vector<bool> &inlier_mask_F,
                                         const int id1, const int id2);
        // 初始化中的三角化,P1,P2: K*[R,t] (3x3*3x4), return: 3d point
        cv::Vec3d Triangulate(const cv::Mat &P1,
                              const cv::Mat &P2,
                              const cv::Point2f &point2D1,
                              const cv::Point2f &point2D2);
        // 自定义比较函数，按照 std::vector<cv::DMatch> 的数量从大到小进行排序
        static bool compareByVectorSize(const std::pair<std::pair<int, int>,
                                                        std::vector<cv::DMatch>> &a,
                                        const std::pair<std::pair<int, int>, std::vector<cv::DMatch>> &b)
        {
            return a.second.size() > b.second.size();
        }
        // 查找matchesVec_里面的keyToFind的cv::DMatch
        std::vector<cv::DMatch> findMatch(std::pair<int, int> &keyToFind,
                                          std::vector<std::pair<std::pair<int, int>, std::vector<cv::DMatch>>> &matchesVec_);

    public:
        vector<Frame::Ptr> frames_; // 所有的frame信息
        Parameters params_;
        Statistics statistics_;
        Returns returns_;
        cv::Mat K_;                // 传递到step里面的
        vector<cv::Mat> K_vector_; // 多个内参的预选
        ImageLoader image_loader_;
        Dataset::Ptr Cdate_;                                                              // 传递一部分到step中
        Map::Ptr map_ = nullptr;                                                          // 初始化时的map,要传递到step里面的
        std::vector<std::pair<std::pair<int, int>, std::vector<cv::DMatch>>> matchesMap_; // 存储每对图像之间的匹配结果,传递到step里面
        Camera::Ptr camera_one_;
        cv::Mat similar_matrix_;
    };
}
#endif
