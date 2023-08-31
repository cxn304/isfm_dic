#ifndef PROJECTION_H
#define PROJECTION_H

#include <opencv2/opencv.hpp>

namespace ISfM
{
    class Projection
    {
    public:
    //函数重载是指在同一个作用域内,可以定义具有相同名称但参数列表不同的多个函数.
    //编译器根据函数调用时提供的参数类型和数量来确定要调用哪个函数.
        static bool HasPositiveDepth(const cv::Vec3d &point3D,
                                     const cv::Mat &R,
                                     const cv::Mat &t);
        static bool HasPositiveDepth(const cv::Vec3d &point3D,
                                     const cv::Mat &Rwto1,
                                     const cv::Mat &t1,
                                     const cv::Mat &Rwto2,
                                     const cv::Mat &t2);

        // proj_matrix = [R | t],判断给定的三维坐标 point3D 在投影矩阵 proj_matrix 对应的相机视角中是否具有正深度
        static bool HasPositiveDepth(const cv::Vec3d &point3D,
                                     const cv::Mat &proj_matrix);
        // 冲突, 只能这样写
        static bool HasPositiveDepth(const cv::Mat &proj_matrix1,
                                     const cv::Mat &proj_matrix2,
                                     const cv::Vec3d &point3D);

        static double CalculateReprojectionError(const cv::Vec3d &point3D,
                                                 const cv::Point2f &point2D,
                                                 const cv::Mat &R,
                                                 const cv::Mat &t,
                                                 const cv::Mat &K);
        static double CalculateReprojectionError(const cv::Vec3d &point3D,
                                                 const cv::Point2f &point2D1,
                                                 const cv::Point2f &point2D2,
                                                 const cv::Mat &Rwto1,
                                                 const cv::Mat &t1,
                                                 const cv::Mat &Rwto2,
                                                 const cv::Mat &t2,
                                                 const cv::Mat &K);
        // proj_matrix = K[R | t]
        static double CalculateReprojectionError(const cv::Vec3d &point3D,
                                                 const cv::Point2f &point2D,
                                                 const cv::Mat &proj_matrix);

        static double CalculateReprojectionError(const cv::Vec3d &point3D,
                                                 const cv::Point2f &point2D1,
                                                 const cv::Point2f &point2D2,
                                                 const cv::Mat &proj_matrix1,
                                                 const cv::Mat &proj_matrix2);

        static double CalculateParallaxAngle(const cv::Vec3d &point3D,
                                             const cv::Mat &Rwto1,
                                             const cv::Mat &t1,
                                             const cv::Mat &Rwto2,
                                             const cv::Mat &t2);
        static double CalculateParallaxAngle(const cv::Vec3d &point3d,
                                             const cv::Vec3d &proj_center1,
                                             const cv::Vec3d &proj_center2);
    };
} // namespace Isfm

#endif
