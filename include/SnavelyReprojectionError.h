#pragma once
#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

namespace ISfM
{
    class SnavelyReprojectionError
    {
    public:
        SnavelyReprojectionError(double observation_x,
                                 double observation_y) : observed_x(observation_x),
                                                         observed_y(observation_y) {}

        template <typename T>
        // operator()是括号运算符，实现了Ceres计算误差的接口
        bool operator()(const T *const camera,
                        const T *const point,
                        T *residuals) const
        {
            // camera[0,1,2] are the angle-axis rotation
            T predictions[2];
            // 实际的计算在CamProjectionWithDistortion中
            CamProjectionWithDistortion(camera, point, predictions);
            residuals[0] = predictions[0] - T(observed_x);
            residuals[1] = predictions[1] - T(observed_y);

            return true;
        }

        // camera : 9 dims array
        // [0-2] : angle-axis rotation
        // [3-5] : translateion
        // [6-7] focal length, [8-9] second and forth order radial distortion
        // [10-11]: cx, cy
        // point : 3D location.
        // predictions : 2D predictions with center of the image plane.
        template <typename T>
        static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions)
        {
            // Rodrigues' formula
            T p[3];
            AngleAxisRotatePoint(camera, point, p);
            // camera[3,4,5] are the translation
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            // Compute the center fo distortion
            T xp = -p[0] / p[2];
            T yp = -p[1] / p[2];

            // Apply second and fourth order radial distortion
            const T &l1 = camera[8];
            const T &l2 = camera[9];
            const T &cx = camera[10];
            const T &cy = camera[11];
            const T &fx = camera[6];
            const T &fy = camera[7];
            T r2 = xp * xp + yp * yp;
            T distortion = T(1.0) + r2 * (l1 + l2 * r2);

            predictions[0] = fx * distortion * xp + cx;
            predictions[1] = fy * distortion * yp + cy;

            return true;
        }
        // 该类的静态函数Create作为外部调用的接口，直接返回一个可以自动求导的Ceres代价函数
        // 我们只要调用create函数，把代价函数放入Ceres::Problem即可
        // static ceres::CostFunction * 表示函数返回值的类型是 ceres::CostFunction*，
        // 也就是指向 ceres::CostFunction 类型对象的指针
        static ceres::CostFunction *Create(const double observed_x, const double observed_y)
        {
            // 创建一个 ceres::AutoDiffCostFunction 对象，
            // 并使用 SnavelyReprojectionError 类型的构造函数初始化这个对象
            return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 12, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
        } // 2表示这个cost function的残差数量，也就是输出的维度,即图像平面上的x和y坐标残差
        //  9表示优化问题中相机参数的数量.在这个例子中,相机有9个参数,包括旋转,平移,焦距和径向畸变参数
        // 3表示优化问题中三维点坐标的数量。在这个例子中，三维点有3个坐标值（x、y、z）

    private:
        double observed_x;
        double observed_y;
    };
}

#endif // SnavelyReprojection.h
