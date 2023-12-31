#pragma once
#ifndef CAMERA_H
#define CAMERA_H
#include "common.h"

namespace ISfM
{
    class Camera
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Camera> Ptr;

        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
        double k1_ = 0, k2_ = 0; // distortion param set to public

        Camera();

        Camera(double fx, double fy, double cx, double cy)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy)
        {
        }

        Vec2 projectWithDistortion(double xp, double yp)
        {
            double r2 = xp * xp + yp * yp;
            double distortion = 1.0 + r2 * (k1_ + k2_ * r2);
            Vec2 predictions;
            predictions[0] = fx_ * distortion * xp + cx_;
            predictions[1] = fy_ * distortion * yp + cy_;
            return predictions;
        };

        void setIntrinsic(Vec6 &out_intrinsic)
        {
            fx_ = out_intrinsic[0];
            fy_ = out_intrinsic[1];
            cx_ = out_intrinsic[2];
            cy_ = out_intrinsic[3];
            k1_ = out_intrinsic[4];
            k2_ = out_intrinsic[5];
        }

        Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

        Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

        Vec2 camera2pixel(const Vec3 &p_c,double k1, double k2);

        Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

        Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);

        Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);
    };
}
#endif