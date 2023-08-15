#pragma once
#ifndef CAMERA_HPP
#define CAMERA_HPP
#include "common.hpp"

namespace ISfM
{
    class Camera
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Camera> Ptr;

        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
        double k1_ = 0, k2_ = 0;
        SE3 pose_;     // extrinsic, from stereo camera to single camera
        SE3 pose_inv_; // inverse of extrinsics

        Camera();

        Camera(double fx, double fy, double cx, double cy,
               const SE3 &pose)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy), pose_(pose)
        {
            pose_inv_ = pose_.inverse();
        }

        SE3 pose() const { return pose_; }

        // return intrinsic matrix
        Mat33 K() const
        {
            Mat33 k;
            k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
            return k;
        }

        Vec2 projectWithDistortion(double xp, double yp)
        {
            double r2 = xp * xp + yp * yp;
            double distortion = 1.0 + r2 * (k1 + k2 * r2);
            Vec2 predictions;
            predictions[0] = fx_ * distortion * xp;
            predictions[1] = fy_ * distortion * yp;
            return predictions;
        }

        // coordinate transform: world, camera, pixel
        Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

        Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

        Vec2 camera2pixel(const Vec3 &p_c);

        Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

        Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);

        Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);
    };
}
#endif