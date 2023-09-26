#pragma once
#ifndef STEPS_H
#define STEPS_H

#include <iostream>
#include <opencv2/features2d.hpp>
#include "common.h"
#include "frame.h"
#include "map.h"
#include "inits.h"
#include "camera.h"
#include "feature.h"
#include "triangulate.h"
#include "g2o_types.h"

namespace ISfM
{
    enum class ConstructionStatus
    {
        TRACKING_GOOD,
        TRACKING_BAD,
        LOST
    };

    /**
     * 优化过程
     * 估计当前帧Pose,在满足关键帧条件时向地图加入关键帧并触发优化
     */
    class Steps
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Steps> Ptr;

        Steps(const Initializer::Returns &returns, const ImageLoader &Cimage_loader, Camera::Ptr &camera_one); // 构造函数

        /// 外部接口,添加一个帧(图像)并计算其定位结果
        bool AddFrame(Frame::Ptr frame);

        void SetMap(Map::Ptr map) { map_ = map; }

        ConstructionStatus GetStatus() const { return status_; }

        void SetLastFrame(Frame::Ptr &frame)
        {
            last_frame_ = frame;
        }

        void SetCameras(Camera::Ptr camera_one)
        {
            camera_ = camera_one;
        }

        std::vector<Frame::Ptr> getFrames() const
        {
            return frames_;
        }

        void localBA(Map::KeyframesType &keyframes, Map::LandmarksType &landmarks);
        void gloabalBA(Map::LandmarksType &landmarks);

        void count_feature_point(Map::LandmarksType &landmarks);

        vector<double> average_reprojection_error_;
        vector<pair<pair<int, int>, vector<cv::DMatch>>> matchesMap_; // 存储每对图像之间的匹配结果,传递到step里面

    private:
        std::vector<cv::DMatch> findMatch(std::pair<int, int> &keyToFind,
                                          std::vector<std::pair<std::pair<int, int>, std::vector<cv::DMatch>>> &matchesVec_);
        void setIntrinsic(const Vec6 &out_intrinsic)
        {
            intrinsic_ = out_intrinsic;
        }
        /**
         * Track in normal mode
         * @return true if success
         */
        bool Track();

        /**
         * Reset when lost
         * @return true if success
         */
        bool Reset();

        /**
         * estimate current frame's pose
         * @return num of inliers
         */
        int EstimateCurrentPose();

        /**
         * set current frame as a keyframe and insert it into backend
         * @return true if success
         */
        bool InsertKeyframe();

        /**
         * Find the corresponding features in current_frame_ of last_frame_
         * @return num of features found
         */
        int FindFeaturesInSecond();

        /**
         * Build the initial map with single image
         * @return true if succeed
         */
        bool BuildInitMap();

        /**
         * Triangulate the 2D points in current frame
         * @return num of triangulated points
         */
        int TriangulateNewPoints(Frame::Ptr &frame_one, Frame::Ptr &frame_two);

        /**
         * Set the features in keyframe as new observation of the map points
         */
        void SetObservationsForKeyFrame();

        // data
        ConstructionStatus status_ = ConstructionStatus::TRACKING_GOOD;

        Frame::Ptr current_frame_ = nullptr; // 当前帧, 这里也承载相机的职能算了
        Frame::Ptr last_frame_ = nullptr;    // 上一帧
        Camera::Ptr camera_ = nullptr;       // 当前相机

        SE3 relative_motion_; // 当前帧与上一帧的相对运动,用于估计当前帧pose初值

        int tracking_inliers_ = 0; // inliers, used for testing new keyframes
        Vec6 intrinsic_ = Vec6::Zero();
        Map::Ptr map_ = nullptr;
        vector<Frame::Ptr> frames_; // 所有的frame信息

        ImageLoader image_loader_;

        // params
        int num_features_ = 200;
        int num_features_tracking_ = 50;
        int num_features_tracking_bad_ = 20;
        int num_features_needed_for_keyframe_ = 80;
    };

} // namespace myslam

#endif // STEPS_H
