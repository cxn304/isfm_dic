#pragma once
#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <opencv2/features2d.hpp>
#include <ceres/ceres.h>
#include "common.h"
#include "frame.h"
#include "map.h"
#include "inits.h"
#include "camera.h"
#include "feature.h"
#include "triangulate.h"
#include "SnavelyReprojectionError.h"

namespace ISfM
{
    enum class ConstructionStatus
    {
        INITING,
        TRACKING_GOOD,
        TRACKING_BAD,
        LOST
    };

    /**
     * 优化过程
     * 估计当前帧Pose，在满足关键帧条件时向地图加入关键帧并触发优化
     */
    class Steps
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Steps> Ptr;

        Steps();

        /// 外部接口，添加一个帧(图像)并计算其定位结果
        bool AddFrame(Frame::Ptr frame);

        /// Set函数
        void SetMap(Map::Ptr map) { map_ = map; }

        ConstructionStatus GetStatus() const { return status_; }
        void SetCameras(Camera::Ptr camera_one, Camera::Ptr camera_two) {
        camera_one_ = camera_one;
        camera_two_ = camera_two;
    }

    private:
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
         * Track with last frame
         * @return num of tracked points
         */
        int TrackLastFrame();

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
         * Try init the frontend with stereo images saved in current_frame_
         * @return true if success
         */
        bool Init(Initializer initialize_);

        /**
         * Detect features in left image in current_frame_
         * keypoints will be saved in current_frame_
         * @return
         */
        int DetectFeatures();

        /**
         * Find the corresponding features in right image of current_frame_
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
        int TriangulateNewPoints();

        /**
         * Set the features in keyframe as new observation of the map points
         */
        void SetObservationsForKeyFrame();

        void UpdateMap();

        // data
        ConstructionStatus status_ = ConstructionStatus::INITING;

        Frame::Ptr current_frame_ = nullptr; // 当前帧, 这里也承载相机的职能算了
        Frame::Ptr last_frame_ = nullptr;    // 上一帧
        Camera::Ptr camera_one_ = nullptr;  // 当前相机
        Camera::Ptr camera_two_ = nullptr;  // 当前相机2

        Map::Ptr map_ = nullptr;

        SE3 relative_motion_; // 当前帧与上一帧的相对运动，用于估计当前帧pose初值

        int tracking_inliers_ = 0; // inliers, used for testing new keyframes

        // params
        int num_features_ = 200;
        int num_features_init_ = 100;
        int num_features_tracking_ = 50;
        int num_features_tracking_bad_ = 20;
        int num_features_needed_for_keyframe_ = 80;

        // utilities
        cv::Ptr<cv::GFTTDetector> gftt_; // feature detector in opencv
    };

} // namespace myslam

#endif // RECONSTRUCTION_H
