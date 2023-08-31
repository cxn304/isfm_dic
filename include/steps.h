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
#include "g2o_types.h"

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
     * 估计当前帧Pose,在满足关键帧条件时向地图加入关键帧并触发优化
     */
    class Steps
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Steps> Ptr;

        Steps(Initializer::Returns &returns, ImageLoader &Cimage_loader){}; // 构造函数

        /// 外部接口,添加一个帧(图像)并计算其定位结果
        bool AddFrame(Frame::Ptr frame);

        void SetMap(Map::Ptr map) { map_ = map; }

        ConstructionStatus GetStatus() const { return status_; }
        
        void SetCameras(Camera::Ptr camera_one)
        {
            camera_one_ = camera_one;
        }

        std::vector<Frame::Ptr> getFrames() const
        {
            return frames_;
        }
        
        void Optimize(Map::KeyframesType &keyframes,Map::LandmarksType &landmarks);

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
        int TriangulateNewPoints(Frame::Ptr &frame_one, Frame::Ptr &frame_two);

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

        SE3 relative_motion_; // 当前帧与上一帧的相对运动,用于估计当前帧pose初值

        int tracking_inliers_ = 0; // inliers, used for testing new keyframes
        Vec6 intrinsic_ = Vec6::Zero();
        Map::Ptr map_ = nullptr;
        vector<Frame::Ptr> frames_;                          // 所有的frame信息
        Frame::Ptr frameone_ = nullptr;                      // 选定的一帧,这里暂时第一第二帧,记住要初始化pose
        Frame::Ptr frametwo_ = nullptr;                      // 选定的另一帧,也传递到step里面
        vector<vector<Feature::Ptr>> features_;              // 转换后的 Feature 对象,2维点,要传递到step里面的
        map<pair<int, int>, vector<cv::DMatch>> matchesMap_; // 存储每对图像之间的匹配结果,传递到step里面
        ImageLoader image_loader_;

        // params
        int num_features_ = 200;
        int num_features_tracking_ = 50;
        int num_features_tracking_bad_ = 20;
        int num_features_needed_for_keyframe_ = 80;

    };

} // namespace myslam

#endif // RECONSTRUCTION_H
