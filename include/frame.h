#pragma once
#ifndef FRAME_H
#define FRAME_H
#include "common.h"

namespace ISfM
{

    // forward declare
    struct Poind3d;
    struct Feature;

    /**
     * 帧
     * 每一帧分配独立id，关键帧分配关键帧ID
     */
    struct Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned long id_ = 0;          // id of this frame
        unsigned long keyframe_id_ = 0; // id of key frame
        bool is_keyframe_ = false;      // 是否为关键帧
        SE3 pose_;                      // Tcw 形式Pose
        std::mutex pose_mutex_;         // Pose数据锁
        cv::Mat img_d;                  // images

        // extracted features in left image
        std::vector<std::shared_ptr<Feature>> features_img_;

    public: // data members
        Frame() {}

        Frame(long id, const SE3 &pose, const Mat &img_d);

        // set and get pose, thread safe
        SE3 Pose()
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(const SE3 &pose)
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        /// 设置关键帧并分配并键帧id
        void SetKeyFrame();

        /// 工厂构建模式，分配id
        static std::shared_ptr<Frame> CreateFrame();
    };

}
#endif
