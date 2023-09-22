#pragma once
#ifndef FRAME_H
#define FRAME_H
#include "common.h"

using namespace std;
namespace ISfM
{

    // forward declare
    struct MapPoint;
    struct Feature;

    /**
     * 帧
     * 每一帧分配独立id,关键帧分配关键帧ID
     * 这里的frame在sfm里面就算是image
     */
    struct Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned id_ = 0;          // id of this frame
        unsigned keyframe_id_ = 0; // id of key frame
        bool is_keyframe_ = false;      // 是否为关键帧
        SE3 pose_;                      // Tcw 形式Pose, world to camera
        std::mutex pose_mutex_;          // Pose数据锁
        Vec6 pose_vector = pose_.log(); // 将 pose_ 转换为向量形式,6维向量
        Vec6 intrix_; //fx_,fy_,cx_,cy_,k1_,k2_,暂时内参都设置为一样的
        string img_name;
        bool is_registed = false;

        // features in this image
        std::vector<std::shared_ptr<Feature>> features_img_;

    public: // data members
        Frame() {}
        Frame(long id, const SE3 &pose);

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

        /// 设置关键帧并分配键帧id
        void SetKeyFrame();

        /// 工厂构建模式,分配id
        static std::shared_ptr<Frame> CreateFrame();
    };

}
#endif
