#include "map.h"

namespace ISfM
{
    // 插入关键帧
    void Map::InsertKeyFrame(Frame::Ptr frame)
    {
        current_frame_ = frame; // 将传入的帧设置为当前帧

        // 如果关键帧的ID不存在于keyframes_中，则插入关键帧到keyframes_和active_keyframes_中
        if (keyframes_.find(frame->keyframe_id_) == keyframes_.end())
        {
            keyframes_.insert(make_pair(frame->keyframe_id_, frame));
            active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        }
        else
        {
            // 如果关键帧的ID已经存在于keyframes_中，则更新对应的关键帧
            keyframes_[frame->keyframe_id_] = frame;
            active_keyframes_[frame->keyframe_id_] = frame;
        }

        // 如果active_keyframes_的大小超过了num_active_keyframes_，则删除最旧的关键帧
        if (active_keyframes_.size() > num_active_keyframes_)
        {
            RemoveOldKeyframe();
        }
    }

    // 插入地图点
    void Map::InsertMapPoint(MapPoint::Ptr map_point)
    {
        // 如果地图点的ID不存在于landmarks_中，则插入地图点到landmarks_和active_landmarks_中
        if (landmarks_.find(map_point->id_) == landmarks_.end())
        {
            landmarks_.insert(make_pair(map_point->id_, map_point));
            active_landmarks_.insert(make_pair(map_point->id_, map_point));
        }
        else
        {
            // 如果地图点的ID已经存在于landmarks_中，则更新对应的地图点
            landmarks_[map_point->id_] = map_point;
            active_landmarks_[map_point->id_] = map_point;
        }
    }

    // 删除最旧的关键帧
    void Map::RemoveOldKeyframe()
    {
        if (current_frame_ == nullptr)
            return;

        // 寻找与当前帧最近与最远的两个关键帧
        double max_dis = 0, min_dis = 9999;
        double max_kf_id = 0, min_kf_id = 0;
        auto Twc = current_frame_->Pose().inverse(); // 当前帧的位姿的逆
        for (auto &kf : active_keyframes_)
        {
            if (kf.second == current_frame_)
                continue;

            auto dis = (kf.second->Pose() * Twc).log().norm(); // 计算当前帧与其他关键帧的相对位姿差
            if (dis > max_dis)
            {
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis)
            {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }

        const double min_dis_th = 0.2; // 最近阈值
        Frame::Ptr frame_to_remove = nullptr;
        if (min_dis < min_dis_th)
        {
            // 如果存在很近的帧,优先删除最近的关键帧
            frame_to_remove = keyframes_.at(min_kf_id);
        }
        else
        {
            // 删除最远的关键帧
            frame_to_remove = keyframes_.at(max_kf_id);
        }

        cout << "remove keyframe " << frame_to_remove->keyframe_id_;

        // 删除关键帧及其对应的地图点观测
        active_keyframes_.erase(frame_to_remove->keyframe_id_);
        for (auto feat : frame_to_remove->features_img_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
            {
                mp->RemoveObservation(feat);
            }
        }

        CleanMap(); // 清理地图
    }

    // 清理地图中不再被观测到的地图点
    void Map::CleanMap()
    {
        int cnt_landmark_removed = 0;
        for (auto iter = active_landmarks_.begin(); iter != active_landmarks_.end();)
        {
            if (iter->second->observed_times_ == 0)
            {
                iter = active_landmarks_.erase(iter);
                cnt_landmark_removed++;
            }
            else
            {
                ++iter;
            }
        }
        cout << "Removed " << cnt_landmark_removed << " active landmarks";
    }
}