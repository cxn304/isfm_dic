#include "frame.h"

namespace ISfM
{

    Frame::Frame(long id, const SE3 &pose, const Mat &img)
        : id_(id), pose_(pose), img_d(img) {}

    Frame::Ptr Frame::CreateFrame()
    {
        static long factory_id = 0;
        Frame::Ptr new_frame(new Frame);
        new_frame->id_ = factory_id++;
        return new_frame;
    }

    void Frame::SetKeyFrame()
    {
        static long keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }

}