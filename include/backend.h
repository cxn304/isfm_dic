#ifndef BACKEND_H
#define BACKEND_H

#include "common.h"
#include "frame.h"
#include "map.h"
#include "camera.h"

namespace ISfM {
class Map;
class Camera;

/**
 * 后端
 * 有单独优化线程，在Map更新时启动优化
 * Map更新由前端触发
 */ 
class Backend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    /// 构造函数中启动优化线程并挂起
    Backend();

    // 设置相机，用于获得内外参
    void SetCameras(Camera::Ptr cam) {
        cam_ = cam;
    }

    /// 设置地图
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    /// 触发地图更新，启动优化
    void UpdateMap();

    /// 关闭后端线程
    void Stop();

   private:
    /// 后端线程
    void BackendLoop();

    /// 对给定关键帧和路标点进行优化
    void localBA(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;//全局互斥锁

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H