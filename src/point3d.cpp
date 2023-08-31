#include "point3d.h"

namespace ISfM
{
    // MapPoint类的构造函数
    MapPoint::MapPoint(long id, cv::Vec3d position) : id_(id), pos_(position) {}

    // 创建新的MapPoint对象
    MapPoint::Ptr MapPoint::CreateNewMappoint()
    {
        // 使用静态变量 factory_id 记录创建的MapPoint对象的数量
        static long factory_id = 0;
        // 创建新的MapPoint对象
        MapPoint::Ptr new_mappoint(new MapPoint);
        // 为新的MapPoint对象设置唯一的id
        new_mappoint->id_ = factory_id++;
        // 返回新创建的MapPoint对象的智能指针
        return new_mappoint;
    }

    // 移除观测关系
    void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat)
    {
        // 上锁以保护数据一致性
        std::unique_lock<std::mutex> lck(data_mutex_);
        for (auto iter = observations_.begin(); iter != observations_.end(); iter++)
        {
            // 判断观测关系是否与给定特征 feat 相关联
            if (iter->lock() == feat)
            {
                // 从列表中移除该观测关系
                observations_.erase(iter);
                // 重置特征的 map_point_ 指针为 nullptr
                feat->map_point_.reset();
                // 减少 observed_times_ 的计数
                observed_times_--;
                // 找到匹配的观测关系后，跳出循环
                break;
            }
        }
    }
}