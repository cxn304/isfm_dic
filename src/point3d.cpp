#include "point3d.h"

namespace ISfM
{
    Poind3d::Poind3d(long id, Vec3 position) : id_(id), pos_(position) {}

    Poind3d::Ptr Poind3d::CreateNewMappoint()
    {
        static long factory_id = 0;
        Poind3d::Ptr new_mappoint(new Poind3d);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

    void Poind3d::RemoveObservation(std::shared_ptr<Feature> feat)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        for (auto iter = observations_.begin(); iter != observations_.end();
             iter++)
        {
            if (iter->lock() == feat)
            {
                observations_.erase(iter);
                feat->map_point_.reset();
                observed_times_--;
                break;
            }
        }
    }
}
