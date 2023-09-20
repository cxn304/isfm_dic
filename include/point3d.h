#pragma once
#ifndef POINT3D_H
#define POINT3D_H
#include "common.h"
#include "feature.h"

namespace ISfM
{
    struct Feature;
    struct MapPoint
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_ = 0; // ID
        bool is_outlier_ = false;
        cv::Vec3d pos_ ; // Position in world
        Eigen::Matrix<uchar, 3, 1> color_;

        std::mutex data_mutex_;
        int observed_times_ = 0; // being observed by feature matching algo.
        std::list<std::weak_ptr<ISfM::Feature>> observations_; //observations_是存储feature的list

        MapPoint() {}

        MapPoint(long id, cv::Vec3d position);

        cv::Vec3d Pos()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos(const cv::Vec3d &pos)
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        };

        Eigen::Matrix<uchar, 3, 1> Color()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return color_;
        }

        void SetColor(const Eigen::Matrix<uchar, 3, 1> &color)
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            color_ = color;
        };

        void AddObservation(std::shared_ptr<Feature> feature)
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);
            observed_times_++;
        }

        void RemoveObservation(std::shared_ptr<Feature> feat);

        std::list<std::weak_ptr<Feature>> GetObs()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };
}

#endif
