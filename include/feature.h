#pragma once
#ifndef FEATURE_H
#define FEATURE_H
#include <memory>
#include <opencv2/features2d.hpp>
#include "common.h"

namespace ISfM
{
    struct Frame;
    struct MapPoint;
    /**
     * 2D 特征点
     * 在三角化之后会被关联一个地图点
     */
    struct Feature
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Feature> Ptr;

        int img_id_;        // 持有该feature的image
        cv::KeyPoint position_;             // 2D提取位置
        std::weak_ptr<MapPoint> map_point_; // 关联地图点
        std::weak_ptr<Frame> frame_;         // 持有该feature的frame

        bool is_outlier_ = false;      // 是否为异常点
        

    public:
        Feature() {}
        Feature(int &img_id, const cv::KeyPoint &kp)
            : img_id_(img_id), position_(kp) {}
        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
        // 静态函数直接调用
        static std::vector<cv::Point2f> convertFeaturesToPoints(const std::vector<Feature::Ptr> &features)
        {
            std::vector<cv::Point2f> points;
            points.reserve(features.size());

            for (const auto &feature : features)
            {
                const cv::Point2f point(feature->position_.pt.x, feature->position_.pt.y);
                points.push_back(point);
            }

            return points;
        };
    };
}
#endif
