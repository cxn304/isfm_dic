#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "dataset.h"
#include "inits.h"
#include "map.h"
#include "steps.h"

using namespace ISfM;
using namespace std;

int main(int argc, char **argv)
{
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    ISfM::ImageLoader Cimage_loader("./imgs/Viking/");
    ISfM::Dataset Cdates;
    string feature_path = "./data/features.yml";
    string filename_path = "./data/feature_name.txt";
    string matrix_path = "./data/similarityMatrix.yml";
    // step2 : 读取特征及数据库文件,读取匹配信息并存储到数据库
    cv::Mat sMatrix = Cdates.readDateSet(matrix_path, feature_path, filename_path, Cimage_loader.filenames_); // 读取数据库并建立Cdates
    Cdates.loadMatchesFromFile("./data/match_info.yml");
    // step3 : 初始化地图
    ISfM::Initializer Cinitializer(Cimage_loader, Cdates);
    ISfM::Initializer::Returns init_information = Cinitializer.Initialize();
    ISfM::Map::Ptr map_ = nullptr;
    map_ = init_information.map_; // 初始化地图点
    Camera::Ptr camera_one = std::make_shared<Camera>(init_information.K_.at<double>(0, 0),
                                                      init_information.K_.at<double>(1, 1), init_information.K_.at<double>(0, 2),
                                                      init_information.K_.at<double>(1, 2));
    // step4: 初始化step的所有成员变量
    ISfM::Steps::Ptr steps = std::make_shared<Steps>(init_information); // 这里要补充构造函数
    steps->SetMap(map_);
    steps->SetCameras(camera_one);
    for (int i = 2; i < Cimage_loader.filenames_.size(); ++i)
    {
        if (i < 2)
        {
            auto new_frame = steps->getFrames()[i]; // 从数据集中读出下一帧
            new_frame->SetKeyFrame();
            map_->InsertKeyFrame(new_frame);
        }
        else
        {
            auto new_frame = steps->getFrames()[i]; // 从数据集中读出下一帧
            if (new_frame == nullptr)
                return false;                          // 这个数据集跑完了，没有下一帧了
            bool success = steps->AddFrame(new_frame); // 将新的一帧加入到前端中，进行跟踪处理,帧间位姿估计
            if (success)
            {
                Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
                Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
                steps->Optimize(active_kfs, active_landmarks);
            }
        }
    }

    return 0;
}
