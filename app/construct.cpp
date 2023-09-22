#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
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
    string smilar_matrix_path = "./data/similarityMatrix.yml";
    // step2 : 读取特征及数据库文件,读取匹配信息并存储到数据库
    cv::Mat sMatrix = Cdates.readDateSet(smilar_matrix_path, feature_path, filename_path, Cimage_loader.filenames_); // 读取数据库并建立Cdates
    Cdates.loadMatchesFromFile("./data/match_info.yml");
    // step3 : 初始化地图
    ISfM::Initializer Cinitializer(Cimage_loader, Cdates, sMatrix);
    ISfM::Initializer::Returns init_information = Cinitializer.Initialize();
    ISfM::Map::Ptr map_ = nullptr;
    map_ = init_information.map_; // 初始化地图点
    Camera::Ptr camera_one = std::make_shared<Camera>(init_information.K_.at<double>(0, 0),
                                                      init_information.K_.at<double>(1, 1), init_information.K_.at<double>(0, 2),
                                                      init_information.K_.at<double>(1, 2));
    // step4: 初始化step的所有成员变量
    ISfM::Steps::Ptr steps = std::make_shared<Steps>(init_information, Cimage_loader, camera_one); // 这里要补充构造函数
    steps->SetMap(map_);

    // step5: 开始优化
    auto new_frame = steps->getFrames()[init_information.id1];
    new_frame->SetKeyFrame();
    map_->InsertKeyFrame(new_frame);
    new_frame = steps->getFrames()[init_information.id2];
    new_frame->SetKeyFrame();
    map_->InsertKeyFrame(new_frame);
    int tmp_id1 = init_information.id1;
    int tmp_id2 = init_information.id2;
    for (int i = 0; i < Cimage_loader.filenames_.size(); ++i)
    {
        auto new_frame = steps->getFrames()[i]; // 从数据集中读出下一帧,不能是注册过的
        if (new_frame == nullptr || new_frame->is_registed == true)
            return false;
        std::pair pairi1 = make_pair(i,tmp_id1);
        std::pair pairi2 = make_pair(i,tmp_id2);
        std::pair pair1i = make_pair(tmp_id1,i);
        std::pair pair2i = make_pair(tmp_id2,i);
        auto iti1 = init_information.matchesMap_.find(pairi1);
        auto it1i = init_information.matchesMap_.find(pair1i);
        auto iti2 = init_information.matchesMap_.find(pairi2);
        auto it2i = init_information.matchesMap_.find(pair2i);

        bool success = false;
        if (iti1 != init_information.matchesMap_.end() || it1i != init_information.matchesMap_.end()){
            success = steps->AddFrame(new_frame); // 如果是id1和新的帧有关系,就将lastframe设置为id1
            steps->SetLastFrame(steps->getFrames()[tmp_id1]);
        }   
        else if (iti2 != init_information.matchesMap_.end() || it2i != init_information.matchesMap_.end()){
            success = steps->AddFrame(new_frame); // 与上面意思一样
            steps->SetLastFrame(steps->getFrames()[tmp_id2]);
        }
        if (success)
        {
            Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            steps->localBA(active_kfs, active_landmarks);
        }
    }
    Map::LandmarksType all_landmarks = map_->GetAllMapPoints();
    steps->gloabalBA(all_landmarks);

    SavePLY("./test/point.ply", all_landmarks);

    cout << "VO exit";
    return 0;
};
