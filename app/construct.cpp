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

// 第一个值是last_frm_id,第二个值是current_frm_id, is_registed是last
vector<int> findReturnIndex(const vector<pair<pair<int, int>, vector<cv::DMatch>>> &matchMap,
                            const vector<Frame::Ptr> &frames_)
{
    vector<int> return_idx;
    for (const auto &pairs : matchMap)
    {
        // 获取 pairs<int, int> 键部分
        const std::pair<int, int> &key = pairs.first;
        int id1 = key.first;
        int id2 = key.second;
        if (frames_[id1]->is_registed && !frames_[id2]->is_registed)
        {
            return_idx.push_back(id1);
            return_idx.push_back(id2);
            return return_idx;
        }
        else if (!frames_[id1]->is_registed && frames_[id2]->is_registed)
        {
            return_idx.push_back(id2);
            return_idx.push_back(id1);
            return return_idx;
        }
    }
    return return_idx;
};
// 查看是不是所有的图片都已经被注册
bool isImageAllAdded(vector<pair<pair<int, int>, vector<cv::DMatch>>> &matchMap,
                     const vector<Frame::Ptr> &frames_)
{
    for (const auto &pairs : matchMap)
    {
        std::pair<int, int> key = pairs.first;
        int id1 = key.first;
        int id2 = key.second;
        if (!frames_[id1]->is_registed || !frames_[id2]->is_registed)
            return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    cout << "OpenCV version: " << CV_VERSION << endl;
    ISfM::ImageLoader Cimage_loader("./imgs/Viking/");
    ISfM::Dataset::Ptr Cdates = make_shared<Dataset>();
    string feature_path = "./data/features.yml";
    string smilar_matrix_path = "./data/similarityMatrix.yml";
    // step2 : 读取特征及数据库文件,读取匹配信息并存储到数据库
    Cdates->readDateSet(smilar_matrix_path, feature_path, Cimage_loader.filenames_); // 读取数据库并建立Cdates
    Cdates->loadMatchesFromFile("./data/match_info.yml");
    // step3 : 初始化地图
    ISfM::Initializer Cinitializer(Cimage_loader, Cdates, Cdates->similarityMatrix_);
    ISfM::Initializer::Returns init_information = Cinitializer.Initialize();
    ISfM::Map::Ptr map_ = nullptr;
    map_ = init_information.map_; // 初始化地图点
    Camera::Ptr camera_one = make_shared<Camera>(init_information.K_.at<double>(0, 0),
                                                      init_information.K_.at<double>(1, 1), init_information.K_.at<double>(0, 2),
                                                      init_information.K_.at<double>(1, 2));
    // step4: 初始化step的所有成员变量
    ISfM::Steps::Ptr steps = make_shared<Steps>(init_information, Cimage_loader, camera_one); // 这里要补充构造函数
    steps->SetMap(map_);

    // step5: 开始优化
    auto new_frame = steps->getFrames()[init_information.id1];
    new_frame->SetKeyFrame();
    map_->InsertKeyFrame(new_frame);
    new_frame = steps->getFrames()[init_information.id2];
    new_frame->SetKeyFrame();
    map_->InsertKeyFrame(new_frame);
    int last_id;
    int current_id;
    while (!isImageAllAdded(steps->matchesMap_, steps->getFrames()))
    {
        vector<int> idx_last_current = findReturnIndex(steps->matchesMap_, steps->getFrames());
        if (idx_last_current.size() == 2)
        {
            last_id = idx_last_current[0];
            current_id = idx_last_current[1];
            cout << " last_id: " << last_id << " current_id: " << current_id << endl;
            auto new_frame = steps->getFrames()[current_id];
            if (new_frame == nullptr || new_frame->is_registed == true)
                continue;
            steps->SetLastFrame(steps->getFrames()[last_id]);
            steps->AddFrame(new_frame);
            new_frame->is_registed = true; // 将其注册
            Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            steps->localBA(active_kfs, active_landmarks);
        }
        else
        {
            cout << "no more connected image!";
            break;
        }
    }

    Map::LandmarksType all_landmarks = map_->GetAllMapPoints();
    steps->gloabalBA(all_landmarks);

    ISfM::SavePLY("./test/point.ply", all_landmarks);

    cout << "VO exit";
    return 0;
};
