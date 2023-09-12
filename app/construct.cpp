#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
// #include <pcl/io/ply_io.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
#include "common.h"
#include "dataset.h"
#include "inits.h"
#include "map.h"
#include "steps.h"

using namespace ISfM;
using namespace std;

void drawCameraPose(cv::Mat& image, const Sophus::SE3d& cameraPose, double scale = 1.0)
{
    Eigen::Vector3d cameraPosition = cameraPose.translation() * 200.0;
    Eigen::Matrix3d cameraRotation = cameraPose.rotationMatrix();

    Eigen::Vector3d cameraDirection = cameraRotation * Eigen::Vector3d(0, 0, 1);
    Eigen::Vector3d cameraTarget = cameraPosition + cameraDirection * scale;

    cv::arrowedLine(image, cv::Point(cameraPosition.x(), cameraPosition.y()), cv::Point(cameraTarget.x(), cameraTarget.y()), cv::Scalar(0, 0, 255), 2);
    cv::circle(image, cv::Point(cameraPosition.x(), cameraPosition.y()), 4, cv::Scalar(0, 0, 255), -1);
}

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
    ISfM::Steps::Ptr steps = std::make_shared<Steps>(init_information,Cimage_loader,camera_one); // 这里要补充构造函数
    steps->SetMap(map_);
    
    // step5: 开始优化
    for (int i = 0; i < Cimage_loader.filenames_.size(); ++i)
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

    cv::Mat pose_image(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    for(int i = 0; i < Cimage_loader.filenames_.size(); ++i){
        auto new_frame = steps->getFrames()[i];
        SE3 c_pose = new_frame->Pose();
        drawCameraPose(pose_image, c_pose);
    }
    std::string filename = "./test/camera_poses.jpg";
    cv::imwrite(filename, pose_image);
    cout << "VO exit";
    return 0;
};



// 差个空,输出pcl点云供查看
    // typedef pcl::PointXYZ PointType;
    // Map::LandmarksType landmarked = map_->GetAllMapPoints();
    // pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    // for (auto &landmark : landmarked){
    //    cv::Vec3d pos = landmark.second->pos_;
    //    PointType pclPoint;
    //    pclPoint.x = pos[0];
    //    pclPoint.y = pos[1];
    //    pclPoint.z = pos[2];
    //    cloud->points.push_back(pclPoint);
    // }
    // pcl::io::savePLYFile("./tmp/point_cloud.ply", *cloud, true); // 存储点云到 PCL 文件