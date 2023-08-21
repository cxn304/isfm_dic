#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "dataset.h"
#include "inits.h"

using namespace std;
int main(int argc, char **argv)
{
    // step1 : 提取特征, 并存储到数据库
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    ISfM::ImageLoader Cimage_loader("./imgs/Viking/");
    ISfM::Dataset Cdates;
    // Cdates.establishDbo(Cimage_loader.filenames);
    string feature_path = "./features.yml";
    string filename_path = "./feature_name.txt";
    string matrix_path = "./similarityMatrix.yml";
    cv::Mat sMatrix = Cdates.readDateSet(matrix_path,feature_path,filename_path);// 读取数据库并建立Cdates
    // step2 : 提取特征, 并存储到数据库

    ISfM::Initializer Cinitializer(Cimage_loader);
    vector<cv::Point2f> pts1; // 得到返回值,
    vector<cv::Point2f> pts2;

    Cinitializer.featureMatching(sMatrix,Cdates.kpoints_,Cdates.descriptors_,pts1,pts2);
    Cinitializer.Initialize(pts1,pts2);



    int max_image_size = 3200;
    int num_features = 8024;
    int normalization_type = 0;

    return 0;
}