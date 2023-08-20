#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "dataset.h"
#include "inits.h"

using namespace std;
int main(int argc, char **argv)
{
    // step1 : 提取特征, 并存储到数据库
    ISfM::ImageLoader Cimage_loader("./imgs/Viking/");
    ISfM::Dataset Cdates;
    // feature.establishDbo(image_loader.filenames);
    string vocab_file_path = "./vocab_larger.yml.gz";
    string feature_path = "./features.yml";
    string filename_path = "./feature_name.txt";
    // Cdates.saveSimilarMatrix(vocab_file_path,feature_path,filename_path);
    string matrix_path = "./similarityMatrix.yml";
    Cdates.readImageSave(feature_path, filename_path); // 将数据读取到Cdates类
    cv::Mat sMatrix = Cdates.readSimilarMatrix(matrix_path,feature_path,filename_path);
    // step2 : 提取特征, 并存储到数据库

    ISfM::Initializer Cinitializer;
    vector<cv::Point2f> pts1; // 得到返回值,
    vector<cv::Point2f> pts2;

    Cinitializer.featureMatching(sMatrix,Cdates.kpoints_,Cdates.descriptors_,pts1,pts2);
    Cinitializer.Initialize(pts1,pts2);

    int max_image_size = 3200;
    int num_features = 8024;
    int normalization_type = 0;

    return 0;
}