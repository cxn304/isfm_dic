#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "dataset.h"

using namespace std;
int main(int argc, char **argv)
{
    // step1 : 提取特征, 并存储到数据库
    ISfM::ImageLoader image_loader("./imgs/Viking/");
    ISfM::Dataset feature;
    // feature.establishDbo(image_loader.filenames);
    string vocab_file_path = "./vocab_larger.yml.gz";
    string feature_path = "./features.yml";
    string filename_path = "./feature_name.txt";
    feature.findImageSimilar(vocab_file_path,feature_path,filename_path);


    int max_image_size = 3200;
    int num_features = 8024;
    int normalization_type = 0;

    return 0;
}