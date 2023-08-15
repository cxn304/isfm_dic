#include "DBoW3/DBoW3.h"
#include "files.hpp"
#include "common.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

class Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset() {}
    void establishDbo(const std::vector<std::string> &filenames);
    const std::map<std::string, std::pair<std::vector<cv::KeyPoint>,
                                          cv::Mat>> &
    getFeatures() const { return features_; } // image name and their kpts, des
    getDboScore() const { return dbo_score_; }
    void setDboScore(const cv::Mat& score) { dbo_score_ = score; }
    DBoW3::Vocabulary vocab;
    std::string vocab_file_path;
    std::vector<cv::Mat> descriptors_; // descriptor vectors
private:
    std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> features_;
    cv::Mat dbo_score_;
};