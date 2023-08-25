#include "dataset.h"
#include <boost/format.hpp>

using namespace std;
namespace ISfM
{
    // extract all orbsift in all images, then save it
    void Dataset::establishDbo(const vector<string> &file_paths)
    {
        // 找到当前路径
        char cwd[PATH_MAX];
        getcwd(cwd, sizeof(cwd));
        string current_dir(cwd);
        // 创建ORB特征提取器对象
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
        // 创建文件存储对象,包括features和descriptor,文件名
        cv::FileStorage fs(current_dir + "/data/features.yml", cv::FileStorage::WRITE);
        std::ofstream file("./data/feature_name.txt"); // 打开一个输出文件流
        // 创建词袋模型存储mat
        vector<cv::Mat> descriptors;
        // 加载第一张图像
        cv::Mat image = cv::imread(file_paths[0], cv::IMREAD_GRAYSCALE);
        int width = image.cols;
        int height = image.rows;
        // 遍历图片文件
        int img_id = 0;
        for (const string &file_path : file_paths)
        {
            // 加载图像
            cv::Mat image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            cv::Mat mask(image.size(), CV_8UC1, 255);
            if (image.empty())
            {
                cerr << "Failed to read image: " << file_path << endl;
                continue;
            }
            // 检测图像中的关键点
            vector<cv::KeyPoint> kpts;
            // 计算关键点的描述符
            cv::Mat descriptor;
            int cnt = DetectFeatures(image, kpts, descriptor);
            // cv::drawKeypoints(image, kpts, image, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            // cv::imwrite("keypoints_with_descriptors.jpg", image);
            // 将关键点和描述符保存到文件
            // 提取文件名
            size_t lastSlash = file_path.find_last_of("/\\");           // 查找路径中最后一个斜杠或反斜杠的位置
            string filename = "img_" + file_path.substr(lastSlash + 1); // 提取最后一个斜杠之后的部分
            replace(filename.begin(), filename.end(), '.', 'd');
            fs << filename << "{"
               << "imd_id" << img_id
               << "file_path" << file_path
               << "kpts" << kpts
               << "descriptor" << descriptor
               << "}";
            if (file.is_open())
            {
                file << filename << '\n';
            }
            descriptors.push_back(descriptor);
            features_[filename] = make_pair(kpts, descriptor);
            file_paths_[img_id] = file_path;
            img_id++;
        }
        // 关闭文件存储
        fs.release();
        file.close();
        // save vocab data base
        DBoW3::Vocabulary vocab;
        vocab.create(descriptors);
        // cout << "vocabulary info: " << vocab << endl;
        string file_path = current_dir + "/data/vocab_larger.yml.gz";
        vocab.save(file_path);
        saveSimilarMatrix(vocab, "./data/features.yml", "./data/feature_name.txt");
        cout << "features detect done" << endl;
    }

    int Dataset::DetectFeatures(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
    {
        cv::Mat mask(image.size(), CV_8UC1, 255);
        // 创建 ORB 对象
        cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
        // 检测特征点
        orb->detect(image, keypoints);
        // 计算特征描述符
        orb->compute(image, keypoints, descriptors);
        // 设置特征点之间的最小距离阈值
        float minDistancePixels = 1.0f;
        // 迭代遍历特征点，根据距离阈值筛选特征点
        std::vector<cv::KeyPoint> filteredKeypoints;
        vector<int> extracted_id;
        for (int i = 0; i < keypoints.size(); ++i)
        {
            const auto &keypoint = keypoints[i];
            bool keepKeypoint = true;
            for (int j = 0; j < keypoints.size(); ++j)
            {
                if (i == j)
                    continue;
                const auto &otherKeypoint = keypoints[j];
                float distance = cv::norm(keypoint.pt - otherKeypoint.pt);

                if (distance < minDistancePixels)
                {
                    keepKeypoint = false;
                    break;
                }
            }
            if (keepKeypoint)
            {
                filteredKeypoints.push_back(keypoint);
                extracted_id.push_back(i);
            }
        }
        cv::Mat filteredDescriptors(extracted_id.size(), descriptors.cols, CV_32F);
        for (int i = 0; i < extracted_id.size(); ++i)
        {
            // 将每个描述符复制到矩阵中的相应位置
            cv::Mat descript = descriptors.row(extracted_id[i]); // 获取描述符
            descript.copyTo(filteredDescriptors.row(i));
        }

        // 将筛选后的特征点和描述符赋值给原始变量
        keypoints = filteredKeypoints;
        descriptors = filteredDescriptors;
        return keypoints.size();
    }

    // using DBoW3 to find similar image, for initialization
    void Dataset::saveSimilarMatrix(DBoW3::Vocabulary &vocab,
                                    const string &feature_path, const string &filename_path)
    {
        readImageSave(feature_path, filename_path);
        if (vocab.empty())
        {
            cerr << "Vocabulary does not exist." << endl;
        }
        cv::Mat similarityMatrix;
        similarityMatrix = cv::Mat::zeros(filenames_.size(), filenames_.size(), CV_32FC1);
        cout << "comparing images with images " << endl;
        for (int i = 0; i < filenames_.size(); i++)
        {
            DBoW3::BowVector v1;
            vocab.transform(descriptors_[i], v1);
            for (int j = i + 1; j < filenames_.size(); j++)
            {
                DBoW3::BowVector v2;
                vocab.transform(descriptors_[j], v2);
                float score = vocab.score(v1, v2);
                if (score > 0.5)
                    score = 0.0;
                similarityMatrix.at<float>(i, j) = score;
            }
        }
        // similarityMatrix_ = similarityMatrix; // 所有图片的相似性矩阵
        cv::FileStorage file("./data/similarityMatrix.yml", cv::FileStorage::WRITE);
        file << "matrix" << similarityMatrix << "size" << similarityMatrix.cols;
        file.release();
    };
    ////////////////////////////////////////////////////////////////////////////////////
    // using orb nums to find similar image, for initialization
    void Dataset::saveORBSimilar(const string feature_path, const string filename_path)
    {
        Eigen::MatrixXd ORBSimilar = Eigen::MatrixXd::Zero(filenames_.size(), filenames_.size());
        // 创建特征匹配器
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        cout << "comparing images with images " << endl;
        for (int i = 0; i < filenames_.size(); i++)
        {
            for (int j = i + 1; j < filenames_.size(); j++)
            {
                vector<cv::DMatch> matches;
                // 特征匹配
                matcher.match(descriptors_[i], descriptors_[j], matches);
                // 统计匹配点数量
                int numMatches = matches.size();
                ORBSimilar(i, j) = numMatches;
            }
        }
        // cout << similarityMatrix.matrix();
        ofstream file("ORBSimilar.txt");
        if (file.is_open())
        {
            // 将矩阵的行和列写入文件
            file << ORBSimilar.rows() << " " << ORBSimilar.cols() << endl;
            // 逐行写入矩阵元素
            for (int i = 0; i < ORBSimilar.rows(); ++i)
            {
                for (int j = 0; j < ORBSimilar.cols(); ++j)
                {
                    file << ORBSimilar(i, j) << " ";
                }
                file << endl;
            }
            file.close();
            cout << "Matrix saved to file." << endl;
        }
        else
        {
            cerr << "Failed to open file." << endl;
        }
    };
    ////////////////////////////////////////////////////////////////////////////////////
    void Dataset::readImageSave(const string feature_path, const string filename_path)
    {
        ifstream file(filename_path); // 打开一个输入文件流
        if (filenames_.empty())
        {
            if (file.is_open())
            {
                string line;
                while (getline(file, line))
                {
                    if (line.substr(line.length() - 3) == "jpg" || line.substr(line.length() - 3) == "png")
                    {
                        filenames_.push_back(line); // 存储以 ".jpg" 或 ".png" 结尾的行
                    }
                }
                file.close(); // 关闭文件流
            }
            else
            {
                cout << "can't open file" << endl;
            }
            cout << "reading database" << endl;
            cv::FileStorage fs(feature_path, cv::FileStorage::READ);
            if (!fs.isOpened())
            {
                cout << "can't open file" << feature_path << endl;
            }
            for (auto &fms : filenames_)
        {
            cv::FileNode kptsNode = fs[fms]["kpts"];
            vector<cv::KeyPoint> kpts;
            for (const auto &kptNode : kptsNode)
            {
                cv::KeyPoint kpt;
                kptNode >> kpt;
                kpts.push_back(kpt);
            }
            kpoints_.push_back(kpts); // 这里要修改,kpts是一个point2d类才行
            cv::FileNode descriptorNode = fs[fms]["descriptor"];
            cv::Mat descriptor;
            descriptorNode >> descriptor;
            descriptors_.push_back(descriptor);
        }
        fs.release();
        }
    }
    ///////////////////////////////建立好数据库后,再read//////////////////////////////////////////
    cv::Mat Dataset::readDateSet(const string &matrixPath, const string &feature_path, const string &filename_path,
                                 const vector<string> &filenames)
    {
        readImageSave(feature_path, filename_path);
        for (int i = 0; i < filenames.size(); i++)
        {
            file_paths_.insert(std::make_pair(i, filenames[i]));
        }

        cv::FileStorage fs(matrixPath, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            cout << "can't open file" << matrixPath << endl;
        }
        cv::Mat sMatrix;
        fs["matrix"] >> sMatrix;
        fs.release();
        return sMatrix;
    };
    ////////////////////////////计算并储存matches,需要先readDateSet///////////////////////////////////////
    void Dataset::computeAndSaveMatches()
    {
        // 对每对相邻图像进行特征点匹配
        for (int i = 0; i < file_paths_.size() - 1; ++i)
        {
            std::vector<cv::DMatch> matches;
            ComputeMatches(descriptors_[i], descriptors_[i + 1], matches, 0.8);
            // 构建图像对
            std::pair<int, int> imagePair(i, i + 1);
            matchesMap_[imagePair] = matches;
        }
        string filename = "./data/match_info.yml";
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);

        if (!fs.isOpened())
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        fs << "matches"
           << "[";
        for (const auto &pair : matchesMap_)
        {
            fs << "{";
            fs << "imagePair"
               << "[" << pair.first.first << pair.first.second << "]";
            fs << "matches"
               << "[";
            for (const cv::DMatch &match : pair.second)
            {
                fs << "{";
                fs << "queryIdx" << match.queryIdx;
                fs << "trainIdx" << match.trainIdx;
                fs << "}";
            }
            fs << "]";
            fs << "}";
        }
        fs << "]";
        fs.release();
    };

    void Dataset::ComputeMatches(const cv::Mat &desc1,
                                 const cv::Mat &desc2,
                                 std::vector<cv::DMatch> &matches,
                                 const float distance_ratio)
    {
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
        std::vector<std::vector<cv::DMatch>> initial_matches;

        matcher->knnMatch(desc1, desc2, initial_matches, 2);
        for (auto &m : initial_matches)
        {
            if (m[0].distance < distance_ratio * m[1].distance)
            {
                matches.push_back(m[0]);
            }
        }
    }

    ///////////////////////////将match读取储存到Dataset的matchesMap_当中///////////////////////////////////
    void Dataset::loadMatchesFromFile(const std::string &filename)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);

        if (!fs.isOpened())
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        cv::FileNode matchesNode = fs["matches"];
        for (cv::FileNodeIterator it = matchesNode.begin(); it != matchesNode.end(); ++it)
        {
            cv::FileNode matchNode = *it;
            std::pair<int, int> imagePair;
            std::vector<cv::DMatch> matches;

            imagePair.first = (int)matchNode["imagePair"][0];
            imagePair.second = (int)matchNode["imagePair"][1];

            cv::FileNode matchesListNode = matchNode["matches"];
            for (cv::FileNodeIterator it2 = matchesListNode.begin(); it2 != matchesListNode.end(); ++it2)
            {
                cv::FileNode matchObjNode = *it2;
                cv::DMatch match;
                matchObjNode["queryIdx"] >> match.queryIdx;
                matchObjNode["trainIdx"] >> match.trainIdx;
                matches.push_back(match);
            }

            matchesMap_[imagePair] = matches;
        }

        fs.release();
    }
}