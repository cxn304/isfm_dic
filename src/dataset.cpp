#include "dataset.h"

using namespace std;
namespace ISfM
{
    Dataset::Dataset(const ImageLoader &image_loader)
    {
        least_match_num_ = image_loader.leastMatchNum_;
        img_file_path_ = image_loader.dataset_dir_;
        boarder_size_ = image_loader.board_size_;
    };
    // extract all orbsift in all images, then save it
    void Dataset::establishDbo(const vector<string> &file_paths)
    {
        // 找到当前路径
        char cwd[PATH_MAX];
        getcwd(cwd, sizeof(cwd));
        string current_dir(cwd);
        cv::FileStorage fs("./data/features.yml", cv::FileStorage::WRITE);
        // 创建描述符
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
            // 提取文件名
            int lastSlash = file_path.find_last_of("/\\"); // 查找路径中最后一个斜杠或反斜杠的位置
            int lastDotPos = file_path.find_last_of('.');
            string filename = "img_" + file_path.substr(lastSlash + 1, lastDotPos - lastSlash - 1); // 提取最后一个斜杠之后的部分
            fs << filename << "{"
               << "imd_id" << img_id
               << "file_path" << file_path
               << "kpts" << kpts
               << "descriptor" << descriptor
               << "}";
            descriptors.push_back(descriptor);
            file_paths_[img_id] = file_path;
            img_id++;
        }
        // 关闭文件存储
        fs.release();
        // save vocab data base
        // DBoW3::Vocabulary vocab;
        // vocab.create(descriptors);
        // cout << "vocabulary info: " << vocab << endl;
        // string file_path = current_dir + "/data/vocab_larger.yml.gz";
        // vocab.save(file_path);
        // saveSimilarMatrix(vocab, "./data/features.yml", "./data/feature_name.txt");
        std::cout << "features detect done" << endl;
    }
    // 主要是计算棋盘格的数据**************************
    void Dataset::readChess(bool is_read_chess, const vector<string> &file_paths)
    {
        char cwd[PATH_MAX];
        getcwd(cwd, sizeof(cwd));
        string current_dir(cwd);
        cv::FileStorage fs("./data/features.yml", cv::FileStorage::WRITE);
        vector<cv::Mat> descriptors;
        cv::Mat image = cv::imread(file_paths[0], cv::IMREAD_GRAYSCALE);
        int width = image.cols;
        int height = image.rows;
        int img_id = 0;
        cv::Size boardSize(boarder_size_, boarder_size_);
        for (const string &file_path : file_paths)
        {
            // 加载图像
            cv::Mat image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            if (image.empty())
            {
                cerr << "Failed to read image: " << file_path << endl;
                continue;
            }
            std::vector<cv::Point2f> corners;
            corners_.push_back(corners);
            bool found = cv::findChessboardCorners(image, boardSize, corners);
            if (found)
            {
                std::vector<cv::KeyPoint> kpts;
                for (const auto &corner : corners)
                    kpts.emplace_back(corner, 1.0);                              // 构造特征点，将角点坐标作为特征点位置
                                                                                 // 计算描述子
                cv::Ptr<cv::Feature2D> descriptorExtractor = cv::SIFT::create(); // 使用SIFT算法作为描述子提取器
                cv::Mat descriptor;
                descriptorExtractor->compute(image, kpts, descriptor);
                int lastSlash = file_path.find_last_of("/\\"); // 查找路径中最后一个斜杠或反斜杠的位置
                int lastDotPos = file_path.find_last_of('.');
                string filename = "img_" + file_path.substr(lastSlash + 1, lastDotPos - lastSlash - 1); // 提取最后一个斜杠之后的部分
                fs << filename << "{"
                   << "imd_id" << img_id
                   << "file_path" << file_path
                   << "kpts" << kpts
                   << "descriptor" << descriptor
                   << "}";
                descriptors_.push_back(descriptor);
                file_paths_[img_id] = file_path;
                img_id++;
            }
            else
            {
                std::cout << "Chessboard not found." << std::endl;
            }
        }
        fs.release();
        std::cout << "features detect done" << endl;
    };

    int Dataset::DetectFeatures(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
    {
        cv::Ptr<cv::SIFT> sifts = cv::SIFT::create(4000);
        // 构建图像金字塔
        // 构建图像金字塔
        std::vector<cv::Mat> pyramids;
        cv::buildPyramid(image, pyramids, 3);

        // 在每个金字塔层上检测特征点并计算描述符
        for (int level = 0; level < pyramids.size(); ++level)
        {
            cv::Mat &pyramid = pyramids[level];

            // 检测特征点
            std::vector<cv::KeyPoint> levelKeypoints;
            sifts->detect(pyramid, levelKeypoints);

            // 计算描述符
            cv::Mat levelDescriptors;
            sifts->compute(pyramid, levelKeypoints, levelDescriptors);

            // 转换特征点的坐标到原始图像尺度
            for (cv::KeyPoint &keypoint : levelKeypoints)
            {
                keypoint.pt *= pow(2.0, level);
                keypoints.push_back(keypoint);
            }

            // 将计算得到的描述符添加到总体描述符矩阵中
            descriptors.push_back(levelDescriptors);
        }

        float minDistancePixels = 1.5f;
        // 迭代遍历特征点,根据距离阈值筛选特征点
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

    /*
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
    */

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
    void Dataset::readImageSave(const string feature_path, const vector<string> &filenames)
    {
        if (filenames_.empty())
        {
            filenames_ = filenames;
            cout << "reading database" << endl;
            cv::FileStorage fs(feature_path, cv::FileStorage::READ);
            if (!fs.isOpened())
            {
                cout << "can't open file" << feature_path << endl;
            }
            for (auto &fmss : filenames_)
            {
                int lastSlash = fmss.find_last_of("/\\"); // 查找路径中最后一个斜杠或反斜杠的位置
                int lastDotPos = fmss.find_last_of('.');
                string fms = "img_" + fmss.substr(lastSlash + 1, lastDotPos - lastSlash - 1);
                cv::FileNode kptsNode = fs[fms]["kpts"];
                vector<cv::KeyPoint> kpts;
                for (const auto &kptNode : kptsNode)
                {
                    cv::KeyPoint kpt;
                    kptNode >> kpt;
                    kpts.push_back(kpt);
                }
                kpoints_.push_back(kpts);
                cv::FileNode descriptorNode = fs[fms]["descriptor"];
                cv::Mat descriptor;
                descriptorNode >> descriptor;
                descriptors_.push_back(descriptor);
            }
            fs.release();
        }
    }
    ///////////////////////////////建立好数据库后,再read//////////////////////////////////////////
    void Dataset::readDateSet(const string &matrixPath, const string &feature_path,
                              const vector<string> &filenames)
    {
        readImageSave(feature_path, filenames);
        for (int i = 0; i < filenames.size(); i++)
        {
            file_paths_.insert(std::make_pair(i, filenames[i]));
        }
    };
    ////////////////////////////计算chess matches///////////////////////////////////////
    void Dataset::computeChessMatches()
    {
        similarityMatrix_ = cv::Mat::zeros(filenames_.size(), filenames_.size(), CV_32SC1);
        for (int i = 0; i < file_paths_.size() - 1; ++i)
        {
            int j = i + 1;
            std::vector<cv::DMatch> matches;
            int numMatches = kpoints_[i].size();
            for (int k = 0; k < numMatches; k++)
            {
                cv::DMatch match(k, k, 0); // 创建DMatch对象，参数分别为queryIdx、trainIdx、distance
                matches.push_back(match);  // 将DMatch对象添加到matches向量中
            }

            std::pair<int, int> imagePair(i, j);
            if (!matchesMap_.count(imagePair))
            {
                matchesMap_[imagePair] = matches;
                similarityMatrix_.at<int>(i, j) = boarder_size_*boarder_size_;
            }
        }
        cv::FileStorage file("./data/similarityMatrix.yml", cv::FileStorage::WRITE);
        file << "matrix" << cv::Mat(similarityMatrix_);
        file.release();
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
    ////////////////////////////计算并储存matches,需要先readDateSet///////////////////////////////////////
    void Dataset::computeAndSaveMatches()
    {
        similarityMatrix_ = cv::Mat::zeros(filenames_.size(), filenames_.size(), CV_32SC1);
        // 对每对图像进行特征点匹配,并存储相似性矩阵
        for (int i = 0; i < file_paths_.size() - 1; ++i)
        {
            for (int j = i + 1; j < file_paths_.size(); ++j)
            {
                std::vector<cv::DMatch> matches;
                int match_size = ComputeMatches(kpoints_[i],
                                                kpoints_[j], descriptors_[i], descriptors_[j], matches, 0.9);
                // 构建图像对,当图像对大于30对匹配点才储存进来
                if (match_size > least_match_num_)
                {
                    similarityMatrix_.at<int>(i, j) = match_size;
                    std::pair<int, int> imagePair(i, j);
                    matchesMap_[imagePair] = matches;
                }
            }
        }
        for (int i = 0; i < file_paths_.size() - 1; ++i)
        {
            int j = i + 1;
            std::vector<cv::DMatch> matches;
            int match_size = ComputeMatches(kpoints_[i],
                                            kpoints_[j], descriptors_[i], descriptors_[j], matches, 0.9);
            std::pair<int, int> imagePair(i, j);
            if (!matchesMap_.count(imagePair))
            {
                similarityMatrix_.at<int>(i, j) = match_size;
                matchesMap_[imagePair] = matches;
            }
        }
        cv::FileStorage file("./data/similarityMatrix.yml", cv::FileStorage::WRITE);
        file << "matrix" << cv::Mat(similarityMatrix_);
        file.release();
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

    // 多尺度下计算Match
    int Dataset::ComputeMatches(vector<cv::KeyPoint> &kp01, vector<cv::KeyPoint> &kp02,
                                cv::Mat &desc1, cv::Mat &desc2,
                                std::vector<cv::DMatch> &RR_matches,
                                const float distance_ratio)
    {
        std::vector<cv::DMatch> matches;
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
        std::vector<cv::KeyPoint> R_keypoint01, R_keypoint02;
        std::vector<cv::Point2f> p01, p02;

        for (const auto &match : matches)
        {
            R_keypoint01.push_back(kp01[match.queryIdx]);
            R_keypoint02.push_back(kp02[match.trainIdx]);
            p01.push_back(kp01[match.queryIdx].pt);
            p02.push_back(kp02[match.trainIdx].pt);
        }

        cv::Mat Fundamental;
        std::vector<uchar> RansacStatus;
        Fundamental = cv::findFundamentalMat(p01, p02, cv::FM_RANSAC, 3.0, 0.99, RansacStatus);

        for (unsigned i = 0; i < matches.size(); i++)
        {
            if (RansacStatus[i] != 0)
            {
                RR_matches.push_back(matches[i]);
            }
        }
        return RR_matches.size();
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
    };

}