#include "dataset.h"
#include "frame.h"

using namespace std;
namespace ISfM
{
    // extract all orbsift in all images, then save it
    void Dataset::establishDbo(const vector<string> &file_paths)
    {
        // 找到当前路径
        char cwd[PATH_MAX];
        getcwd(cwd, sizeof(cwd));
        std::string current_dir(cwd);
        // 创建ORB特征提取器对象
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
        // 创建文件存储对象
        cv::FileStorage fs(current_dir + "/features.yml", cv::FileStorage::WRITE);
        // 创建yml文件名存储对象
        std::ofstream file("./feature_name.txt"); // 打开一个输出文件流
        // 创建词袋模型存储mat
        vector<cv::Mat> descriptors;
        // 加载第一张图像
        cv::Mat image = cv::imread(file_paths[0], cv::IMREAD_GRAYSCALE);
        int width = image.cols;
        int height = image.rows;
        Mat33 K;
        K << width / 2., 0., width / 2.,
            0., width / 2., height / 2.,
            0., 0., 1.; // init K matrix
        Vec3 t;
        t << 0., 0., 0.;
        // Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
        //                                       SE3(SO3(), t))); // SO3()初始化为0
        // cameras_.push_back(new_camera);

        // 遍历图片文件
        for (const string &file_path : file_paths)
        {
            // 加载图像
            cv::Mat image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            if (image.empty())
            {
                std::cerr << "Failed to read image: " << file_path << std::endl;
                continue;
            }
            // 检测图像中的关键点
            std::vector<cv::KeyPoint> kpts;
            // 计算关键点的描述符
            cv::Mat descriptor;
            orb->detectAndCompute(image, cv::noArray(), kpts, descriptor);
            // 将关键点和描述符保存到文件
            // 提取文件名
            size_t lastSlash = file_path.find_last_of("/\\");                // 查找路径中最后一个斜杠或反斜杠的位置
            std::string filename = "img_" + file_path.substr(lastSlash + 1); // 提取最后一个斜杠之后的部分
            std::replace(filename.begin(), filename.end(), '.', 'd');
            fs << filename << "{"
               << "kpts" << kpts
               << "descriptor" << descriptor
               << "}";
            if (file.is_open())
            {
                file << filename << '\n';
                // 写入内外参数据
                file << K << '\n';
                file << t << '\n';
            }
            descriptors.push_back(descriptor);
            features_[filename] = std::make_pair(kpts, descriptor);
        }
        // 关闭文件存储
        fs.release();
        file.close();
        // save vocab data base
        DBoW3::Vocabulary vocab;
        vocab.create(descriptors);
        // std::cout << "vocabulary info: " << vocab << std::endl;
        std::string file_path = current_dir + "/vocab_larger.yml.gz";
        vocab.save(file_path);
        vocab_file_path = file_path;
        std::cout << "done" << std::endl;
    }

    // using DBoW3 to find similar image, for initialization
    void Dataset::saveSimilarMatrix(const string vocab_file_path,
                                    const string feature_path, const string filename_path)
    {
        readImageSave(feature_path, filename_path);
        DBoW3::Vocabulary vocab(vocab_file_path);
        if (vocab.empty())
        {
            std::cerr << "Vocabulary does not exist." << std::endl;
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
        cv::FileStorage file("./similarityMatrix.yml", cv::FileStorage::WRITE);
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
                std::vector<cv::DMatch> matches;
                // 特征匹配
                matcher.match(descriptors_[i], descriptors_[j], matches);
                // 统计匹配点数量
                int numMatches = matches.size();
                ORBSimilar(i, j) = numMatches;
            }
        }
        // cout << similarityMatrix.matrix();
        std::ofstream file("ORBSimilar.txt");
        if (file.is_open())
        {
            // 将矩阵的行和列写入文件
            file << ORBSimilar.rows() << " " << ORBSimilar.cols() << std::endl;
            // 逐行写入矩阵元素
            for (int i = 0; i < ORBSimilar.rows(); ++i)
            {
                for (int j = 0; j < ORBSimilar.cols(); ++j)
                {
                    file << ORBSimilar(i, j) << " ";
                }
                file << std::endl;
            }
            file.close();
            std::cout << "Matrix saved to file." << std::endl;
        }
        else
        {
            std::cerr << "Failed to open file." << std::endl;
        }
    };
    ////////////////////////////////////////////////////////////////////////////////////
    void Dataset::readImageSave(const string feature_path, const string filename_path)
    {
        std::ifstream file(filename_path); // 打开一个输入文件流
        if (file.is_open())
        {
            std::string line;
            while (std::getline(file, line))
            {
                if (line.substr(0, 3) == "img")
                {
                    filenames_.push_back(line); // 存储以"img"开头的字段
                }
            }
            file.close(); // 关闭文件流
        }
        else
        {
            std::cout << "无法打开文件。" << std::endl;
        }

        std::cout << "reading database" << std::endl;
        cv::FileStorage fs(feature_path, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cout << "无法打开文件：" << feature_path << std::endl;
        }
        for (auto &fms : filenames_)
        {
            cv::FileNode kptsNode = fs[fms]["kpts"];
            std::vector<cv::KeyPoint> kpts;
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
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat Dataset::readSimilarMatrix(const string matrixPath,const string feature_path, const string filename_path)
    {
        cv::FileStorage fs(matrixPath, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cout << "无法打开文件：" << matrixPath << std::endl;
        }
        cv::Mat sMatrix;
        fs["matrix"] >> sMatrix;
        fs.release();
        return sMatrix;
    }
}