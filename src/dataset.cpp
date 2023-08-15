#include "dataset.hpp"

namespace ISfM
{
    // extract all orbsift in all images, then save it
    void Dataset::establishDbo(const std::vector<std::string> &filenames)
    {
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
        for (const std::string &filename : filenames)
        {
            cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            int width = image.cols;
            int height = image.rows;
            Mat33 K;
            K << width / 2., 0., width / 2.,
                0., width / 2., height / 2.,
                0., 0., 1.; // init K matrix
            Vec3 t;
            t << 0., 0., 0.;
            Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                              SE3(SO3(), t))); // SO3()初始化为0
            cameras_.push_back(new_camera);

            if (image.empty())
            {
                std::cerr << "Failed to read image: " << filename << std::endl;
                continue;
            }
            std::vector<cv::KeyPoint> kpts;
            cv::Mat descriptor;
            orb->detectAndCompute(image, cv::noArray(), kpts, descriptor);
            descriptors_.push_back(descriptor);
            features_[filename] = std::make_pair(kpts, descriptor);
        }
        vocab.create(descriptors); // save vocab data base
        std::cout << "vocabulary info: " << vocab << std::endl;
        char cwd[PATH_MAX];
        getcwd(cwd, sizeof(cwd));
        std::string current_dir(cwd);
        std::string file_path = current_dir + "/vocab_larger.yml.gz";
        vocab.save(file_path);
        vocab_file_path = file_path;
        std::cout << "done" << std::endl;
    }

    // using DBoW3 to find similar image, for initialization
    void Dataset::findImageSimilar(const string vocab_file_path, int num_img)
    {
        std::cout << "reading database" << std::endl;
        DBoW3::Vocabulary vocab(vocab_file_path);
        // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");
        if (vocab.empty())
        {
            std::cerr << "Vocabulary does not exist." << std::endl;
            return 1;
        }
        std::cout << "reading images... " << std::endl;
        std::vector<cv::Mat> images;
        for (const auto &entry : feature.getFeatures())
        {
            const std::string &filename = entry.first;
            images.push_back(imread(filename));
        }

        std::cout << "comparing images with database " << std::endl;
        DBoW3::Database db(vocab, false, 0);
        for (int i = 0; i < descriptors_.size(); i++)
        {
            db.add(descriptors_[i]);
        }
        // 计算num_img张图片之间的相似度得分
        cv::Mat db_scores(num_img, num_img, CV_64F);
        for (int i = 0; i < num_img; i++)
        {
            for (int j = i + 1; j < num_img; j++)
            {
                // 计算第i张图片和第j张图片之间的相似度得分
                double score = vocab.score(bowVecs[i], bowVecs[j]);
                db_scores.at<double>(i, j) = score;
            }
        }
        // 设置相似度得分矩阵
        setDboScore(db_scores);
    }
}

int main()
{
    ImageLoader image_loader("./images");
    Dataset feature;
    feature.establishDbo(image_loader.filenames);
    feature.findImageSimilar(feature.vocab_file_path);

    return 0;
}