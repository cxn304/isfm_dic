#include "inits.h"

using namespace std;
namespace ISfM
{
    Initializer::Initializer(const Parameters &params, const cv::Mat &K)
        : params_(params), K_(K)
    {
        assert(K_.type() == CV_64F);
    };
    Initializer::Initializer(const ImageLoader &image_loader, const Dataset &Cdate)
        : image_loader_(image_loader), Cdate_(Cdate)
    {
        K_ = (cv::Mat_<double>(3, 3) << image_loader.width_, 0, image_loader.width_,
              0, image_loader.width_, image_loader.height_,
              0, 0, 1);
        // 遍历 kpoints_, 将二维特征点转换为Feature对象
        for (int i = 0; i < Cdate.kpoints_.size(); i++)
        {
            vector<Feature::Ptr> featureRow;

            for (int j = 0; j < Cdate.kpoints_[i].size(); j++)
            {
                const cv::KeyPoint &kp = Cdate.kpoints_[i][j];
                Feature::Ptr feature = make_shared<Feature>(i, kp);

                featureRow.push_back(feature);
            }
            features_.push_back(featureRow);
        }
        // 建立所有的frame
        for (int i = 0; i < image_loader_.filenames_.size(); i++) 
        {
            auto frame = Frame::CreateFrame();
            for (auto &feature_tmp : features_[i])
            {
                frame->features_img_.push_back(feature_tmp); // 将当前图片里的所有特征点都加入frame里面
                feature_tmp->frame_ = frame;    // 这些特征点都和当前frame关联起来
            }
            frame->img_name = image_loader_.filenames_[frame->id_];
            frames_.push_back(frame);
        }

        matchesMap_ = Cdate.matchesMap_;
        map_ = Map::Ptr(new Map);
    };
    //////////////////////////////////////////////////////////////////////////////////////
    Initializer::Returns Initializer::Initialize()
    {
        cv::Mat H;
        cv::Mat F;
        vector<bool> inlier_mask_H;
        vector<bool> inlier_mask_F;
        size_t num_inliers_H;
        size_t num_inliers_F;
        auto it = matchesMap_.begin();
        vector<Feature::Ptr> pts1;
        vector<Feature::Ptr> pts2;
        // 检查迭代器是否指向有效元素
        if (it != matchesMap_.end())
        {
            // 通过迭代器的指针访问第一个vector<cv::DMatch>的值
            vector<cv::DMatch> firstVector = it->second;

            for (const cv::DMatch &match : firstVector)
            {
                int trainIdx = match.trainIdx;          // 获取trainIdx
                int queryIdx = match.queryIdx;          // 获取queryIdx
                pts1.push_back(features_[0][queryIdx]); // queryIdx是匹配时的第一张图片
                pts2.push_back(features_[1][trainIdx]);
            }
        }

        FindHomography(pts1, pts2, H, inlier_mask_H, num_inliers_H);
        FindFundanmental(pts1, pts2, F, inlier_mask_F, num_inliers_F);

        assert(H.type() == CV_64F);
        assert(F.type() == CV_64F);

        double H_F_ratio = static_cast<double>(num_inliers_H) / static_cast<double>(num_inliers_F);

        statistics_.is_succeed = false; // 在这些地方进行了赋值和初始化?
        statistics_.num_inliers_F = num_inliers_F;
        statistics_.num_inliers_H = num_inliers_H;
        statistics_.H_F_ratio = H_F_ratio;

        // 不管使用何种方法进行初始化, 内点数都要满足要求
        if (H_F_ratio < 0.7 && num_inliers_F >= params_.rel_pose_min_num_inlier)
        {
            // 如果H矩阵的内点数少,使用基础矩阵进行初始化
            RecoverPoseFromFundanmental(F, pts1, pts2, inlier_mask_F);
        }
        else if (num_inliers_H >= params_.rel_pose_min_num_inlier)
        {
            // 使用单应矩阵进行初始化
            RecoverPoseFromHomography(H, pts1, pts2, inlier_mask_H);
        }
        else
        {
            statistics_.fail_reason = "Not sufficient inliers";
        }

        PrintStatistics(statistics_);
        returns_.K_ = K_;
        returns_.features_ = features_;
        returns_.map_ = map_;
        returns_.matchesMap_ = matchesMap_;
        returns_.frames_ = frames_;

        return returns_;
    };

    /////////////////////////////要注意的是feature2D1和feature2D2得是对应点////////////////////////////////
    void Initializer::FindHomography(const vector<Feature::Ptr> &feature2D1,
                                     const vector<Feature::Ptr> &feature2D2,
                                     cv::Mat &H,
                                     vector<bool> &inlier_mask,
                                     size_t &num_inliers)
    {
        // 使用引用作为函数参数的好处是可以避免在函数调用时进行对象的复制,从而提高性能
        cv::Mat cv_inlier_mask; // 输出的内点掩码,用于标记哪些点对被认为是内点
        vector<cv::Point2f> points2D1 = Feature::convertFeaturesToPoints(feature2D1);
        vector<cv::Point2f> points2D2 = Feature::convertFeaturesToPoints(feature2D2);
        H = cv::findHomography(points2D1,
                               points2D2,
                               cv::RANSAC, params_.rel_pose_homography_error,
                               cv_inlier_mask, 10000,
                               params_.rel_pose_ransac_confidence);
        // cv_inlier_mask是一个单通道,以8位无符号整数类型存储的灰度图像矩阵,每个像素值为0或255,用于表示点对的内点掩码
        assert(cv_inlier_mask.type() == CV_8U);
        inlier_mask.resize(cv_inlier_mask.rows, false);
        num_inliers = 0;

        for (int i = 0; i < cv_inlier_mask.rows; ++i)
        {
            if (cv_inlier_mask.at<uchar>(i, 0) == 0)
                continue;
            // cv_inlier_mask为255表示是内点
            inlier_mask[i] = true;
            num_inliers += 1;
        }
    };
    //////////////////////////////////////////////////////////////////////////////////////
    void Initializer::FindFundanmental(const vector<Feature::Ptr> &feature2D1,
                                       const vector<Feature::Ptr> &feature2D2,
                                       cv::Mat &F,
                                       vector<bool> &inlier_mask,
                                       size_t &num_inliers)
    {
        cv::Mat cv_inlier_mask;
        vector<cv::Point2f> points2D1 = Feature::convertFeaturesToPoints(feature2D1);
        vector<cv::Point2f> points2D2 = Feature::convertFeaturesToPoints(feature2D2);
        F = cv::findFundamentalMat(points2D1,
                                   points2D2,
                                   cv::FM_RANSAC, params_.rel_pose_essential_error,
                                   params_.rel_pose_ransac_confidence, cv_inlier_mask);
        // opencv自带RANSAC算法,我们只需要自己设定误差阈值和置信度
        assert(cv_inlier_mask.type() == CV_8U);

        inlier_mask.resize(cv_inlier_mask.rows, false);
        num_inliers = 0;

        for (int i = 0; i < cv_inlier_mask.rows; ++i)
        {
            if (cv_inlier_mask.at<uchar>(i, 0) == 0)
                continue;

            inlier_mask[i] = true;
            num_inliers += 1;
        }
    };
    //////////////////////////////////////////////////////////////////////////////////////
    bool Initializer::RecoverPoseFromHomography(const cv::Mat &H,
                                                const vector<Feature::Ptr> &feature2D1,
                                                const vector<Feature::Ptr> &feature2D2,
                                                const vector<bool> &inlier_mask_H)
    {
        vector<cv::Mat> Rs; // 用于输出
        vector<cv::Mat> ts;
        vector<cv::Point2f> points2D1 = Feature::convertFeaturesToPoints(feature2D1);
        vector<cv::Point2f> points2D2 = Feature::convertFeaturesToPoints(feature2D2);
        // 将单应性矩阵分解为旋转矩阵和平移向量
        cv::decomposeHomographyMat(H, K_, Rs, ts, cv::noArray()); // Rs和ts有多个解
        size_t best_num_inlier = 0;
        // 在这些解中找到最合适的解
        for (size_t k = 0; k < Rs.size(); ++k)
        {
            cv::Mat Rwto1 = cv::Mat::eye(3, 3, CV_64F); // 表示参考图像本身的旋转
            cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat Rwto2 = Rs[k]; // Rs表示从世界坐标系到相机2坐标系的旋转变换
            cv::Mat t2 = ts[k];

            cv::Mat P1, P2; // for output,投影矩阵
            vector<double> tri_angles(points2D1.size(), 0);
            cv::hconcat(K_ * Rwto1, K_ * t1, P1); // 投影矩阵: 内参和外参的乘积
            cv::hconcat(K_ * Rwto2, K_ * t2, P2);

            size_t num_inliers = 0;
            double sum_residual = 0;
            double sum_tri_angle = 0;
            size_t cnt_init_landmarks = 0; // 初始化的路标数目
            for (size_t i = 0; i < points2D1.size(); ++i)
            {
                cv::Vec3d p3d = Triangulate(P1, P2, points2D1[i], points2D2[i]);

                bool has_positive_depth = Projection::HasPositiveDepth(p3d, Rwto1, t1, Rwto2, t2);
                // 计算重投影误差,points2D1是观测到的坐标
                double error = Projection::CalculateReprojectionError(p3d,
                                                                      points2D1[i], points2D2[i],
                                                                      Rwto1, t1, Rwto2, t2, K_);
                // 计算视差角(parallax angle)
                double tri_angle = Projection::CalculateParallaxAngle(p3d, Rwto1, t1, Rwto2, t2);

                tri_angles[i] = tri_angle;

                if (has_positive_depth && error < params_.init_tri_max_error)
                {
                    auto new_map_point = MapPoint::CreateNewMappoint(); // 工厂模式创建一个新的地图点
                    new_map_point->SetPos(p3d);
                    new_map_point->AddObservation(feature2D1[i]);
                    new_map_point->AddObservation(feature2D2[i]);
                    feature2D1[i]->map_point_ = new_map_point;
                    feature2D2[i]->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point); // 地图插入新的地图点
                    cnt_init_landmarks++;

                    num_inliers += 1;
                    sum_residual += error;
                    sum_tri_angle += tri_angle;
                }
            }
            if (num_inliers > best_num_inlier)
            {

                sort(tri_angles.begin(), tri_angles.end());

                double ave_tri_angle = sum_tri_angle / num_inliers;
                double ave_residual = sum_residual / num_inliers;
                double median_tri_angle = 0.0f;

                if (tri_angles.size() % 2 == 1)
                {
                    median_tri_angle = tri_angles[tri_angles.size() / 2];
                }
                else
                {
                    median_tri_angle = tri_angles[(tri_angles.size() - 1) / 2] + tri_angles[tri_angles.size() / 2];
                    median_tri_angle /= 2;
                }

                best_num_inlier = num_inliers;
                statistics_.method = "Homography";
                statistics_.num_inliers = num_inliers;
                statistics_.median_tri_angle = median_tri_angle;
                statistics_.ave_tri_angle = ave_tri_angle;
                statistics_.ave_residual = ave_residual;

                Eigen::Matrix3d R1;
                Eigen::Vector3d t11;
                cv::cv2eigen(Rwto1, R1);
                cv::cv2eigen(t1, t11);
                Sophus::SE3d pose1(R1, t11);
                frames_[0]->SetPose(pose1);

                Eigen::Matrix3d R2;
                Eigen::Vector3d t22;
                cv::cv2eigen(Rwto2, R2);
                cv::cv2eigen(t2, t22);
                Sophus::SE3d pose2(R2, t22);
                frames_[1]->SetPose(pose2);

            }
        }
        // 判断是否初始化成功
        if (statistics_.num_inliers < params_.rel_pose_min_num_inlier ||
            statistics_.median_tri_angle < params_.init_tri_min_angle ||
            statistics_.ave_tri_angle < params_.init_tri_min_angle ||
            statistics_.ave_residual > params_.init_tri_max_error)
        {
            statistics_.is_succeed = false;
            statistics_.fail_reason = GetFailReason();
        }
        else
        {
            statistics_.is_succeed = true;
        }

        return statistics_.is_succeed;
    };
    //////////////////////////////////////////////////////////////////////////////////////
    bool Initializer::RecoverPoseFromFundanmental(const cv::Mat &F,
                                                  const vector<Feature::Ptr> &feature2D1,
                                                  const vector<Feature::Ptr> &feature2D2,
                                                  const vector<bool> &inlier_mask_F)
    {
        cv::Mat E, Rwto1, t1, Rwto2, t2;
        cv::Mat inlier;
        vector<cv::Point2f> points2D1 = Feature::convertFeaturesToPoints(feature2D1);
        vector<cv::Point2f> points2D2 = Feature::convertFeaturesToPoints(feature2D2);
        // 由于使用 E = K_.t() * F * K_
        // 然后recoverPose会出错
        // 所以直接使用opencv的findEssentialMat
        // 然后再recoverPose
        E = cv::findEssentialMat(points2D1, points2D2, K_, cv::RANSAC,
                                 params_.rel_pose_ransac_confidence,
                                 params_.rel_pose_essential_error, inlier);

        cv::recoverPose(E, points2D1, points2D2, K_, Rwto2, t2);

        Rwto1 = cv::Mat::eye(3, 3, CV_64F);
        t1 = cv::Mat::zeros(3, 1, CV_64F);

        cv::Mat P1, P2;
        cv::hconcat(K_ * Rwto1, K_ * t1, P1);
        cv::hconcat(K_ * Rwto2, K_ * t2, P2);
        vector<double> tri_angles(points2D1.size(), 0);
        size_t num_inliers = 0;
        double sum_residual = 0.0;
        double sum_tri_angle = 0.0;
        size_t cnt_init_landmarks = 0; // 初始化的路标数目
        for (size_t i = 0; i < points2D1.size(); ++i)
        {
            if (inlier.at<uchar>(i, 0) == 0)
                continue;
            if (!inlier_mask_F[i])
                continue;
            cv::Vec3d p3d = Triangulate(P1, P2, points2D1[i], points2D2[i]);
            // 三角测量出来的点,要满足 正深度 重投影误差小于阈值 三角测量的角度大于阈值
            bool has_positive_depth = Projection::HasPositiveDepth(p3d, Rwto1, t1, Rwto2, t2);
            double error = Projection::CalculateReprojectionError(p3d, points2D1[i], points2D2[i],
                                                                  Rwto1, t1, Rwto2, t2, K_);
            double tri_angle = Projection::CalculateParallaxAngle(p3d, Rwto1, t1, Rwto2, t2);

            tri_angles[i] = tri_angle;

            if (has_positive_depth && error < params_.init_tri_max_error)
            {
                auto new_map_point = MapPoint::CreateNewMappoint(); // 工厂模式创建一个新的地图点
                new_map_point->SetPos(p3d);
                new_map_point->AddObservation(feature2D1[i]);
                new_map_point->AddObservation(feature2D2[i]);
                feature2D1[i]->map_point_ = new_map_point;
                feature2D2[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point); // 地图插入新的地图点
                cnt_init_landmarks++;

                num_inliers += 1;
                sum_residual += error;
                sum_tri_angle += tri_angle;
            }
        }

        sort(tri_angles.begin(), tri_angles.end());

        double ave_tri_angle = sum_tri_angle / num_inliers;
        double ave_residual = sum_residual / num_inliers;
        double median_tri_angle = 0.0;

        // 获取已经成功三角测量（即重投影误差小于一定的阈值）的3D点的角度的中位数
        if (tri_angles.size() % 2 == 1)
        {
            median_tri_angle = tri_angles[tri_angles.size() / 2];
        }
        else
        {
            median_tri_angle = tri_angles[(tri_angles.size() - 1) / 2] + tri_angles[tri_angles.size() / 2];
            median_tri_angle /= 2;
        }

        // 判断是否初始化成功
        if (num_inliers < params_.rel_pose_min_num_inlier ||
            median_tri_angle < params_.init_tri_min_angle ||
            ave_tri_angle < params_.init_tri_min_angle ||
            ave_residual > params_.init_tri_max_error)
        {
            statistics_.is_succeed = false;
            statistics_.fail_reason = GetFailReason();
            statistics_.num_inliers = num_inliers;
            statistics_.median_tri_angle = median_tri_angle;
            statistics_.ave_tri_angle = ave_tri_angle;
            statistics_.ave_residual = ave_residual;
            return false;
        }

        Eigen::Matrix3d R1;
        Eigen::Vector3d t11;
        cv::cv2eigen(Rwto1, R1);
        cv::cv2eigen(t1, t11);
        Sophus::SE3d pose1(R1, t11);
        frames_[0]->SetPose(pose1);

        Eigen::Matrix3d R2;
        Eigen::Vector3d t22;
        cv::cv2eigen(Rwto2, R2);
        cv::cv2eigen(t2, t22);
        Sophus::SE3d pose2(R2, t22);
        frames_[1]->SetPose(pose2);

        statistics_.is_succeed = true;
        statistics_.method = "Essential";
        statistics_.num_inliers = num_inliers;
        statistics_.median_tri_angle = median_tri_angle;
        statistics_.ave_tri_angle = ave_tri_angle;
        statistics_.ave_residual = ave_residual;

        return true;
    }
    //////////////////////////////////////////////////////////////////////////////////////
    cv::Vec3d Initializer::Triangulate(const cv::Mat &P1,
                                       const cv::Mat &P2,
                                       const cv::Point2f &point2D1,
                                       const cv::Point2f &point2D2)
    {
        // P1,P2是投影矩阵,内参和外参的乘积
        cv::Mat A(4, 4, CV_64F);
        // DLT
        A.row(0) = point2D1.x * P1.row(2) - P1.row(0); // 计算第一个二维点在相机1投影矩阵上的投影坐标与齐次坐标之间的关系的表达式
        A.row(1) = point2D1.y * P1.row(2) - P1.row(1);
        A.row(2) = point2D2.x * P2.row(2) - P2.row(0); // 计算第二个二维点在相机2投影矩阵上的投影坐标与齐次坐标之间的关系的表达式
        A.row(3) = point2D2.y * P2.row(2) - P2.row(1);
        // s * [u, v, 1]^T = P * [X, Y, Z, 1]^T
        // [u, v]是二维图像上的点坐标,[X, Y, Z]是三维空间中的点坐标,P是相机投影矩阵,s是一个尺度因子
        cv::Mat u, w, vt; // w存储奇异值,u和vt存储相应的左和右奇异向量
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        // 最小特征值所对应的特征向量, 就是该三维点的齐次坐标形式(x,y,z,w),非齐次(x,y,z)
        cv::Mat p3d = vt.row(3).t();
        assert(p3d.type() == CV_64F);
        // 从齐次坐标 -> 非齐次坐标
        p3d = p3d.rowRange(0, 3) / p3d.at<double>(3, 0);

        double x = p3d.at<double>(0);
        double y = p3d.at<double>(1);
        double z = p3d.at<double>(2);
        return cv::Vec3d(x, y, z);
    };
    //////////////////////////////////////////////////////////////////////////////////////
    void Initializer::PrintStatistics(const Statistics &statistics)
    {
        const size_t kWidth = 20;
        cout.flags(ios::left); // 左对齐
        cout << endl;
        cout << "--------------- Initialize Summary Start ---------------" << endl;
        cout << setw(kWidth) << "Initialize status"
                  << " : " << (statistics.is_succeed ? "true" : "false") << endl;
        cout << setw(kWidth) << "Initialize method"
                  << " : " << statistics.method << endl;
        if (!statistics.is_succeed)
        {
            cout << setw(kWidth) << "Fail reason"
                      << " : " << statistics.fail_reason << endl;
        }
        cout << setw(kWidth) << "Num inliers H"
                  << " : " << statistics.num_inliers_H << endl;
        cout << setw(kWidth) << "Num inliers F"
                  << " : " << statistics.num_inliers_F << endl;
        cout << setw(kWidth) << "H F ratio"
                  << " : " << statistics.H_F_ratio << endl;
        cout << setw(kWidth) << "Num inliers"
                  << " : " << statistics.num_inliers << endl;
        cout << setw(kWidth) << "Median tri angle"
                  << " : " << statistics.median_tri_angle << endl;
        cout << setw(kWidth) << "Ave tri angle"
                  << " : " << statistics.ave_tri_angle << endl;
        cout << setw(kWidth) << "Ave residual"
                  << " : " << statistics.ave_residual << endl;
        cout << "--------------- Initialize Summary End ---------------" << endl;
        cout << endl;
    };
    //////////////////////////////////////////////////////////////////////////////////////
    string Initializer::GetFailReason()
    {

        assert(!statistics_.is_succeed);

        string fail_reason = "";

        if (statistics_.num_inliers < params_.rel_pose_min_num_inlier)
        {
            fail_reason = "Not sufficient inliers";
        }
        if (statistics_.median_tri_angle < params_.init_tri_min_angle ||
            statistics_.ave_residual < params_.init_tri_min_angle)
        {
            fail_reason = (fail_reason.size() == 0) ? ("Not sufficient angle") : (fail_reason + " & Not sufficient angle");
        }
        if (statistics_.ave_residual > params_.init_tri_max_error)
        {
            fail_reason = (fail_reason.size() == 0) ? ("Too large triangulation error") : (fail_reason + " & Too large triangulation error");
        }

        return fail_reason;
    };
}
