#include "inits.h"

using namespace std;
namespace ISfM
{
    Initializer::Initializer(const Parameters &params, const cv::Mat &K)
        : params_(params), K_(K)
    {
        assert(K_.type() == CV_64F);
    };
    Initializer::Initializer(const ImageLoader &image_loader, const Dataset &Cdate, const cv::Mat &sMatrix)
        : image_loader_(image_loader), Cdate_(Cdate), similar_matrix_(sMatrix)
    { // 2892.0,2883.0,823,605
        K_ = (cv::Mat_<double>(3, 3) << 1200.0, 0.0, 720.0,
              0.0, 1200.0, 540.0,
              0.0, 0.0, 1.0);
        // 建立所有的frame
        for (int i = 0; i < image_loader_.filenames_.size(); i++)
        {
            auto frame = Frame::CreateFrame();
            frame->img_name = image_loader_.filenames_[frame->id_];
            frames_.push_back(frame);
        }
        // 遍历 kpoints_, 将二维特征点转换为Feature对象,储存到frame中
        for (int i = 0; i < Cdate.kpoints_.size(); i++)
        {
            for (int j = 0; j < Cdate.kpoints_[i].size(); j++)
            {
                const cv::KeyPoint &kp = Cdate.kpoints_[i][j];
                Feature::Ptr feature = make_shared<Feature>(frames_[i], kp);
                feature->img_id_ = i;
                frames_[i]->features_img_.push_back(feature);
            }
        }
        
        matchesMap_ = Cdate.matchesMap_;
        map_ = Map::Ptr(new Map);
        Camera::Ptr camera_one = std::make_shared<Camera>(K_.at<double>(0, 0),
                                                      K_.at<double>(1, 1), K_.at<double>(0, 2),
                                                      K_.at<double>(1, 2));
        camera_one_ = camera_one;                                              
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
        vector<Feature::Ptr> pts1;
        vector<Feature::Ptr> pts2;
        vector<cv::Point2f> points1;
        vector<cv::Point2f> points2;
        // 通过迭代器的指针访问最多特征点vector<cv::DMatch>的值
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(similar_matrix_, &minVal, &maxVal, &minLoc, &maxLoc);
        auto it = matchesMap_.find(std::make_pair(maxLoc.y, maxLoc.x));
        if (it != matchesMap_.end())
        {
            vector<cv::DMatch> firstVector = it->second;
            for (const cv::DMatch &match : firstVector)
            {
                int trainIdx = match.trainIdx; // 获取trainIdx
                int queryIdx = match.queryIdx; // 获取queryIdx
                cv::Point2f point1(frames_[maxLoc.y]->features_img_[queryIdx]->position_.pt.x,
                                   frames_[maxLoc.y]->features_img_[queryIdx]->position_.pt.y);
                cv::Point2f point2(frames_[maxLoc.x]->features_img_[trainIdx]->position_.pt.x,
                                   frames_[maxLoc.x]->features_img_[trainIdx]->position_.pt.y);
                points1.push_back(point1);
                points2.push_back(point2);
                pts1.push_back(frames_[maxLoc.y]->features_img_[queryIdx]); // queryIdx是匹配时的第一张图片
                pts2.push_back(frames_[maxLoc.x]->features_img_[trainIdx]);  
            }
        }
        frames_[maxLoc.y]->is_registed = true;
        frames_[maxLoc.x]->is_registed = true;

        FindFundanmental(pts1, pts2, F, inlier_mask_F, num_inliers_F);

        assert(F.type() == CV_64F);

        statistics_.is_succeed = false; // 在这些地方进行了赋值和初始化?
        statistics_.num_inliers_F = num_inliers_F;

        RecoverPoseFromFundanmental(F, pts1, pts2, inlier_mask_F,maxLoc.y,maxLoc.x);

        PrintStatistics(statistics_);
        returns_.K_ = K_;
        returns_.map_ = map_;
        returns_.matchesMap_ = matchesMap_;
        returns_.frames_ = frames_;
        returns_.similar_matrix_ = similar_matrix_;
        returns_.id1 = maxLoc.y;
        returns_.id2 = maxLoc.x;
        return returns_;
    };

    /////////////////////////////要注意的是feature2D1和feature2D2得是对应点////////////////////////////////
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
    bool Initializer::RecoverPoseFromFundanmental(const cv::Mat &F,
                                                  const vector<Feature::Ptr> &feature2D1,
                                                  const vector<Feature::Ptr> &feature2D2,
                                                  const vector<bool> &inlier_mask_F,
                                                  const int id1, const int id2)
    {
        cv::Mat E, Rwto1, t1, Rwto2, t2;
        cv::Mat inlier;
        vector<cv::Point2f> points2D1 = Feature::convertFeaturesToPoints(feature2D1);
        vector<cv::Point2f> points2D2 = Feature::convertFeaturesToPoints(feature2D2);
        // 所以直接使用opencv的findEssentialMat
        // 然后再recoverPose
        E = cv::findEssentialMat(points2D1, points2D2, K_, cv::RANSAC,
                                 params_.rel_pose_ransac_confidence,
                                 params_.rel_pose_essential_error, inlier);

        cv::recoverPose(E, points2D1, points2D2, K_, Rwto2, t2); // Rwto2也是从points2D1到points2D2的转换
        t2 = t2 * 2.0; // 没有尺度信息，这里先给尺度大一些

        Rwto1 = cv::Mat::eye(3, 3, CV_64F);
        t1 = cv::Mat::zeros(3, 1, CV_64F);
        Eigen::Matrix3d R1;
        Eigen::Vector3d t11;
        cv::cv2eigen(Rwto1, R1);
        cv::cv2eigen(t1, t11);
        Sophus::SE3d pose1(R1, t11);
        frames_[id1]->SetPose(pose1);

        Eigen::Matrix3d R2;
        Eigen::Vector3d t22;
        cv::cv2eigen(Rwto2, R2);
        cv::cv2eigen(t2, t22);
        Sophus::SE3d pose2(R2, t22);
        frames_[id2]->SetPose(pose2);
        std::vector<SE3> poses{frames_[id1]->pose_, frames_[id2]->pose_};

        cv::Mat P1, P2;
        cv::hconcat(K_ * Rwto1, K_ * t1, P1);
        cv::hconcat(K_ * Rwto2, K_ * t2, P2);
        vector<double> tri_angles(points2D1.size(), 0);
        size_t num_inliers = 0;
        double sum_residual = 0.0;
        double sum_tri_angle = 0.0;
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
                num_inliers += 1;
                sum_residual += error;
                sum_tri_angle += tri_angle;
            }
        }

        TriangulateInitPoints(frames_[id1], frames_[id2]);
        
        cout << "cnt_init_landmarks:" << num_inliers << endl;
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
        cout << setw(kWidth) << "Num inliers F"
             << " : " << statistics.num_inliers_F << endl;
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

    void Initializer::coutFeaturePoint(const vector<Feature::Ptr> &feature2D1,
                                         const vector<Feature::Ptr> &feature2D2)
    {
        int f0n = 0;
        int f1n = 0;
        for (int i = 0; i < feature2D1.size(); ++i)
        {
            if (!feature2D1[i]->map_point_.expired())
                f0n++;
            if (!feature2D2[i]->map_point_.expired())
                f1n++;
        }
        cout << "frame0:" << f0n << endl;
        cout << "frame1:" << f1n << endl;
    }
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

    void Initializer::TriangulateInitPoints(Frame::Ptr &frame_one, Frame::Ptr &frame_two)
    {
        cv::Mat image = cv::imread(frame_one->img_name, cv::IMREAD_COLOR);
        std::vector<SE3> poses{frame_one->pose_, frame_two->pose_};
        SE3 current_pose_Twc_one = frame_one->Pose().inverse(); // camera to world,frame里的是world to camera
        SE3 current_pose_Twc_two = frame_two->Pose().inverse();
        int cnt_triangulated_pts = 0; // 三角化成功的点的数目
        vector<cv::DMatch> &this_match = matchesMap_[make_pair(frame_one->id_, frame_two->id_)];
        for (size_t i = 0; i < this_match.size(); ++i)
        {
            // 取出对应的特征点索引
            int queryIdx = this_match[i].queryIdx;
            int trainIdx = this_match[i].trainIdx;
            // 根据索引获取特征点
            std::shared_ptr<Feature> queryFeature = frame_one->features_img_[queryIdx];
            std::shared_ptr<Feature> trainFeature = frame_two->features_img_[trainIdx];

            // 遍历两个frame match的所有点
            if (queryFeature->map_point_.expired() &&
                trainFeature->map_point_.expired()) // expired()为true的话表示关联的std::shared_ptr已经被销毁
            {
                // 左图的特征点未关联地图点且存在右图匹配点,尝试三角化
                std::vector<Vec3> points{
                    camera_one_->pixel2camera(
                        Vec2(queryFeature->position_.pt.x,
                             queryFeature->position_.pt.y)),
                    camera_one_->pixel2camera(
                        Vec2(trainFeature->position_.pt.x,
                             trainFeature->position_.pt.y))};
                Vec3 pworld = Vec3::Zero();

                if (triangulation(poses, points, pworld) && pworld[2] > 0)
                {
                    auto new_map_point = MapPoint::CreateNewMappoint();
                    uchar* pixel = image.ptr<uchar>(queryFeature->position_.pt.y, queryFeature->position_.pt.x);
                    Eigen::Matrix<uchar, 3, 1> color;
                    color << pixel[2], pixel[1], pixel[0];
                    new_map_point->SetColor(color);
                    cv::Vec3d cvVector;
                    cvVector[0] = pworld(0);
                    cvVector[1] = pworld(1);
                    cvVector[2] = pworld(2);
                    new_map_point->SetPos(cvVector); // 设置mapoint类中的坐标
                    new_map_point->AddObservation(
                        queryFeature); // 增加mappoint类中的对应的那个feature（左右目）
                    new_map_point->AddObservation(
                        trainFeature);

                    queryFeature->map_point_ = new_map_point;
                    trainFeature->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        std::cout << "new landmarks: " << cnt_triangulated_pts << std::endl; // 这一步成功率太低
    };
}