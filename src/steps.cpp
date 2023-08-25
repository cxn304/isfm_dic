#include "steps.h"

namespace ISfM
{
    Steps::Steps()
    {
        /*
        最大特征点数量 num_features，
        角点可以接受的最小特征值 检测到的角点的质量等级，角点特征值小于qualityLevel*最大特征值的点将被舍弃 0.01
        角点最小距离 20
        */
        gftt_ =
            cv::GFTTDetector::create(150, 0.01, 20);
        num_features_init_ = 50;
        num_features_ = 50;
    }
    // 在增加某一帧时，根据目前的状况选择不同的处理函数
    bool Steps::AddFrame(Frame::Ptr frame)
    {
        current_frame_ = frame;
        // Track()是Frontend的成员函数,status_是Frontend的数据,可以直接使用
        switch (status_)
        {
        case ConstructionStatus::INITING:
            Init(initialize_);
            break; // 双目初始化函数 StereoInit() 跑完了以后，不管状态是TRACKING_GOOD还是，TRACKING_BAD，都会运行 Track() 函数了
        case ConstructionStatus::TRACKING_GOOD:
        case ConstructionStatus::TRACKING_BAD:
            Track();
            break;
        case ConstructionStatus::LOST:
            Reset(); // 如果前端跟丢了，就运行Reset()函数，但是这里Reset()函数其实是空的哈，跟丢了以后我们其实啥也不做
            break;
        }

        last_frame_ = current_frame_;
        return true;
    }

    // 在执行Track之前，需要明白，Track究竟在做一件什么事情
    // Track是当前帧和上一帧之间进行的匹配
    // 而初始化是某一帧左右目（双目）之间进行的匹配
    bool Steps::Track()
    {
        // 先看last_frame_是不是正常存在的
        if (last_frame_)
        {
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose()); // 用匀速模型给current_frame_当前帧的位姿设置一个初值
        }

        int num_track_last = TrackLastFrame();     // 使用光流法得到前后两帧之间匹配特征点并返回匹配数（前后两帧都只用左目图像）
        tracking_inliers_ = EstimateCurrentPose(); // 接下来根据跟踪到的内点的匹配数目，可以分类进行后续操作，优化当前帧的位置

        if (tracking_inliers_ > num_features_tracking_)
        {
            // tracking good
            status_ = ConstructionStatus::TRACKING_GOOD;
        }
        else if (tracking_inliers_ > num_features_tracking_bad_)
        {
            // tracking bad
            status_ = ConstructionStatus::TRACKING_BAD;
        }
        else
        {
            // lost
            status_ = ConstructionStatus::LOST;
        }

        InsertKeyframe();                                                          // 在此函数里面判断当前要不要插入关键帧
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse(); // 更新当前帧和上一帧的位置差,也就是ΔT,变量名是relative_motion_

        return true;
    }

    bool Steps::InsertKeyframe()
    {
        // 需要前后帧追踪的匹配点大于一定数量才可以成为匹配点
        if (tracking_inliers_ >= num_features_needed_for_keyframe_)
        {
            // still have enough features, don't insert keyframe,可以节约计算资源
            return false;
        }
        // current frame is a new keyframe,在内点数少于80个的时候插入关键帧
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                  << current_frame_->keyframe_id_; // frame有自身的id,他作为keyframe也会有自己的id

        SetObservationsForKeyFrame(); // 添加关键帧的路标点
        DetectFeatures();             // 对当前帧提取新的GFTT特征点,检测当前关键帧的左目特征点

        // track in right image
        FindFeaturesInSecond(); // 接着匹配右目特征点
        // triangulate map points
        TriangulateNewPoints(); // 三角化新特征点并加入到地图中去
        // 因为添加了新的关键帧，所以在后端里面 运行 Backend::UpdateMap() 更新一下局部地图，启动一次局部地图的BA优化
        UpdateMap();

        return true;
    }
    // map_point_就是路标点
    void Steps::SetObservationsForKeyFrame()
    {
        for (auto &feat : current_frame_->features_img_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
                mp->AddObservation(feat);
        }
    }

    // 当一个新的关键帧到来后，我们势必需要补充一系列新的特征点，
    // 此时则需要像建立初始地图一样，对这些新加入的特征点进行三角化，求其3D位置,三角化要考虑去畸变,之后进行
    int Steps::TriangulateNewPoints(Frame::Ptr frame_one, Frame::Ptr frame_two)
    {
        std::vector<SE3> poses{frame_one->pose_, frame_two->pose_};
        SE3 current_pose_Twc_one = frame_one->Pose().inverse(); // camera to world
        SE3 current_pose_Twc_two = frame_two->Pose().inverse();
        int cnt_triangulated_pts = 0; // 三角化成功的点的数目
        for (size_t i = 0; i < frame_one->features_img_.size(); ++i)
        {
            // 遍历左目的特征点
            if (frame_one->features_img_[i]->map_point_.expired() &&
                frame_two->features_img_[i] != nullptr)
            {
                // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
                std::vector<Vec3> points{
                    camera_one_->pixel2camera(
                        Vec2(frame_one->features_img_[i]->position_.pt.x,
                             frame_one->features_img_[i]->position_.pt.y)),
                    camera_two_->pixel2camera(
                        Vec2(frame_two->features_img_[i]->position_.pt.x,
                             frame_two->features_img_[i]->position_.pt.y))};
                Vec3 pworld = Vec3::Zero();

                if (triangulation(poses, points, pworld) && pworld[2] > 0)
                {
                    auto new_map_point = MapPoint::CreateNewMappoint();
                    // 注意这里与初始化地图不同 triangulation计算出来的点pworld，
                    // 实际上是相机坐标系下的点，所以需要乘以一个TWC
                    // 但是初始化地图时，一般以第一幅图片为世界坐标系
                    pworld = current_pose_Twc_one * pworld;
                    cv::Vec3d cvVector;
                    cvVector[0] = pworld(0);
                    cvVector[1] = pworld(1);
                    cvVector[2] = pworld(2);
                    new_map_point->SetPos(cvVector); // 设置mapoint类中的坐标
                    new_map_point->AddObservation(
                        frame_one->features_img_[i]); // 增加mappoint类中的对应的那个feature（左右目）
                    new_map_point->AddObservation(
                        frame_two->features_img_[i]);

                    frame_one->features_img_[i]->map_point_ = new_map_point;
                    frame_two->features_img_[i]->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    // ceres优化单张图片的代码,因为要求出这张图片的初始位姿
    int Steps::EstimateCurrentPose(Frame::Ptr frame_one)
    {
        // setup g2o
        ceres::Problem problem; // 构建ceres的最小二乘问题
        // K
        Mat33 K = camera_one_->K(); // Camera类的成员函数K()

        // edges 边是地图点(3d世界坐标)在当前帧的投影位置(像素坐标)
        int index = 1;
        std::vector<Feature::Ptr> features; // features 存储的是one相机的特征点
        for (const auto featured : frame_one->features_img_)
        {
            auto mp = featured->map_point_.lock(); // weak_ptr是有lock()函数的
            if (mp)
            {
                ceres::CostFunction *cost_function;                             // 代价函数
                ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0); // 鲁棒核函数
                features.push_back(featured);
                ISfM::Feature &feature_extract_match = *featured;
                cost_function = SnavelyReprojectionError::Create(feature_extract_match.position_.pt.x,
                                                                 feature_extract_match.position_.pt.y);
                index++;
                problem.AddResidualBlock(cost_function, loss_function, camera, point); // 添加误差项
            }
        }

        // estimate the Pose the determine the outliers
        const double chi2_th = 5.991; // 重投影误差边界值，大于这个就设置为outline
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            // 总共优化了40遍，以10遍为一个优化周期，对outlier进行一次判断
            // 舍弃掉outlier的边，随后再进行下一个10步优化
            vertex_pose->setEstimate(current_frame_->Pose()); // 这里的顶点是SE3位姿,待优化的变量
            optimizer.initializeOptimization();
            optimizer.optimize(10); // 每次循环迭代10次
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i)
            {
                auto e = edges[i];
                if (features[i]->is_outlier_)
                { // 特征点本身就是异常点，计算重投影误差
                    e->computeError();
                }
                // （信息矩阵对应的范数）误差超过阈值，判定为异常点，并计数，否则恢复为正常点
                if (e->chi2() > chi2_th)
                { // chi2代表卡方检验
                    features[i]->is_outlier_ = true;
                    // 设置等级  一般情况下g2o只处理level = 0的边，设置等级为1，下次循环g2o不再优化异常值
                    // 这里每个边都有一个level的概念，
                    // 默认情况下，g2o只处理level=0的边，在orbslam中，
                    // 如果确定某个边的重投影误差过大，则把level设置为1，
                    // 也就是舍弃这个边对于整个优化的影响
                    e->setLevel(1);
                    cnt_outlier++;
                }
                else
                {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);
                };
                // 后20次不设置鲁棒核函数了，意味着此时不太可能出现大的异常点
                if (iteration == 2)
                {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
                  << features.size() - cnt_outlier;
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate()); // 保存优化后的位姿

        LOG(INFO) << "Current Pose = \n"
                  << current_frame_->Pose().matrix();
        // 清除异常点 但是只在feature中清除了
        // mappoint中仍然存在，仍然有使用的可能
        for (auto &feat : features)
        {
            if (feat->is_outlier_)
            {
                feat->map_point_.reset();  //.reset()方法的作用是将该弱引用指针设置为空nullptr,但不影响指向该对象的强引用数量,只会使得其弱引用数量减少
                feat->is_outlier_ = false; // maybe we can still use it in future
            }
        }
        return features.size() - cnt_outlier;
    }

    void Steps::Optimize(Map::KeyframesType &keyframes,
                                  Map::LandmarksType &landmarks)
    {
        // 主优化函数
        // 其实 Backend::Optimize()函数 和前端的 EstimateCurrentPose() 函数流有点类似，
        // 不同的地方是，在前端做这个优化的时候，只有一个顶点，也就是仅有化当前帧位姿这一个变量，
        // 因此边也都是一元边。在后端优化里面，局部地图中的所有关键帧位姿和地图点都是顶点，
        // 边也是二元边，在 g2o_types.h 文件中 class EdgeProjection 的 linearizeOplus()函数中，
        // 新增了一项 重投影误差对地图点的雅克比矩阵，187页，公式(7.48)
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer; // 创建稀疏优化器
        optimizer.setAlgorithm(solver); // 打开调试输出

        // pose 顶点，使用Keyframe id
        std::map<unsigned long, VertexPose *> vertices; // https://www.cnblogs.com/yimeixiaobai1314/p/14375195.html map和unordered_map
        unsigned long max_kf_id = 0;
        for (auto &keyframe : keyframes)
        { // 遍历关键帧   确定第一个顶点
            auto kf = keyframe.second;
            VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
            vertex_pose->setId(kf->keyframe_id_);
            vertex_pose->setEstimate(kf->Pose()); // keyframe的pose(SE3)是待估计的第一个对象
            optimizer.addVertex(vertex_pose);
            if (kf->keyframe_id_ > max_kf_id)
            {
                max_kf_id = kf->keyframe_id_;
            }

            vertices.insert({kf->keyframe_id_, vertex_pose}); // 插入自定义map类型的vertices 不要make_pair也可以嘛
        }

        // 路标顶点，使用路标id索引
        std::map<unsigned long, VertexXYZ *> vertices_landmarks;

        // K 和左右外参
        Mat33 K = cam_left_->K();
        SE3 left_ext = cam_left_->pose();
        SE3 right_ext = cam_right_->pose();

        // edges
        int index = 1;
        double chi2_th = 5.991; // robust kernel 阈值
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features;
        // std::pair主要的两个成员变量是first和second
        for (auto &landmark : landmarks)
        { // 遍历所有活动路标点,就是最多7个
            if (landmark.second->is_outlier_)
                continue;                                     // 外点不优化,first是id,second包括该点的位置,观测的若干相机的位姿,该点在各相机中的像素坐标
            unsigned long landmark_id = landmark.second->id_; // mappoint的id
            auto observations = landmark.second->GetObs();    // 得到所有观测到这个路标点的feature，是features
            for (auto &obs : observations)
            { // 遍历所有观测到这个路标点的feature，得到第二个顶点，形成对应的点边关系
                if (obs.lock() == nullptr)
                    continue; // 如果对象销毁则继续
                auto feat = obs.lock();
                // weak_ptr提供了expired()与lock()成员函数，前者用于判断weak_ptr指向的对象是否已被销毁，
                // 后者返回其所指对象的shared_ptr智能指针(对象销毁时返回”空”shared_ptr)
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr)
                    continue;

                auto frame = feat->frame_.lock(); // 得到该feature所在的frame
                EdgeProjection *edge = nullptr;
                if (feat->is_on_left_image_)
                { // 判断这个feature在哪个相机
                    edge = new EdgeProjection(K, left_ext);
                }
                else
                {
                    edge = new EdgeProjection(K, right_ext);
                }

                // 如果landmark还没有被加入优化，则新加一个顶点
                // 意思是无论mappoint被观测到几次，只与其中一个形成关系
                if (vertices_landmarks.find(landmark_id) ==
                    vertices_landmarks.end())
                {
                    VertexXYZ *v = new VertexXYZ;
                    v->setEstimate(landmark.second->Pos()); // Position in world，是作为estimate的第二个对象
                    v->setId(landmark_id + max_kf_id + 1);
                    v->setMarginalized(true); // 边缘化
                    // 简单的说G2O 中对路标点设置边缘化(Point->setMarginalized(true))是为了 在计算求解过程中，
                    // 先消去路标点变量，实现先求解相机位姿，然后再利用求解出来的相机位姿，反过来计算路标点的过程，
                    // 目的是为了加速求解，并非真的将路标点给边缘化掉。
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v); // 增加point顶点
                }

                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_));   // pose
                edge->setVertex(1, vertices_landmarks.at(landmark_id)); // landmark
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Mat22::Identity()); // e转置*信息矩阵*e,所以由此可以看出误差向量为n×1,则信息矩阵为n×n
                auto rk = new g2o::RobustKernelHuber();  // 定义robust kernel函数
                rk->setDelta(chi2_th);                   // 设置阈值
                // 设置核函数
                // 设置鲁棒核函数，之所以要设置鲁棒核函数是为了平衡误差，不让二范数的误差增加的过快。
                //  鲁棒核函数里要自己设置delta值，
                //  这个delta值是，当误差的绝对值小于等于它的时候，误差函数不变。否则误差函数根据相应的鲁棒核函数发生变化。
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, feat});

                optimizer.addEdge(edge); // 增加边

                index++;
            }
        }

        // do optimization and eliminate the outliers
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;
        while (iteration < 5)
        { // 确保内点占1/2以上，否则调整阈值，直到迭代结束
            cnt_outlier = 0;
            cnt_inlier = 0;
            // determine if we want to adjust the outlier threshold
            for (auto &ef : edges_and_features)
            {
                if (ef.first->chi2() > chi2_th)
                {
                    cnt_outlier++;
                }
                else
                {
                    cnt_inlier++;
                }
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5)
            {
                break;
            }
            else
            {
                chi2_th *= 2;
                iteration++;
            }
        }

        for (auto &ef : edges_and_features)
        { // 根据新的阈值，调整哪些是外点 ，并移除
            if (ef.first->chi2() > chi2_th)
            {
                ef.second->is_outlier_ = true;
                // remove the observation
                ef.second->map_point_.lock()->RemoveObservation(ef.second);
            }
            else
            {
                ef.second->is_outlier_ = false;
            }
        }

        LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
                  << cnt_inlier;

        // Set pose and lanrmark position，这样也就把后端优化的结果反馈给了前端
        for (auto &v : vertices)
        {
            keyframes.at(v.first)->SetPose(v.second->estimate()); // KeyframesType是unordered_map
        }                                                         // unordered_map.at()和unordered_map[]都用于引用给定位置上存在的元素
        for (auto &v : vertices_landmarks)
        {
            landmarks.at(v.first)->SetPos(v.second->estimate()); // landmarks:unordered_map<unsigned long, myslam::MapPoint::Ptr>
        }
    }

    ////////////////////////////////利用之前写的初始化函数进行优化过程的初始化///////////////////////////////
    bool Steps::Init(Initializer initialize_)
    {
        int num_features_left = DetectFeatures(); // 找左相机feature
        // 一个frame其实就是一个时间点，
        // 里面同时含有左，右目的图像，以及对应的feature的vector
        // 这一步在提取左目特征，通常在左目当中提取特征时特征点数量是一定能保证的。
        int num_coor_features = FindFeaturesInSecond();
        if (num_coor_features < num_features_init_)
        {
            return false;
        }

        bool build_map_success = BuildInitMap();
        if (build_map_success)
        {
            status_ = ConstructionStatus::TRACKING_GOOD;
            return true;
        }
        return false;
    }

    // 检测当前帧的做图的特征点，并放入feature的vector容器中
    int Steps::DetectFeatures()
    {
        // 掩膜，灰度图，同时可以看出，DetectFeatures是对左目图像的操作
        cv::Mat mask(current_frame_->features_img_.size(), CV_8UC1, 255);
        for (auto &feat : current_frame_->features_img_)
        {
            // 在已经存在特征点的地方，画一个20x20的矩形框，掩膜设置为0
            // 即在这个矩形区域中不提取特征了，保持均匀性，并避免重复
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }

        std::vector<cv::KeyPoint> keypoints;
        // detect函数，第三个参数是用来指定特征点选取区域的，一个和原图像同尺寸的掩膜，其中非0区域代表detect函数感兴趣的提取区域，
        // 相当于为detect函数明确了提取的大致位置
        gftt_->detect(current_frame_->img_d, keypoints, mask); // detect是opencv的函数,自带mask的选项
        int cnt_detected = 0;
        for (auto &kp : keypoints)
        {
            current_frame_->features_img_.push_back(
                Feature::Ptr(new Feature(current_frame_, kp))); // 检测到的新特征点的像素位置和当前帧关联起来
            cnt_detected++;
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected;
    }

    // 找到左目图像的feature之后，就在右目里面找特征点
    int Steps::FindFeaturesInSecond()
    {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_left, kps_right;
        for (auto &kp : current_frame_->features_img_)
        {
            // 遍历左目特征的特征点（feature）
            kps_left.push_back(kp->position_.pt); // feature类中的keypoint对应的point2f
            auto mp = kp->map_point_.lock();      // feature类中的mappoint
            if (mp)
            { // 如果当前特征点有在地图上有对应的点，那么将根据特征点的3D POSE和当前帧的位姿反求出特征点在当前帧的像素坐标
                // use projected points as initial guess
                Vec3 eigen_vec(mp->pos_[0], mp->pos_[1], mp->pos_[2]);
                auto px =
                    camera_two_->world2pixel(eigen_vec, current_frame_->Pose());
                kps_right.push_back(cv::Point2f(px[0], px[1]));
            }
            else
            { // 如果没有对应特征点，右目的特征点初值就是和左目一样
                // use same pixel in left image
                kps_right.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status; // 光流跟踪成功与否的状态向量（无符号字符），成功则为1,否则为0
        Mat error;
        // 进行光流跟踪，从这条opencv光流跟踪语句我们就可以知道，
        // 前面遍历左目特征关键点是为了给光流跟踪提供一个右目初始值
        // OPTFLOW_USE_INITIAL_FLOW使用初始估计，存储在nextPts中;
        // 如果未设置标志，则将prevPts复制到nextPts并将其视为初始估计。
        cv::calcOpticalFlowPyrLK(
            current_frame_->left_img_, current_frame_->right_img_, kps_left,
            kps_right, status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                             0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);
        // 右目中光流跟踪成功的点
        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                // KeyPoint构造函数中7代表着关键点直径
                cv::KeyPoint kp(kps_right[i], 7);
                Feature::Ptr feat(new Feature(current_frame_, kp));
                // 指明是右侧相机feature
                feat->is_on_left_image_ = false;
                current_frame_->features_right_.push_back(feat);
                num_good_pts++;
            }
            else
            {
                // 左右目匹配失败
                current_frame_->features_right_.push_back(nullptr); // 没匹配上就放个空指针
            }
        }
        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
    }

    // 初始地图
    bool Steps::BuildInitMap()
    {
        // 构造一个存储SE3的vector，里面初始化就放两个pose，一个左目pose，一个右目pose，
        // 看到这里应该记得，对Frame也有一个pose，Frame里面的
        // pose描述了固定坐标系（世界坐标系）和某一帧间的位姿变化
        std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
        size_t cnt_init_landmarks = 0; // 初始化的路标数目
        // 遍历左目的feature
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            if (current_frame_->features_right_[i] == nullptr)
                continue; // 右目没有对应点，之前设置左右目的特征点数量是一样的
            // create map point from triangulation
            // 对于左右目配对成功的点，三角化它
            // points中保存了双目像素坐标转换到相机（归一化）坐标
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))}; // 这里的depth默认为1.0
            // 待计算的世界坐标系下的点
            Vec3 pworld = Vec3::Zero();
            // 每一个同名点都进行一次triangulation
            // triangulation（）函数 相机位姿,某个feature左右目的坐标,三角化后的坐标保存
            if (triangulation(poses, points, pworld) && pworld[2] > 0)
            {
                // 根据前面存放的左右目相机pose和对应点相机坐标points进行三角化，得到对应地图点的深度，构造出地图点pworld
                // 需要对pworld进行判断，看其深度是否大于0, pworld[2]即是其深度。
                auto new_map_point = MapPoint::CreateNewMappoint(); // 工厂模式创建一个新的地图点
                new_map_point->SetPos(pworld);                      // mappoint类主要的数据成员  pos 以及 观测到的feature的vector
                // 为这个地图点添加观测量，这个地图点对应到了当前帧（应有帧ID）
                // 左目图像特征中的第i个以及右目图像特征中的第i个
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]);
                // 上两句是为地图点添加观测，这两句就是为特征类Feature对象填写地图点成员
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                cnt_init_landmarks++;
                // 对Map类对象来说，地图里面应当多了一个地图点，所以要将这个地图点加到地图中去
                map_->InsertMapPoint(new_map_point);
            }
        }
        // 当前帧能够进入初始化说明已经满足了初始化所需的帧特征数量，
        // 作为初始化帧，可看做开始的第一帧，所以应当是一个关键帧
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_); // 对Map类对象来说，地图里面应当多了一个关键帧，所以要将这个关键帧加到地图中去
        backend_->UpdateMap();

        LOG(INFO) << "Initial map created with " << cnt_init_landmarks
                  << " map points";

        return true;
    }

    bool Steps::Reset()
    {
        LOG(INFO) << "Reset is not implemented. ";
        return true;
    }
}
