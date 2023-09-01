#include "steps.h"

namespace ISfM
{
    Steps::Steps(const Initializer::Returns &returns,
     const ImageLoader &Cimage_loader, Camera::Ptr &camera_one) : 
     image_loader_(Cimage_loader),camera_one_(camera_one)
    {
        intrinsic_ << returns.K_.at<double>(0, 0), returns.K_.at<double>(1, 1),
            returns.K_.at<double>(0, 2), returns.K_.at<double>(1, 2), 0.0, 0.0;
        features_ = returns.features_;
        frames_ = returns.frames_;
        map_ = returns.map_;
        matchesMap_ = returns.matchesMap_;
        camera_one_->setIntrinsic(intrinsic_);
    };
    // 在增加某一帧时,根据目前的状况选择不同的处理函数
    bool Steps::AddFrame(Frame::Ptr frame)
    {
        current_frame_ = frame;
        // Track()是Frontend的成员函数,status_是Frontend的数据,可以直接使用
        switch (status_)
        {
        case ConstructionStatus::TRACKING_GOOD:
        case ConstructionStatus::TRACKING_BAD:
            Track(); // 在TRACKING_GOOD和TRACKING_BAD的时候都执行Track函数
            break;
        case ConstructionStatus::LOST:
            break;
        }
        last_frame_ = current_frame_;
        return true;
    }

    // 在执行Track之前,需要明白,Track究竟在做一件什么事情,Track是当前帧和上一帧之间进行的匹配
    bool Steps::Track()
    {
        // 先看last_frame_是不是正常存在的
        if (last_frame_)
        {
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose()); // 用匀速模型给current_frame_当前帧的位姿设置一个初值
        }
        tracking_inliers_ = EstimateCurrentPose(); // 接下来根据跟踪到的内点的匹配数目,可以分类进行后续操作,优化当前帧的位置

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
        // current frame is a new keyframe,在内点数少于80个的时候插入关键帧
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        cout << "Set frame " << current_frame_->id_ << " as keyframe "
             << current_frame_->keyframe_id_; // frame有自身的id,他作为keyframe也会有自己的id

        SetObservationsForKeyFrame(); // 添加关键帧的路标点

        // triangulate map points
        TriangulateNewPoints(last_frame_, current_frame_); // 三角化新特征点并加入到地图中去

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

    // 当一个新的关键帧到来后,我们势必需要补充一系列新的特征点,
    // 此时则需要像建立初始地图一样,对这些新加入的特征点进行三角化,求其3D位置,三角化要考虑去畸变,之后进行
    int Steps::TriangulateNewPoints(Frame::Ptr &frame_one, Frame::Ptr &frame_two)
    {
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
                trainFeature != nullptr)
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
                    // 注意这里与初始化地图不同 triangulation计算出来的点pworld,
                    // 实际上是相机坐标系下的点,所以需要乘以一个TWC
                    // 但是初始化地图时,一般以第一幅图片为世界坐标系
                    pworld = current_pose_Twc_one * pworld;
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
        cout << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    // ceres优化单张图片的代码,因为要求出这张图片的初始位姿
    int Steps::EstimateCurrentPose()
    {
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType; // pose is 6 dof, landmarks is 3 dof
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::unique_ptr<BlockSolverType>(new BlockSolverType(
                std::unique_ptr<LinearSolverType>(new LinearSolverType()))));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // vertex,顶点是当前帧到上一帧的位姿变化T
        VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.addVertex(vertex_pose);
        // K
        Mat33 K = Mat33::Zero();
        K(0,0) = intrinsic_[0];
        K(1,1) = intrinsic_[1];
        K(2,2) = 1.0;
        K(0,2) = intrinsic_[2];
        K(1,2) = intrinsic_[3];
        // edges 边是地图点(3d世界坐标)在当前帧的投影位置(像素坐标)
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features; // features 存储的是相机的特征点
        //!!!!在此之前要把last_frame的map_point和current_frame的mappoint联系起来
        for (size_t i = 0; i < current_frame_->features_img_.size(); ++i)
        {
            auto mp = current_frame_->features_img_[i]->map_point_.lock(); // weak_ptr是有lock()函数的
            if (mp)
            {
                features.push_back(current_frame_->features_img_[i]);
                Vec3 poss_;
                cv::cv2eigen(mp->pos_, poss_);

                EdgeProjectionPoseOnly *edge =
                    new EdgeProjectionPoseOnly(poss_, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose); // 只有一个顶点,第一个数是0
                edge->setMeasurement(
                    toVec2(current_frame_->features_img_[i]->position_.pt)); // 测量值是图像上的点
                // 图中的Q就是信息矩阵,为了表示我们对误差各分量重视程度的不一样.
                //  一般情况下,我们都设置这个矩阵为单位矩阵,表示我们对所有的误差分量的重视程度都一样.
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber); // 鲁棒核函数
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }
        }

        // estimate the Pose the determine the outliers
        const double chi2_th = 5.991; // 重投影误差边界值,大于这个就设置为outline
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            // 总共优化了40遍,以10遍为一个优化周期,对outlier进行一次判断
            // 舍弃掉outlier的边,随后再进行下一个10步优化
            vertex_pose->setEstimate(current_frame_->Pose()); // 这里的顶点是SE3位姿,待优化的变量
            optimizer.initializeOptimization();
            optimizer.optimize(10); // 每次循环迭代10次
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i)
            {
                auto e = edges[i];
                if (features[i]->is_outlier_)
                { // 特征点本身就是异常点,计算重投影误差
                    e->computeError();
                }
                // （信息矩阵对应的范数）误差超过阈值,判定为异常点,并计数,否则恢复为正常点
                if (e->chi2() > chi2_th)
                { // chi2代表卡方检验
                    features[i]->is_outlier_ = true;
                    // 设置等级  一般情况下g2o只处理level = 0的边,设置等级为1,下次循环g2o不再优化异常值
                    // 这里每个边都有一个level的概念,默认情况下,g2o只处理level=0的边,
                    // 如果确定某个边的重投影误差过大,则把level设置为1,
                    // 也就是舍弃这个边对于整个优化的影响
                    e->setLevel(1);
                    cnt_outlier++;
                }
                else
                {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);
                };
                // 后20次不设置鲁棒核函数了,意味着此时不太可能出现大的异常点
                if (iteration == 2)
                {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        cout << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
             << features.size() - cnt_outlier;
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate()); // 保存优化后的位姿

        cout << "Current Pose = \n"
             << current_frame_->Pose().matrix();
        // 清除异常点 但是只在feature中清除了
        // mappoint中仍然存在,仍然有使用的可能
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
        // 主优化函数,在后端优化里面,局部地图中的所有关键帧位姿和地图点都是顶点,相机内参也是顶点
        // 边是自定义边,在 g2o_types.h
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic>> BlockSolverType;
        // 根据图优化类型，定义线性方程求解器,根据g2o的调用规则，这里需要使用智能指针
        std::unique_ptr<BlockSolverType::LinearSolverType> linearSolver(
            new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>());
        // 根据线性方程求解器定义矩阵块求解器
        std::unique_ptr<BlockSolverType> matSolver(new BlockSolverType(std::move(linearSolver)));
        // 选取优化过程中使用的梯度下降算法,类似于ceres里面的loss function的那个函数的选择
        g2o::OptimizationAlgorithmLevenberg *solver =
            new g2o::OptimizationAlgorithmLevenberg(std::move(matSolver));
        g2o::SparseOptimizer optimizer; // 创建稀疏优化器
        optimizer.setAlgorithm(solver); // 打开调试输出

        // 内参,只加进去一个, id一直是0
        VertexIntrinsics *vertex_intrinsics = new VertexIntrinsics();
        vertex_intrinsics->setId(0);                // 内参id始终是0
        vertex_intrinsics->setEstimate(intrinsic_); // 内参设定为step类内内参,这里假设相机都是同一个
        // 添加内参节点
        optimizer.addVertex(vertex_intrinsics);
        // pose 顶点,使用Keyframe id
        std::map<unsigned long, VertexPose *> vertices;
        unsigned long max_kf_id = 1;

        for (auto &keyframe : keyframes)
        { // 遍历关键帧,确定第一个顶点,注意!!!!!这里由于有了内参设置id=0, 所以keyframe_id_统一加1
            auto kf = keyframe.second;
            auto kf_id = kf->keyframe_id_ + 1;
            VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
            vertex_pose->setId(kf_id);
            vertex_pose->setEstimate(kf->Pose()); // keyframe的pose(SE3)是待估计的第一个对象
            optimizer.addVertex(vertex_pose);
            if (kf_id > max_kf_id)
            {
                max_kf_id = kf_id;
            }

            vertices.insert({kf_id, vertex_pose}); // 插入自定义map类型的vertices
        }

        // 路标顶点,使用路标id索引
        std::map<unsigned long, VertexXYZ *> vertices_landmarks;

        // edges
        int index = 1;
        double chi2_th = 5.991; // robust kernel 阈值
        std::map<EdgeReprojectionIntrisic *, Feature::Ptr> edges_and_features;
        // std::pair主要的两个成员变量是first和second
        for (auto &landmark : landmarks)
        { // 遍历所有活动路标点
            if (landmark.second->is_outlier_)
                continue;                                         // 外点不优化,first是id,second包括该点的位置,观测的若干相机的位姿,该点在各相机中的像素坐标
            unsigned long landmark_id = landmark.second->id_ + 1; // mappoint的id!!!!!!!!!加了1
            auto observations = landmark.second->GetObs();        // 得到所有观测到这个路标点的feature,是features
            for (auto &obs : observations)
            { // 遍历所有观测到这个路标点的feature,得到第二个顶点,形成对应的点边关系
                if (obs.lock() == nullptr)
                    continue; // 如果对象销毁则继续
                auto feat = obs.lock();
                // weak_ptr提供了expired()与lock()成员函数,前者用于判断weak_ptr指向的对象是否已被销毁,
                // 后者返回其所指对象的shared_ptr智能指针(对象销毁时返回”空”shared_ptr)
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr)
                    continue;

                auto frame = feat->frame_.lock(); // 得到该feature所在的frame
                EdgeReprojectionIntrisic *edge = nullptr;
                edge = new EdgeReprojectionIntrisic();

                // 如果landmark还没有被加入优化,则新加一个顶点
                // 意思是无论mappoint被观测到几次,只与其中一个形成关系
                if (vertices_landmarks.find(landmark_id) ==
                    vertices_landmarks.end())
                {
                    VertexXYZ *v = new VertexXYZ;
                    Eigen::Matrix<double, 3, 1> pose_tmp;
                    pose_tmp << landmark.second->Pos()[0], landmark.second->Pos()[1], landmark.second->Pos()[2];
                    v->setEstimate(pose_tmp); // Position in world,是作为estimate的第二个对象
                    v->setId(landmark_id + max_kf_id);
                    v->setMarginalized(true); // 边缘化
                    // 简单的说G2O 中对路标点设置边缘化(Point->setMarginalized(true))是为了在计算求解过程中,
                    // 先消去路标点变量,实现先求解相机位姿,然后再利用求解出来的相机位姿,反过来计算路标点的过程,
                    // 目的是为了加速求解,并非真的将路标点给边缘化掉.
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v); // 增加point顶点
                }

                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_ + 1)); // pose!!!!!!!!id!!!!!!!!!
                edge->setVertex(1, vertices_landmarks.at(landmark_id));   // landmark
                edge->setVertex(2, vertex_intrinsics);
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Mat22::Identity()); // e转置*信息矩阵*e,所以由此可以看出误差向量为n×1,则信息矩阵为n×n
                auto rk = new g2o::RobustKernelHuber();  // 定义robust kernel函数
                rk->setDelta(chi2_th);                   // 设置阈值
                // 设置核函数 之所以要设置鲁棒核函数是为了平衡误差,不让二范数的误差增加的过快.
                // 鲁棒核函数里要自己设置delta值,
                // 这个delta值是,当误差的绝对值小于等于它的时候,误差函数不变.否则误差函数根据相应的鲁棒核函数发生变化.
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
        { // 确保内点占1/2以上,否则调整阈值,直到迭代结束
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
        { // 根据新的阈值,调整哪些是外点 ,并移除
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

        cout << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
             << cnt_inlier;

        // Set pose and lanrmark position,这样也就把后端优化的结果反馈给了前端
        for (auto &v : vertices)
        {
            // 因为之前的id都加了1,这里的id都要减1
            keyframes.at(v.first - 1)->SetPose(v.second->estimate()); // KeyframesType是unordered_map
        }                                                             // unordered_map.at()和unordered_map[]都用于引用给定位置上存在的元素
        for (auto &v : vertices_landmarks)
        {
            cv::Vec3d lms(v.second->estimate()(0), v.second->estimate()(1), v.second->estimate()(2));
            landmarks.at(v.first - 1)->SetPos(lms); // landmarks:unordered_map<unsigned long, MapPoint::Ptr>
        }
        camera_one_->setIntrinsic(intrinsic_); // 优化完成后要更新相机内参
    };
}
