#include "steps.h"
#define SLAM_MODE

namespace ISfM
{
    Steps::Steps(const Initializer::Returns &returns,
                 const ImageLoader &Cimage_loader, Camera::Ptr &camera_one) : image_loader_(Cimage_loader),
                 camera_(camera_one)
    {
        intrinsic_ << returns.K_.at<double>(0, 0), returns.K_.at<double>(1, 1),
            returns.K_.at<double>(0, 2), returns.K_.at<double>(1, 2), 0.0, 0.0;
        frames_ = returns.frames_;
        map_ = returns.map_;
        matchesMap_ = returns.matchesMap_;
        std::vector<bool> myVector(matchesMap_.size(), false);
        matchesMapRetriangulated = myVector;
        camera_->setIntrinsic(intrinsic_);
        auto tmp_points = map_->GetAllMapPoints();
        count_feature_point(tmp_points);
    };

    // 在增加某一帧时,根据目前的状况选择不同的处理函数
    bool Steps::AddFrame(Frame::Ptr frame)
    {
        current_frame_ = frame; // Track()是Frontend的成员函数,status_是Frontend的数据,可以直接使用
        Track();                // 在TRACKING_GOOD和TRACKING_BAD的时候都执行Track函数
        return true;
    }

    // 在执行Track之前,需要明白,Track究竟在做一件什么事情,Track是当前帧和上一帧之间进行的匹配
    bool Steps::Track()
    {
        // relative motion,不一定方向是对的,如果当前图向右转,下一张图相对向左转,初始误差会大
        if (last_frame_)
        {
            current_frame_->SetPose(last_frame_->Pose()); // 给current_frame_当前帧的位姿设置一个初值
        }
        int num_track_last = FindFeaturesInSecond();
        tracking_inliers_ = EstimateCurrentPose(); // 接下来根据跟踪到的内点的匹配数目,可以分类进行后续操作,优化当前帧的位置

        InsertKeyframe(); // 在此函数里面判断当前要不要插入关键帧
        // relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse(); // 更新当前帧和上一帧的位置差,也就是ΔT,变量名是relative_motion_

        return true;
    }

    bool Steps::InsertKeyframe()
    {
        // current frame is a new keyframe,在内点数少于80个的时候插入关键帧
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        std::cout << " Set frame " << current_frame_->id_ << " as keyframe "
                  << current_frame_->keyframe_id_ << std::endl; // frame有自身的id,他作为keyframe也会有自己的id

        SetObservationsForKeyFrame(); // 添加关键帧的路标点

        // triangulate map points
        TriangulateNewPoints(last_frame_, current_frame_); // 三角化新特征点并加入到地图中去
        auto tmp_points = map_->GetAllMapPoints();
        count_feature_point(tmp_points); // 这里第一次加入新帧观测数量有问题
        return true;
    }
    ////////////////////////////////////////////////////////////////////////////////
    std::vector<cv::DMatch> Steps::findMatch(int id1, int id2, bool &reverse,
                                             std::vector<std::pair<std::pair<int, int>, std::vector<cv::DMatch>>> &matchesVec_)
    {
        std::vector<cv::DMatch> vectorToFind;
        std::pair<int, int> keyToFind;
        if (id1 > id2)
        {
            keyToFind = std::make_pair(id2, id1);
            reverse = true;
        }
        else
            keyToFind = std::make_pair(id1, id2);
        // 循环查找特定的 std::vector<cv::DMatch>
        for (const auto &pairs : matchesVec_)
        {
            if (pairs.first == keyToFind)
            {
                vectorToFind = pairs.second; // 找到匹配的键
                break;
            }
        }
        return vectorToFind; // 在这里使用 vectorToFind
    }

    // map_point_就是路标点////////////////////
    void Steps::SetObservationsForKeyFrame()
    {
        for (auto &feat : current_frame_->features_img_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
                mp->AddObservation(feat);
        }
    }
    // 从matchesMap_里面寻找已经注册过的一对对图像,对这些图像进行三角化
    int Steps::ReTriangulate()
    {
        int cnt_triangulated_pts = 0; // 三角化成功的点的数目
        std::pair<int, int> keyToFind;
        int index = 0;
        for (const auto &matchs : matchesMap_)
        {
            if(matchesMapRetriangulated[index]){
                index++; // 减少判断时间,已经用这两幅图三角化的点就不要继续三角化了
                continue;
            }
            keyToFind = matchs.first; // 这里的顺序是从小到大排列的
            if (frames_[keyToFind.first]->is_registed && frames_[keyToFind.second]->is_registed)
            {
                vector<SE3> poses = {frames_[keyToFind.first]->pose_,frames_[keyToFind.second]->pose_};
                cv::Mat image = cv::imread(frames_[keyToFind.first]->img_name, cv::IMREAD_COLOR);
                vector<cv::DMatch> this_match = matchs.second;
                for (auto &mt : this_match)
                {
                    int queryIdx, trainIdx; // 取出对应的特征点索引
                    queryIdx = mt.queryIdx;
                    trainIdx = mt.trainIdx;
                    // 根据索引获取特征点
                    std::shared_ptr<Feature> lastFeature = frames_[keyToFind.first]->features_img_[queryIdx];
                    std::shared_ptr<Feature> currentFeature = frames_[keyToFind.second]->features_img_[trainIdx];

                    // 遍历两个frame match的所有点
                    if (lastFeature->map_point_.expired() &&
                        currentFeature->map_point_.expired()) // expired()为true的话表示关联的std::shared_ptr已经被销毁
                    {
                        // 编号小的图的特征点未关联地图点且编号大的图未关联地图点,尝试三角化
                        std::vector<Vec3> points{
                            camera_->pixel2camera(
                                Vec2(lastFeature->position_.pt.x,
                                     lastFeature->position_.pt.y)),
                            camera_->pixel2camera(
                                Vec2(currentFeature->position_.pt.x,
                                     currentFeature->position_.pt.y))};
                        Vec3 pworld = Vec3::Zero(); // points第一个是编号小的图对应的相机

                        if (triangulation(poses, points, pworld) && pworld[2] > 0)
                        {
                            auto new_map_point = MapPoint::CreateNewMappoint();
                            // 这里的poses是Tcw 形式Pose, world to camera, 所以pworld是在世界坐标系下
                            uchar *pixel = image.ptr<uchar>(lastFeature->position_.pt.y, lastFeature->position_.pt.x);
                            Eigen::Matrix<uchar, 3, 1> color;
                            color << pixel[2], pixel[1], pixel[0];
                            new_map_point->SetColor(color);
                            cv::Vec3d cvVector;
                            cvVector[0] = pworld(0);
                            cvVector[1] = pworld(1);
                            cvVector[2] = pworld(2);
                            new_map_point->SetPos(cvVector); // 设置mapoint类中的坐标
                            new_map_point->AddObservation(
                                lastFeature); // 增加mappoint类中的对应的那个feature（左右目）
                            new_map_point->AddObservation(
                                currentFeature);

                            lastFeature->map_point_ = new_map_point;
                            currentFeature->map_point_ = new_map_point;
                            map_->InsertMapPoint(new_map_point);
                            cnt_triangulated_pts++;
                        }
                    }
                }
                matchesMapRetriangulated[index] = true;
            }
            index++; // 不论有没有改变matchesMapRetriangulated,index都要自增
        }
        std::cout << "re triangulate new landmarks: " << cnt_triangulated_pts << std::endl;
        return cnt_triangulated_pts;
    };

    // 新帧进来,需要对这些新加入的特征点进行三角化,求其3D位置,三角化先不考虑去畸变
    int Steps::TriangulateNewPoints(Frame::Ptr &frame_last, Frame::Ptr &frame_current)
    {
        cv::Mat image = cv::imread(frame_last->img_name, cv::IMREAD_COLOR);
        int cnt_triangulated_pts = 0; // 三角化成功的点的数目
        bool reverse = false;
        vector<cv::DMatch> this_match = findMatch(frame_last->id_, frame_current->id_, reverse, matchesMap_);
        vector<SE3> poses = {last_frame_->pose_,current_frame_->pose_};
        for (auto &mt : this_match)
        {
            int queryIdx, trainIdx; // 取出对应的特征点索引
            if (reverse == false)
            {
                queryIdx = mt.queryIdx;
                trainIdx = mt.trainIdx;
            }
            else
            {
                queryIdx = mt.trainIdx;
                trainIdx = mt.queryIdx;
            }
            // 根据索引获取特征点, feature就是lastframe中的feature,有可能是训练集里面的,上面的判断筛选过了
            std::shared_ptr<Feature> lastFeature = frame_last->features_img_[queryIdx];
            std::shared_ptr<Feature> currentFeature = frame_current->features_img_[trainIdx];

            // 遍历两个frame match的所有点
            if (lastFeature->map_point_.expired() &&
                currentFeature->map_point_.expired()) // expired()为true的话表示关联的std::shared_ptr已经被销毁
            {
                // lastFeature特征点未关联地图点且currentFeature未关联地图点,尝试三角化
                std::vector<Vec3> points{
                    camera_->pixel2camera(
                        Vec2(lastFeature->position_.pt.x,
                             lastFeature->position_.pt.y)),
                    camera_->pixel2camera(
                        Vec2(currentFeature->position_.pt.x,
                             currentFeature->position_.pt.y))};
                Vec3 pworld = Vec3::Zero(); // points第一个是编号小的图对应的相机

                if (triangulation(poses, points, pworld) && pworld[2] > 0)
                {
                    auto new_map_point = MapPoint::CreateNewMappoint();
                    // 这里的poses是Tcw 形式Pose, world to camera, 所以pworld是在世界坐标系下
                    uchar *pixel = image.ptr<uchar>(lastFeature->position_.pt.y, lastFeature->position_.pt.x);
                    Eigen::Matrix<uchar, 3, 1> color;
                    color << pixel[2], pixel[1], pixel[0];
                    new_map_point->SetColor(color);
                    cv::Vec3d cvVector;
                    cvVector[0] = pworld(0);
                    cvVector[1] = pworld(1);
                    cvVector[2] = pworld(2);
                    new_map_point->SetPos(cvVector); // 设置mapoint类中的坐标
                    new_map_point->AddObservation(
                        lastFeature); // 增加mappoint类中的对应的那个feature（左右目）
                    new_map_point->AddObservation(
                        currentFeature);

                    lastFeature->map_point_ = new_map_point;
                    currentFeature->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        std::cout << "new frame's new landmarks: " << cnt_triangulated_pts << std::endl;
        return cnt_triangulated_pts;
    }

    // 找出当前frame中特征点与上一张frame拥有同一个3d点的函数,并关联
    int Steps::FindFeaturesInSecond()
    {
        unsigned l_id = last_frame_->id_;
        unsigned c_id = current_frame_->id_;
        unsigned js = 0;
        bool reverse = false;
        std::vector<cv::DMatch> this_match = findMatch(l_id, c_id, reverse, matchesMap_);
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &mt : this_match)
        {
            int queryIdx, trainIdx; // 取出对应的特征点索引,如果不反转,照常取,如果反转,trainIdx和queryIdx反转
            if (reverse == false)
            {
                queryIdx = mt.queryIdx;
                trainIdx = mt.trainIdx;
            }
            else
            {
                queryIdx = mt.trainIdx;
                trainIdx = mt.queryIdx; // queryIdx和trainIdx已经自适应了,queryIdx要放到标号小的图上
            }
            std::shared_ptr<Feature> lastFeature = last_frame_->features_img_[queryIdx]; // 根据索引获取特征点
            // 如果上一帧中的该特征点不是空指针,则将该帧中的这个特征点的mappoint赋值
            if (lastFeature->map_point_.lock())
            {
                auto mp = lastFeature->map_point_.lock();
                Vec3 positions;
                cv::cv2eigen(mp->pos_, positions);
                auto currentPx = camera_->world2pixel(positions, current_frame_->Pose()); // 当前帧的Pose
                std::shared_ptr<Feature> currentFeature = current_frame_->features_img_[trainIdx];
                currentFeature->map_point_ = lastFeature->map_point_;
                // lastFeature->map_point_.lock()->AddObservation(currentFeature);
                js++;
                kps_last.push_back(lastFeature->position_.pt);
                kps_current.push_back(cv::Point2f(currentPx[0], currentPx[1]));
            }
        }
        return js; // 第一次新增的AddObservation数量也是对的上的
    }

    // ceres优化单张图片的代码,因为要求出这张图片的初始位姿
    int Steps::EstimateCurrentPose()
    {
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType; // pose is 6 dof, landmarks is 3 dof
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        // optimizer.setVerbose(true); // 开启优化信息输出
        optimizer.setAlgorithm(solver);

        // vertex,顶点是当前帧到上一帧的位姿变化T
        VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.addVertex(vertex_pose);
        // K
        Mat33 K = Mat33::Zero();
        K(0, 0) = intrinsic_[0];
        K(1, 1) = intrinsic_[1];
        K(2, 2) = 1.0;
        K(0, 2) = intrinsic_[2];
        K(1, 2) = intrinsic_[3];
        // edges 边是地图点(3d世界坐标)在当前帧的投影位置(像素坐标)
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features; // features存储的是相机的特征点
        // 在此之前要把last_frame的map_point和current_frame的mappoint联系起来
        // 这一段代码是将新frame中所有与之前图像有共同观测点的feature找出来
        for (auto &c_feature : current_frame_->features_img_)
        {
            auto mp = c_feature->map_point_.lock();
            if (mp)
            {
                features.push_back(c_feature);
                Vec3 positions;
                cv::cv2eigen(mp->pos_, positions);

                EdgeProjectionPoseOnly *edge =
                    new EdgeProjectionPoseOnly(positions, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose); // 只有一个顶点,第一个数是0
                edge->setMeasurement(
                    toVec2(c_feature->position_.pt)); // 测量值是图像上的点
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
        const double chi2_th = 10.991; // 重投影误差边界值,大于这个就设置为outline
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration)
        {
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

        std::cout << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
                  << features.size() - cnt_outlier << endl;
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate()); // 保存优化后的位姿

        std::cout << "Current Pose = \n"
                  << current_frame_->Pose().matrix() << std::endl;
        // 清除异常点,这里的feature是新帧中的feature
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

// #define SFM_MODE // 优化内参的
#ifdef SFM_MODE
    void Steps::localBA(Map::KeyframesType &keyframes,
                        Map::LandmarksType &landmarks)
    {
        // 局部地图中的所有关键帧位姿和地图点都是顶点,相机内参也是顶点,边是自定义边,在 g2o_types.h
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic>> BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        // 选取优化过程中使用的梯度下降算法,类似于ceres里面的loss function的那个函数的选择
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(
                std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer; // 创建稀疏优化器
        optimizer.setVerbose(true);     // 开启优化信息输出
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
        double chi2_th = 10.991; // robust kernel 阈值
        std::map<EdgeReprojectionIntrisic *, Feature::Ptr> edges_and_features;
        // std::pair主要的两个成员变量是first和second
        for (auto &landmark : landmarks)
        { // 遍历所有活动路标点
            if (landmark.second->is_outlier_)
                continue;                                     // 外点不优化,first是id,second包括该点的位置,观测的若干相机的位姿,该点在各相机中的像素坐标
            unsigned long landmark_id = landmark.second->id_; // mappoint的id
            auto observations = landmark.second->GetObs();    // 得到所有观测到这个路标点的feature,是features
            for (auto &obs : observations)
            { // 遍历所有观测到这个路标点的feature,得到第二个顶点,形成对应的点边关系
                if (obs.lock() == nullptr)
                    continue; // 如果对象销毁则继续
                auto feat = obs.lock();
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr || !feat->frame_.lock()->is_keyframe_)
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
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v); // 增加point顶点
                }

                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_ + 1)); // pose!!!!!!!!id!!!!!!!!!
                edge->setVertex(1, vertices_landmarks.at(landmark_id));   // landmark
                edge->setVertex(2, vertex_intrinsics);                    // 内参,始终是只有一个先
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Mat22::Identity()); // e转置*信息矩阵*e,所以由此可以看出误差向量为n×1,则信息矩阵为n×n
                auto rk = new g2o::RobustKernelHuber();  // 定义robust kernel函数
                rk->setDelta(chi2_th);                   // 设置阈值
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
                    cnt_outlier++;
                else
                    cnt_inlier++;
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5)
                break;
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
                ef.second->is_outlier_ = false;
        }

        std::cout << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
                  << cnt_inlier << std::endl;
        double total_reprojection_error = 0.0;
        int num_reprojections = 0;
        for (auto &ef : edges_and_features)
        {
            double reprojection_error = ef.first->chi2();
            total_reprojection_error += reprojection_error;
            num_reprojections++;
        }
        double average_reprojection_error = total_reprojection_error / num_reprojections;
        average_reprojection_error_.push_back(average_reprojection_error);
        // Set pose and lanrmark position,这样也就把后端优化的结果反馈给了前端
        for (auto &v : vertices)
        {
            // 因为之前的id都加了1,这里的id都要减1
            keyframes.at(v.first - 1)->SetPose(v.second->estimate()); // KeyframesType是unordered_map
        }                                                             // unordered_map.at()和unordered_map[]都用于引用给定位置上存在的元素
        for (auto &v : vertices_landmarks)
        {
            cv::Vec3d lms(v.second->estimate()(0), v.second->estimate()(1), v.second->estimate()(2));
            landmarks.at(v.first)->SetPos(lms); // landmarks:unordered_map<unsigned long, MapPoint::Ptr>
        }
        setIntrinsic(vertex_intrinsics->estimate());
        camera_->setIntrinsic(intrinsic_); // 优化完成后要更新相机内参
    };

#else // 不优化内参的
    void Steps::localBA(Map::KeyframesType &keyframes,
                        Map::LandmarksType &landmarks)
    {
        typedef g2o::BlockSolver_6_3 BlockSolverType; // pose is 6 dof, landmarks is 3 dof
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(
                std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        //optimizer.setVerbose(true);     // 开启优化信息输出
        optimizer.setAlgorithm(solver); // 打开调试输出
        // pose 顶点，使用Keyframe id
        std::map<unsigned long, VertexPose *> vertices;
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

            vertices.insert({kf->keyframe_id_, vertex_pose}); // 插入自定义map类型的vertices
        }
        // 路标顶点，使用路标id索引
        std::map<unsigned long, VertexXYZ *> vertices_landmarks;
        // K
        Mat33 K = Mat33::Zero();
        K(0, 0) = intrinsic_[0];
        K(1, 1) = intrinsic_[1];
        K(2, 2) = 1.0;
        K(0, 2) = intrinsic_[2];
        K(1, 2) = intrinsic_[3];
        // edges
        int index = 1;
        double chi2_th = 10.991; // robust kernel 阈值
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features;
        for (auto &landmark : landmarks)
        {
            if (landmark.second->is_outlier_)
                continue;                                     // 外点不优化,first是id,second包括该点的位置,观测的若干相机的位姿,该点在各相机中的像素坐标
            unsigned long landmark_id = landmark.second->id_; // mappoint的id
            auto observations = landmark.second->GetObs();    // 得到所有观测到这个路标点的feature，是features
            for (auto &obs : observations)
            { // 遍历所有观测到这个路标点的feature，得到第二个顶点，形成对应的点边关系
                if (obs.lock() == nullptr)
                    continue; // 如果对象销毁则继续
                auto feat = obs.lock();
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr || !feat->frame_.lock()->is_keyframe_)
                    continue;

                auto frame = feat->frame_.lock(); // 得到该feature所在的frame
                EdgeProjection *edge = nullptr;
                edge = new EdgeProjection(K);
                // 如果landmark还没有被加入优化，则新加一个顶点
                // 意思是无论mappoint被观测到几次，只与其中一个形成关系
                if (vertices_landmarks.find(landmark_id) ==
                    vertices_landmarks.end())
                {
                    VertexXYZ *v = new VertexXYZ;
                    Eigen::Matrix<double, 3, 1> pose_tmp;
                    pose_tmp << landmark.second->Pos()[0], landmark.second->Pos()[1], landmark.second->Pos()[2];
                    v->setEstimate(pose_tmp); // Position in world，是作为estimate的第二个对象
                    v->setId(landmark_id + max_kf_id + 1);
                    v->setMarginalized(true); // 边缘化
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
                    cnt_outlier++;
                else
                    cnt_inlier++;
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5)
                break;
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
                // remove the observation,这里只remove一个feature
                ef.second->map_point_.lock()->RemoveObservation(ef.second);
            }
            else
                ef.second->is_outlier_ = false;
        }

        cout << "****************************Outlier/Inlier in optimization: " << cnt_outlier << "/"
             << cnt_inlier << "****************************" << endl;

        // Calculate and print average reprojection error
        double total_reprojection_error = 0.0;
        int num_reprojections = 0;
        for (auto &ef : edges_and_features)
        {
            double reprojection_error = ef.first->chi2();
            total_reprojection_error += reprojection_error;
            num_reprojections++;
        }
        double average_reprojection_error = total_reprojection_error / num_reprojections;
        average_reprojection_error_.push_back(average_reprojection_error);

        // Set pose and lanrmark position，这样也就把后端优化的结果反馈给了前端
        for (auto &v : vertices)
        {
            keyframes.at(v.first)->SetPose(v.second->estimate()); // KeyframesType是unordered_map
        }                                                         // unordered_map.at()和unordered_map[]都用于引用给定位置上存在的元素
        for (auto &v : vertices_landmarks)
        {
            cv::Vec3d lms(v.second->estimate()(0), v.second->estimate()(1), v.second->estimate()(2));
            landmarks.at(v.first)->SetPos(lms); // landmarks:unordered_map<unsigned long, MapPoint::Ptr>
        }
    }
#endif

    void Steps::gloabalBA(Map::LandmarksType &landmarks)
    {
        // 所有位姿和地图点都是顶点,相机内参也是顶点,边是自定义边,在 g2o_types.h
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic>> BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(
                std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer; // 创建稀疏优化器
        optimizer.setVerbose(true);     // 开启优化信息输出
        optimizer.setAlgorithm(solver); // 打开调试输出

        // 内参,只加进去一个, id一直是0
        VertexIntrinsics *vertex_intrinsics = new VertexIntrinsics();
        vertex_intrinsics->setId(0);                // 内参id始终是0
        vertex_intrinsics->setEstimate(intrinsic_); // 内参设定为step类内内参,这里假设相机都是同一个
        // 添加内参节点
        optimizer.addVertex(vertex_intrinsics);
        // pose 顶点,使用所有frame id
        std::map<unsigned long, VertexPose *> vertices;
        unsigned long max_ff_id = 1;

        for (auto &frame : frames_)
        { // 遍历所有帧,确定第一个顶点,注意!!!!!这里由于有了内参设置id=0, 所以keyframe_id_统一加1
            auto ff = frame;
            auto ff_id = ff->id_ + 1;
            VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
            vertex_pose->setId(ff_id);
            vertex_pose->setEstimate(ff->Pose()); // keyframe的pose(SE3)是待估计的第一个对象
            optimizer.addVertex(vertex_pose);
            if (ff_id > max_ff_id)
            {
                max_ff_id = ff_id;
            }

            vertices.insert({ff_id, vertex_pose}); // 插入自定义map类型的vertices
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
                continue;                                     // 外点不优化,first是id,second包括该点的位置,观测的若干相机的位姿,该点在各相机中的像素坐标
            unsigned long landmark_id = landmark.second->id_; // mappoint的id
            auto observations = landmark.second->GetObs();    // 得到所有观测到这个路标点的feature,是features
            for (auto &obs : observations)
            { // 遍历所有观测到这个路标点的feature,得到第二个顶点,形成对应的点边关系
                if (obs.lock() == nullptr)
                    continue; // 如果对象销毁则继续
                auto feat = obs.lock();
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
                    v->setId(landmark_id + max_ff_id);
                    v->setMarginalized(true); // 边缘化
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v); // 增加point顶点
                }

                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_ + 1)); // pose!!!!!!!!id!!!!!!!!!
                edge->setVertex(1, vertices_landmarks.at(landmark_id));   // landmark
                edge->setVertex(2, vertex_intrinsics);                    // 内参,始终是只有一个先
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Mat22::Identity()); // e转置*信息矩阵*e,所以由此可以看出误差向量为n×1,则信息矩阵为n×n
                auto rk = new g2o::RobustKernelHuber();  // 定义robust kernel函数
                rk->setDelta(chi2_th);                   // 设置阈值
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
                    cnt_outlier++;
                else
                    cnt_inlier++;
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5)
                break;
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
                ef.second->is_outlier_ = false;
        }

        std::cout << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
                  << cnt_inlier << std::endl;
        double total_reprojection_error = 0.0;
        int num_reprojections = 0;
        for (auto &ef : edges_and_features)
        {
            double reprojection_error = ef.first->chi2();
            total_reprojection_error += reprojection_error;
            num_reprojections++;
        }
        double average_reprojection_error = total_reprojection_error / num_reprojections;
        cout << "final average reprojection error is: " << average_reprojection_error << endl;
        // Set pose and lanrmark position,这样也就把后端优化的结果反馈给了前端
        for (auto &v : vertices)
        {
            // 因为之前的id都加了1,这里的id都要减1
            frames_.at(v.first - 1)->SetPose(v.second->estimate()); // KeyframesType是unordered_map
        }                                                           // unordered_map.at()和unordered_map[]都用于引用给定位置上存在的元素
        for (auto &v : vertices_landmarks)
        {
            cv::Vec3d lms(v.second->estimate()(0), v.second->estimate()(1), v.second->estimate()(2));
            landmarks.at(v.first)->SetPos(lms); // landmarks:unordered_map<unsigned long, MapPoint::Ptr>
        }
        setIntrinsic(vertex_intrinsics->estimate());
        camera_->setIntrinsic(intrinsic_); // 优化完成后要更新相机内参
    };

    void Steps::count_feature_point(Map::LandmarksType &landmarks)
    {
        int mp_f_num = 0;
        int obs1 = 0;
        for (auto &mp : landmarks)
        {
            mp_f_num += mp.second->observed_times_;
            if (mp.second->observations_.size() == 1)
                obs1++;
        }
        cout << "observed time is 1: " << obs1 << endl;
        cout << "all map number is: " << landmarks.size() << ", all observed feature:" << mp_f_num << endl;
        for (auto &framed : frames_)
        {
            if (framed->is_keyframe_)
            {
                int f0n = 0;
                auto f0 = framed->features_img_;
                for (auto &tmp : f0)
                {
                    if (!tmp->map_point_.expired())
                        f0n++;
                }
                cout << "frame" << framed->id_ << " mappoint is: " << f0n << endl;
            }
        }
        std::cout << "intrinsic: " << std::endl
                  << intrinsic_ << std::endl;
    }
}
