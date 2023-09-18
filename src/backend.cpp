
#include "backend.h"
#include "triangulate.h"
#include "feature.h"
#include "g2o_types.h"
#include "map.h"
#include "point3d.h"

namespace ISfM {

Backend::Backend() {
    backend_running_.store(true);// 构造函数中启动优化线程并挂起:通过原子操作实现
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();//notify_one()与notify_all()常用来唤醒阻塞的线程
}

void Backend::Stop() {
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

void Backend::BackendLoop() {
    while (backend_running_.load()) {//用 load() 函数进行读操作
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);
        /// 后端仅优化激活的Frames和Landmarks
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        localBA(active_kfs, active_landmarks);//开始做全局优化
    }
}

void Backend::localBA(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {

    typedef g2o::BlockSolver_6_3 BlockSolverType; // pose is 6 dof, landmarks is 3 dof
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(
                std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        // optimizer.setVerbose(true);     // 开启优化信息输出
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

            vertices.insert({kf->keyframe_id_, vertex_pose}); // 插入自定义map类型的vertices 不要make_pair也可以嘛
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
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr)
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
                if(!feat->map_point_.lock()) throw;
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
                // remove the observation
                ef.second->map_point_.lock()->RemoveObservation(ef.second);
            }
            else
                ef.second->is_outlier_ = false;
        }

        cout << "****************************Outlier/Inlier in optimization: " << cnt_outlier << "/"
             << cnt_inlier << endl;

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

}  // namespace myslam