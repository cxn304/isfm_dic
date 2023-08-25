#include "map_builder.h"
#include "scene_graph.h"

namespace ISfM
{
    MapBuilder::MapBuilder(const std::string &database_path, const MapBuilder::Parameters &params)
        : database_path_(database_path), params_(params)
    {
    }

    void LoadRegisterGraphFromSceneGraph(cv::Ptr<SceneGraph> scene_grph, cv::Ptr<RegisterGraph> register_graph)
    {
        for (const auto &image_pair_el : scene_grph->ImagePairs())
        {
            int image_pair = image_pair_el.first;
            int image_id1;
            int image_id2;
            Database::PairIdToImagePair(image_pair, &image_id1, &image_id2);
            register_graph->AddEdge(image_id1, image_id2);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool MapBuilder::BuildInitMap()
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

    void MapBuilder::SetUp()
    {
        K_ = (cv::Mat_<double>(3, 3) << params_.fx, 0, params_.cx,
              0, params_.fy, params_.cy,
              0, 0, 1);
        dist_coef_ = (cv::Mat_<double>(4, 1) << params_.k1, params_.k2, params_.p1, params_.p2);

        initailizer_ = cv::Ptr<Initializer>(new Initializer(params_.init_params, K_));
        registrant_ = cv::Ptr<Registrant>(new Registrant(params_.regis_params, K_));
        triangulator_ = cv::Ptr<Triangulator>(new Triangulator(params_.tri_params, K_));

        cv::Ptr<Database> database = cv::Ptr<Database>(new Database());
        database->Open(database_path_);

        // 得到图像的大小
        Database::Image db_image = database->ReadImageById(0);
        cv::Mat image = cv::imread(db_image.name);
        height_ = image.rows;
        width_ = image.cols;

        timer.Start();
        // 加载scene graph
        scene_graph_ = cv::Ptr<SceneGraph>(new SceneGraph());
        scene_graph_->Load(database, params_.min_num_matches);

        const int kWidth = 30;
        std::cout.flags(std::ios::left); // 左对齐
        std::cout << std::endl;
        std::cout << std::setw(kWidth) << "Load Scene Graph ";

        // 加载register graph
        register_graph_ = cv::Ptr<RegisterGraph>(new RegisterGraph(scene_graph_->NumImages()));
        LoadRegisterGraphFromSceneGraph(scene_graph_, register_graph_);
        std::cout << std::setw(kWidth) << "Load Register Graph ";
        timer.PrintSeconds();

        // 加载map
        map_ = cv::Ptr<Map>(new Map(scene_graph_, height_, width_, K_, dist_coef_));
        map_->Load(database);
        std::cout << std::setw(kWidth) << "Load Map ";

        timer.PrintSeconds();
        database->Close();

        bundle_optimizer_ = cv::Ptr<CeresBundelOptimizer>(new CeresBundelOptimizer(params_.ba_params));

        if (params_.is_visualization)
            async_visualization_ = cv::Ptr<AsyncVisualization>(new AsyncVisualization());
    }

    //     void MapBuilder::doBuild()
    //     {

    //         timer_for_total_.Start();

    //         // data for visualization
    //         std::vector<cv::Point3f> points3D;
    //         std::vector<cv::Vec3b> colors;
    //         std::vector<cv::Mat> Rs;
    //         std::vector<cv::Mat> ts;

    //         if (params_.is_visualization)
    //         {
    //             async_visualization_->RunVisualizationThread();
    //         }

    //         // 初始化
    //         bool is_succeed = TryInitialize();
    //         if (!is_succeed)
    //         {
    //             // TODO : print something
    //             return;
    //         }

    //         map_->GetDataForVisualization(points3D, colors, Rs, ts);

    //         if (params_.is_visualization)
    //         {
    //             timer_for_visualization_.Start();
    //             async_visualization_->ShowPointCloud(points3D, colors);
    //             async_visualization_->ShowCameras(Rs, ts);
    //             timer_for_visualization_.Pause();
    //         }
    //         GlobalBA();
    //         FilterAllTracks();

    //         // 注册下一张图片
    //         size_t current_num_registed_images = 2;
    //         size_t prev_num_registed_images = 2;
    //         while (is_succeed)
    //         {

    //             const std::vector<int> &image_ids = register_graph_->GetNextImageIds();
    //             //        const std::vector<int>& image_ids = map_->GetNextImageIds();
    //             if (image_ids.size() == 0)
    //             {
    //                 break;
    //             }
    //             for (const int &image_id : image_ids)
    //             {
    //                 register_graph_->AddNumTrial(image_id);
    //                 std::cout << "==============================================================================" << std::endl;
    //                 std::cout << "Try To Register " << current_num_registed_images << "th Image --- image id : " << image_id << std::endl;
    //                 std::cout << "==============================================================================" << std::endl;
    //                 is_succeed = TryRegisterNextImage(image_id);

    //                 if (is_succeed)
    //                 {
    // #ifdef DEBUG

    //                     SetTimer(timer_for_debug_);
    //                     map_->Debug();
    //                     timer_for_debug_.Pause();
    // #endif
    //                     current_num_registed_images += 1;

    //                     if (params_.is_visualization && current_num_registed_images % 6 == 0)
    //                     {
    //                         timer_for_visualization_.Resume();

    //                         map_->GetDataForVisualization(points3D, colors, Rs, ts);
    //                         async_visualization_->ShowPointCloud(points3D, colors);
    //                         async_visualization_->ShowCameras(Rs, ts);

    //                         timer_for_visualization_.Pause();
    //                     }

    //                     if (current_num_registed_images >= params_.global_ba_ratio * prev_num_registed_images)
    //                     {
    //                         std::cout << "GLOBAL BA" << std::endl;
    //                         prev_num_registed_images = current_num_registed_images;
    //                         GlobalBA();
    //                         FilterAllTracks();
    //                     }
    //                     else
    //                     {
    //                         std::cout << "LOCAL BA" << std::endl;
    //                         LocalBA();
    //                         FilterTracks();
    //                         CompleteTracks();
    //                         MergeTracks();
    //                     }

    //                     break;
    //                 }
    //             }

    //             if (!is_succeed)
    //             {
    //                 break;
    //             }
    //         }

    //         if (current_num_registed_images != prev_num_registed_images)
    //         {
    //             GlobalBA();
    //             FilterAllTracks();
    //         }

    //         if (params_.is_visualization)
    //         {
    //             timer_for_visualization_.Resume();

    //             map_->GetDataForVisualization(points3D, colors, Rs, ts);
    //             async_visualization_->ShowPointCloud(points3D, colors);
    //             async_visualization_->ShowCameras(Rs, ts);

    //             timer_for_visualization_.Pause();
    //         }

    //         Summary();

    //         if (params_.is_visualization)
    //         {
    //             async_visualization_->WaitForVisualizationThread();
    //             async_visualization_->Close();
    //         }
    //     }
    //     }

    //     std::vector<int> MapBuilder::FindFirstInitialImage() const
    //     {
    //         struct ImageInfo
    //         {
    //             int image_id;
    //             int num_correspondences;
    //         };

    //         std::vector<ImageInfo> image_infos;

    //         std::vector<int> image_ids = scene_graph_->GetAllImageIds();
    //         for (int image_id : image_ids)
    //         {
    //             // 初始化失败过的图片就不作为初始化
    //             if (register_graph_->GetNumTrial(image_id) > 0)
    //                 continue;

    //             ImageInfo image_info;
    //             int num_correspondences = scene_graph_->NumCorrespondencesForImage(image_id);
    //             image_info.image_id = image_id;
    //             image_info.num_correspondences = num_correspondences;
    //             image_infos.push_back(image_info);
    //         }

    //         // 按照每张图片的correspondences, 从大到小排序
    //         std::sort(
    //             image_infos.begin(), image_infos.end(),
    //             [](const ImageInfo &image_info1, const ImageInfo &image_info2)
    //             {
    //                 return image_info1.num_correspondences > image_info2.num_correspondences;
    //             });

    //         image_ids.clear();
    //         for (const ImageInfo &image_info : image_infos)
    //         {
    //             image_ids.push_back(image_info.image_id);
    //         }
    //         return image_ids;
    //     }
    //     std::vector<int> MapBuilder::FindSecondInitialImage(int image_id) const
    //     {
    //         std::unordered_map<int, int> num_correspondences;
    //         int num_points2D = map_->NumPoints2DInImage(image_id);

    //         for (int point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx)
    //         {
    //             const std::vector<SceneGraph::Correspondence> &corrs =
    //                 scene_graph_->FindCorrespondences(image_id, point2D_idx);

    //             for (const SceneGraph::Correspondence &corr : corrs)
    //             {
    //                 num_correspondences[corr.image_id] += 1;
    //             }
    //         }
    //         struct ImageInfo
    //         {
    //             int image_id;
    //             int num_correspondences;
    //         };

    //         std::vector<ImageInfo> image_infos;

    //         for (const auto &elem : num_correspondences)
    //         {
    //             if (register_graph_->GetNumTrial(elem.first))
    //                 continue;

    //             ImageInfo image_info;
    //             image_info.image_id = elem.first;
    //             image_info.num_correspondences = elem.second;
    //             image_infos.push_back(image_info);
    //         }

    //         std::sort(
    //             image_infos.begin(), image_infos.end(),
    //             [](const ImageInfo &image_info1, const ImageInfo &image_info2)
    //             {
    //                 return image_info1.num_correspondences > image_info2.num_correspondences;
    //             });
    //         std::vector<int> image_ids;

    //         for (const ImageInfo &image_info : image_infos)
    //         {
    //             image_ids.push_back(image_info.image_id);
    //         }

    //         return image_ids;
    //     }

    //     bool MapBuilder::TryInitialize()
    //     {

    //         SetTimer(timer_for_initialize_);
    //         size_t trial = 0;
    //         std::vector<cv::Vec2d> points2D1;
    //         std::vector<cv::Vec2d> points2D2;
    //         std::vector<int> point2D_idxs1;
    //         std::vector<int> point2D_idxs2;

    //         const std::vector<int> &image_ids1 = FindFirstInitialImage();
    //         for (const int &image_id1 : image_ids1)
    //         {
    //             const std::vector<int> &image_ids2 = FindSecondInitialImage(image_id1);

    //             for (const int &image_id2 : image_ids2)
    //             {
    //                 trial += 1;
    //                 register_graph_->AddNumTrial(image_id1);
    //                 register_graph_->AddNumTrial(image_id2);

    //                 std::cout << "Try To Initialize : " << image_id1 << " - " << image_id2 << std::endl;
    //                 map_->Get2D2DCorrespoindencesBetweenImages(image_id1, image_id2, points2D1, points2D2, point2D_idxs1, point2D_idxs2);
    //                 Initializer::Statistics statistics = initailizer_->Initialize(points2D1, points2D2);

    //                 if (statistics.is_succeed)
    //                 {
    //                     register_graph_->SetRegistered(image_id1);
    //                     register_graph_->SetRegistered(image_id2);

    //                     // 更新map
    //                     map_->AddImagePose(image_id1, statistics.R1, statistics.t1);
    //                     map_->AddImagePose(image_id2, statistics.R2, statistics.t2);

    //                     for (size_t i = 0; i < statistics.inlier_mask.size(); ++i)
    //                     {
    //                         if (!statistics.inlier_mask[i])
    //                             continue;
    //                         int point2D_idx1 = point2D_idxs1[i];
    //                         int point2D_idx2 = point2D_idxs2[i];
    //                         Track track;
    //                         track.AddElement(image_id1, point2D_idx1);
    //                         track.AddElement(image_id2, point2D_idx2);

    //                         map_->AddPoint3D(statistics.points3D[i], track, statistics.residuals[i]);
    //                     }
    //                     timer_for_initialize_.Pause();
    //                     return true;
    //                 }

    //                 if (trial > params_.max_num_init_trials)
    //                 {
    //                     timer_for_initialize_.Pause();

    //                     return false;
    //                 }
    //             }
    //         }
    //         timer_for_initialize_.Pause();
    //         return false;
    //     }

    //     bool MapBuilder::TryRegisterNextImage(const int &image_id)
    //     {
    //         // Get 2D 3D correspondence

    //         SetTimer(timer_for_register_);
    //         register_graph_->AddNumTrial(image_id);

    //         std::vector<cv::Vec2d> points2D;
    //         std::vector<cv::Vec3d> points3D;
    //         std::vector<int> point2D_idxs;
    //         std::vector<point3D_t> point3D_idxs;
    //         map_->Get2D3DCorrespondences(image_id, points2D, points3D, point2D_idxs, point3D_idxs);

    //         Registrant::Statistics statistics = registrant_->Register(points3D, points2D);
    //         registrant_->PrintStatistics(statistics);

    //         timer_for_register_.Pause();

    //         if (statistics.is_succeed)
    //         {
    //             register_graph_->SetRegistered(image_id);
    //             map_->AddImagePose(image_id, statistics.R, statistics.t);

    //             // 更新track
    //             std::unordered_set<int> vis;
    //             for (size_t i = 0; i < statistics.inlier_mask.size(); ++i)
    //             {
    //                 if (!statistics.inlier_mask[i])
    //                     continue;

    //                 const int &point2D_idx = point2D_idxs[i];
    //                 const point3D_t &point3D_idx = point3D_idxs[i];

    //                 // 由于一个2D点， 可能对应多个3D点
    //                 // 所以需要这种来标记， 只添加一个2D-3D对应到map中
    //                 // 否则会出错
    //                 if (vis.count(point2D_idx) > 0)
    //                     continue;
    //                 vis.insert(point2D_idx);

    //                 TrackElement track_el(image_id, point2D_idx);
    //                 const double &error = statistics.residuals[i];
    //                 map_->AddObservation(point3D_idx, track_el, error);
    //             }

    //             SetTimer(timer_for_triangulate_);
    //             // 三角测量
    //             std::vector<std::vector<Map::CorrData>> points2D_corr_datas;
    //             map_->Get2D2DCorrespondences(image_id, points2D_corr_datas);
    //             double ave_residual = 0;
    //             size_t num_triangulated = Triangulate(points2D_corr_datas, ave_residual);

    //             const size_t width = 30;
    //             std::cout.flags(std::ios::left); // 左对齐
    //             std::cout << std::endl;
    //             std::cout << "--------------- Triangulate Summary Start ---------------" << std::endl;
    //             std::cout << std::setw(width) << "Num 2D 2D correspondences"
    //                       << " : " << points2D_corr_datas.size() << std::endl;
    //             std::cout << std::setw(width) << "Num triangulated "
    //                       << " : " << num_triangulated << std::endl;
    //             std::cout << std::setw(width) << "Ave residual "
    //                       << " : " << ave_residual << std::endl;
    //             std::cout << "--------------- Triangulate Summary End ---------------" << std::endl;
    //             std::cout << std::endl;
    //             timer_for_triangulate_.Pause();
    //         }

    //         return statistics.is_succeed;
    //     }

    //     size_t MapBuilder::Triangulate(const std::vector<std::vector<Map::CorrData>> &points2D_corr_datas,
    //                                    double &ave_residual)
    //     {
    //         size_t num_triangulated = 0;
    //         double sum_residual = 0;
    //         for (const auto &corr_datas : points2D_corr_datas)
    //         {
    //             std::vector<Map::CorrData> new_corr_datas;
    //             // 过滤掉已经有3D点对应的2D点
    //             for (const auto &corr : corr_datas)
    //             {
    //                 // 跳过没有注册的图像
    //                 if (!register_graph_->IsRegistered(corr.image_id))
    //                     continue;
    //                 // 跳过已经有3D点的匹配点
    //                 if (map_->HasPoint3DInImage(corr.image_id, corr.point2D_idx))
    //                     continue;
    //                 new_corr_datas.push_back(corr);
    //             }

    //             // 点数小于2， 无法进行三角测量
    //             if (new_corr_datas.size() < 2)
    //                 continue;

    //             // 多视图三角测量
    //             std::vector<cv::Mat> Rs;
    //             std::vector<cv::Mat> ts;
    //             std::vector<cv::Vec2d> points2D;

    //             for (const auto &new_corr : new_corr_datas)
    //             {
    //                 Rs.push_back(new_corr.R);
    //                 ts.push_back(new_corr.t);
    //                 points2D.push_back(new_corr.point2D);
    //             }

    //             Triangulator::Statistics statistics = triangulator_->Triangulate(Rs, ts, points2D);

    //             if (statistics.is_succeed)
    //             {
    //                 Track track;

    //                 for (const auto &new_corr : new_corr_datas)
    //                 {
    //                     track.AddElement(new_corr.image_id, new_corr.point2D_idx);
    //                 }
    //                 map_->AddPoint3D(statistics.point3D, track, statistics.ave_residual);

    //                 sum_residual += statistics.ave_residual;
    //                 num_triangulated += 1;
    //             }
    //         }
    //         ave_residual = sum_residual / num_triangulated;
    //         return num_triangulated;
    //     }

    //     void MapBuilder::LocalBA()
    //     {

    //         SetTimer(timer_for_local_ba_);
    //         BundleData bundle_data;
    //         map_->GetLocalBAData(bundle_data);
    //         bundle_optimizer_->Optimize(bundle_data);
    //         map_->UpdateFromBAData(bundle_data);
    //         timer_for_local_ba_.Pause();
    //         // for debug
    // #ifdef DEBUG
    //         struct Map::Statistics statistics = map_->Statistics();
    //         map_->PrintStatistics(statistics);
    // #endif
    //     }

    //     void MapBuilder::MergeTracks()
    //     {
    //         SetTimer(timer_for_merge_);
    //         map_->MergePoints3D(map_->GetModifiedPoint3DIds(), params_.merge_max_reproj_error);
    //         timer_for_merge_.Pause();
    //     }
    //     void MapBuilder::CompleteTracks()
    //     {
    //         SetTimer(timer_for_complete_);
    //         map_->CompletePoints3D(map_->GetModifiedPoint3DIds(), params_.complete_max_reproj_error);
    //         timer_for_complete_.Pause();
    //     }
    //     void MapBuilder::FilterTracks()
    //     {
    //         SetTimer(timer_for_local_filter_);
    //         map_->FilterPoints3D(map_->GetModifiedPoint3DIds(), params_.filtered_max_reproj_error, params_.filtered_min_tri_angle);
    //         timer_for_local_filter_.Pause();
    //     }

    //     void MapBuilder::GlobalBA()
    //     {
    //         SetTimer(timer_for_global_ba_);
    //         BundleData bundle_data;
    //         map_->GetGlobalBAData(bundle_data);
    //         bundle_optimizer_->Optimize(bundle_data);
    //         map_->UpdateFromBAData(bundle_data);
    //         timer_for_global_ba_.Pause();
    //         // for debug
    // #ifdef DEBUG
    //         struct Map::Statistics statistics = map_->Statistics();
    //         map_->PrintStatistics(statistics);
    // #endif
    //     }
    //     void MapBuilder::FilterAllTracks()
    //     {
    //         SetTimer(timer_for_global_filter_);
    //         map_->FilterAllPoints3D(params_.filtered_max_reproj_error, params_.filtered_min_tri_angle);
    //         timer_for_global_filter_.Pause();
    //     }

    //     void MapBuilder::WriteCOLMAP(const std::string &directory)
    //     {
    //         map_->WriteCOLMAP(directory);
    //     }
    //     void MapBuilder::WriteOpenMVS(const std::string &directory)
    //     {
    //         map_->WriteOpenMVS(directory);
    //     }

    //     void MapBuilder::WritePLY(const std::string &path)
    //     {
    //         map_->WritePLY(path);
    //     }

    //     void MapBuilder::WritePLYBinary(const std::string &path)
    //     {
    //         map_->WritePLYBinary(path);
    //     }
    //     void MapBuilder::Write(const std::string &path)
    //     {
    //         map_->Write(path);
    //     }
    //     void MapBuilder::WriteCamera(const std::string &path)
    //     {
    //         map_->WriteCamera(path);
    //     }
    //     void MapBuilder::WriteImages(const std::string &path)
    //     {
    //         map_->WriteImages(path);
    //     }
    //     void MapBuilder::WritePoints3D(const std::string &path)
    //     {
    //         map_->WritePoints3D(path);
    //     }
}
