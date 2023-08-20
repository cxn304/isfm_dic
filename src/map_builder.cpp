#include "map_builder.h"

namespace ISfM
{
    MapBuilder::MapBuilder(const std::string &database_path, const MapBuilder::Parameters &params)
        : database_path_(database_path), params_(params){};
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
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
        timer.PrintSeconds();

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
}
