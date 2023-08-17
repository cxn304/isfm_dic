#include <gflags/gflags.h>
#include "common.h"

DEFINE_string(config_file, "./config/default.yaml", "config file path");//这里的地址要搞清楚../

int main(int argc, char **argv) {

    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_config_file));//首先实例化了一个VisualOdometry类的类指针vo，传入config的地址
    // assert(vo->Init() == true);
    vo->Init();
    vo->Run();

    return 0;
}
