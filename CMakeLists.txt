cmake_minimum_required(VERSION 3.0)
project(ISfM)

set(CMAKE_BUILD_TYPE "Debug")
add_compile_options(-fPIC)
# set(CMAKE_CXX_FLAGS "-std=c++14 -Wall") # -Wall：开启所有警告(设置c++14标准,并设置编译提示)
# set(CMAKE_CXX_FLAGS_DEBUG  "-std=c++14 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)#设置依赖包的文件夹
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)#设置可执行文件生成位置
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)#设置库文件生成位置
set(CMAKE_MODULE_PATH "/home/g2o/cmake_modules")
set(CMAKE_MODULE_PATH"/usr/local/lib/cmake/Ceres")
############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# 设置 OpenCV 的路径
set(OpenCV_DIR "/etc/opencv4.6")
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog
# find_package(Glog REQUIRED)
# include_directories(${GLOG_INCLUDE_DIRS})

# gtest
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})

# gflags
# find_package(GFlags REQUIRED)
# include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# dbow3 
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension glog
        # ${GTEST_BOTH_LIBRARIES}
        # ${GLOG_LIBRARIES}
        # ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        ${DBoW3_LIBS}
        )#将需要的包,打包放到一个名称下,使得代码更简洁

enable_testing()#启用对当前目录或以下目录的测试

############### source and test ######################
#增加下级目录
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(app)
