# 使用file()命令获取指定文件夹下的所有源文件
file(GLOB SRC_FILES "*.cpp")
add_library(ISfM SHARED
        dataset.cpp
        )
# 创建库
# add_library(ISfM SHARED ${SRC_FILES})

target_link_libraries(ISfM
        ${THIRD_PARTY_LIBS})#链接依赖库和自己的库
