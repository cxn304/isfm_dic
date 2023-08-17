add_executable(feature_extract feature_extract.cpp)
set(CMAKE_BUILD_TYPE "Debug")
target_link_libraries(feature_extract ISfM ${THIRD_PARTY_LIBS})