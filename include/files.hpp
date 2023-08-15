#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>

class ImageLoader {
public:
    std::vector<std::string> filenames;
    
    ImageLoader(const std::string& path) : path_(path), num_images_(0) {
        // 获取文件名列表
        filenames = get_filenames(path);

        // 统计图片数量
        for (const std::string& filename : filenames) {
            if (is_image_file(filename)) {
                num_images_++;
            }
        }
    }

    int get_num_images() const {
        return num_images_;
    }

private:
    std::string path_;
    int num_images_;

    std::vector<std::string> get_filenames(const std::string& path) {
        std::vector<std::string> filenames;
        DIR* dirp = opendir(path.c_str());
        struct dirent* dp;
        while ((dp = readdir(dirp)) != nullptr) {
            if (dp->d_type == DT_REG) { // 如果是普通文件
                filenames.push_back(dp->d_name);
            }
        }
        closedir(dirp);
        return filenames;
    }

    bool is_image_file(const std::string& filename) {
        // 检查文件扩展名是否为jpg、png或bmp
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        return (ext == "jpg" || ext == "png" || ext == "bmp");
    }
};