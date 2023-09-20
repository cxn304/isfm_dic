#include "dense.h"

namespace ISfM
{
    bool readDatasetFiles(
        const string &path,
        vector<string> &color_image_files,
        std::vector<SE3d> &poses,
        cv::Mat &ref_depth)
    {
        ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
        // For example, ifstream fin("abc.txt") will open the file “abc.txt” for reading1.
        // You can also use the open() member function to associate a file with fin, such as fin.open("abc.txt")
        if (!fin)
            return false;

        while (fin)
        {
            // fin.eof() is a member function of ifstream that returns true if the end-of-file
            // indicator is set for fin
            //  数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW,下标是从右读到左
            string image;
            fin >> image; // 把读到的东西push进入image变量里面
            double data[7];
            for (double &d : data)
                fin >> d;

            color_image_files.push_back(path + string("/images/") + image); // file name
            poses.push_back(
                SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                     Vector3d(data[0], data[1], data[2])));
            if (!fin.good())
                break; // 一行8个数据
        }
        fin.close();

        // load reference depth
        fin.open(path + "/depthmaps/scene_000.depth");
        ref_depth = cv::Mat(height, width, CV_64F);
        if (!fin)
            return false;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                double depth = 0;                            // 初始化depth
                fin >> depth;                                // 读出来的depth
                ref_depth.ptr<double>(y)[x] = depth / 100.0; // cv::Mat::ptr<double>(int i0)
            }

        return true;
    };

    void readFramesFromYAML(const std::string &filename, vector<Frame::Ptr> &frames)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);

        if (!fs.isOpened())
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        cv::FileNode framesNode = fs["frame"];

        for (cv::FileNodeIterator it = framesNode.begin(); it != framesNode.end(); ++it)
        {
            Frame frame;

            cv::FileNode frameNode = *it;

            // 读取图像名称
            std::string imgName;
            frameNode["img_name"] >> imgName;
            frame.img_name = imgName;

            // 读取内参
            cv::Mat intrinsics;
            frameNode["intrinsics"] >> intrinsics;
            cv::cv2eigen(intrinsics, frame.intrix_);

            // 读取位姿
            cv::Mat pose;
            frameNode["pose"] >> pose;
            Eigen::Matrix4d poseMatrix;
            cv::cv2eigen(pose, poseMatrix);
            frame.pose_ = Sophus::SE3d(poseMatrix);

            frames.push_back(frame);
        }

        fs.release();
    };

    // 对整个深度图进行更新
    bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2)
    {
        for (int x = boarder; x < width - boarder; x++)
            for (int y = boarder; y < height - boarder; y++)
            {
                // 遍历每个像素,cov表示方差
                if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) // 深度已收敛或发散
                    continue;
                // 在极线上搜索 (x,y) 的匹配
                Vector2d pt_curr;
                Vector2d epipolar_direction;
                bool ret = epipolarSearch(
                    ref,
                    curr,
                    T_C_R,
                    Vector2d(x, y),
                    depth.ptr<double>(y)[x],
                    sqrt(depth_cov2.ptr<double>(y)[x]),
                    pt_curr,
                    epipolar_direction);

                if (ret == false) // 匹配失败
                    continue;
                // 取消该注释以显示匹配
                // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);
                // 匹配成功，更新深度图，由上面求出了极线方向，reference到camera的变换，匹配成功的当前图像的像素坐标
                // 匹配成功的reference图像的x,y。最后更新depth和depth_cov2
                updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
            }
        return true;
    };

    bool epipolarSearch(
        const Mat &ref, const Mat &curr,
        const SE3d &T_C_R, const Vector2d &pt_ref,
        const double &depth_mu, const double &depth_cov,
        Vector2d &pt_curr, Vector2d &epipolar_direction)
    {
        Vector3d f_ref = px2cam(pt_ref);   // 归一化平面的相机坐标系
        f_ref.normalize();                 // normalize之后三个数的平方和为1
        Vector3d P_ref = f_ref * depth_mu; // 参考帧的 P 向量,当前相机坐标系的坐标乘以当前点的深度
        // reference转换到当前camera：T_C_R
        Vector2d px_mean_curr = cam2px(T_C_R * P_ref);                             // 按深度均值投影的像素到当前camera拍的图像
        double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov; // 3 sigma原则
        if (d_min < 0.1)
            d_min = 0.1;
        Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min)); // 按最小深度投影的像素
        Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max)); // 按最大深度投影的像素

        // 最小深度投影的像素和最大深度投影像素之间的连线就是epipolar line
        Vector2d epipolar_line = px_max_curr - px_min_curr; // 极线（线段形式）
        epipolar_direction = epipolar_line;                 // 极线方向,与极线本身一致,之后要归一化
        epipolar_direction.normalize();                     // 像素的长度乘以极线方向,得到x和y方向的长度
        double half_length = 0.5 * epipolar_line.norm();    // 极线线段的半长度,norm是求长度
        if (half_length > 100)
            half_length = 100; // 我们不希望搜索太多东西

        // 取消此句注释以显示极线（线段）
        // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

        // 在极线上搜索，以深度均值点为中心，左右各取半长度
        double best_ncc = -1.0;
        Vector2d best_px_curr;
        for (double l = -half_length; l <= half_length; l += 0.7)
        {                                                             // l+=sqrt(2)
            Vector2d px_curr = px_mean_curr + l * epipolar_direction; // 待匹配点
            if (!inside(px_curr))                                     // 自己写的函数,判断点是否在大图的边框内,即去除图像四边一定距离的区域
                continue;
            // 计算待匹配点与参考帧的 NCC
            double ncc = NCC(ref, curr, pt_ref, px_curr);
            if (ncc > best_ncc)
            {
                best_ncc = ncc;         // NCC是越大越好
                best_px_curr = px_curr; // 坐标，x在前，y在后
            }
        }
        if (best_ncc < 0.85f) // 只相信 NCC 很高的匹配，这里是一个超参数
            return false;
        pt_curr = best_px_curr;
        return true;
    };

    double NCC(const Mat &ref, const Mat &curr,const Vector2d &pt_ref, const Vector2d &pt_curr)
    {
        // 零均值-归一化互相关，pt_ref和pt_curr都是坐标
        // 先算均值
        double mean_ref = 0, mean_curr = 0;
        vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
        for (int x = -ncc_window_size; x <= ncc_window_size; x++)
            for (int y = -ncc_window_size; y <= ncc_window_size; y++)
            {
                double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
                mean_ref += value_ref;

                double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
                mean_curr += value_curr;

                values_ref.push_back(value_ref); // 把图像块内的灰度值都放到一个vector里面
                values_curr.push_back(value_curr);
            }

        mean_ref /= ncc_area; // double除以int
        mean_curr /= ncc_area;

        // 计算 Zero mean NCC
        double numerator = 0, demoniator1 = 0, demoniator2 = 0;
        for (int i = 0; i < values_ref.size(); i++)
        {
            double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr); // A(I,J)*B(I,J)
            numerator += n;
            demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);     // A(I,J)*A(I,J)
            demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr); // B(I,J)*B(I,J)
        }
        return numerator / sqrt(demoniator1 * demoniator2 + 1e-10); // 防止分母出现零
    };

    bool updateDepthFilter(
        const Vector2d &pt_ref,
        const Vector2d &pt_curr,
        const SE3d &T_C_R,
        const Vector2d &epipolar_direction,
        Mat &depth,
        Mat &depth_cov2)
    {
        // 用三角化计算深度
        SE3d T_R_C = T_C_R.inverse(); // SE3下的T矩阵通过inverse可以倒转,李代数的方便之处
        Vector3d f_ref = px2cam(pt_ref);
        f_ref.normalize(); // 两个特征点的归一化坐标,见7.5节
        Vector3d f_curr = px2cam(pt_curr);
        f_curr.normalize(); // normalize 之后使得x到相机主点的距离为1， 这样系数s就是我们要求的深度了

        // 方程，详细见7.5节
        // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC ： s2*x2=s1*R*x1+t
        // f2 = R_RC * f_cur
        // 转化成下面这个矩阵方程组
        // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
        //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
        Vector3d t = T_R_C.translation();
        Vector3d f2 = T_R_C.so3() * f_curr;
        Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
        Matrix2d A;
        A(0, 0) = f_ref.dot(f_ref);
        A(0, 1) = -f_ref.dot(f2); // u.dot(v):u和v的点乘，即对应元素乘积的和，返回一个标量
        A(1, 0) = -A(0, 1);
        A(1, 1) = -f2.dot(f2);
        Vector2d ans = A.inverse() * b;
        Vector3d xm = ans[0] * f_ref;            // ref 侧的结果
        Vector3d xn = t + ans[1] * f2;           // cur 结果，最后转到ref坐标系下，才好做平均
        Vector3d p_esti = (xm + xn) / 2.0;       // P的位置，取两者的平均
        double depth_estimation = p_esti.norm(); // 深度值

        // 另一种方法计算深度值,即p178里面的方法/////////////////////////////
        Matrix3d x2x = Sophus::SO3d::hat(f_curr); // 得到一个反对称矩阵
        Vector3d tmp = x2x * (T_C_R.so3() * f_ref);
        Vector3d right = x2x * t;
        double s1 = -right[0, 0] / tmp[0, 0];
        Vector3d youbian = s1 * (T_C_R.so3() * f_ref + T_C_R.translation());
        double s2 = youbian[0, 0] / f_curr[0, 0];
        // cout << s1 << ";" << depth_estimation << ";" << s2  << endl;
        /////////////////////////////////////////////////////////////////

        // 计算不确定性（以一个像素为误差）
        Vector3d p = f_ref * depth_estimation;
        Vector3d a = p - t; // 式12.7
        double t_norm = t.norm();
        double a_norm = a.norm();
        double alpha = acos(f_ref.dot(t) / t_norm);                   // acos是内置函数,alpha=acos(p,t),cos就是两个向量点乘再除以他们的模
        double beta = acos(-a.dot(t) / (a_norm * t_norm));            // beta=acos(a,-t)
        Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction); // 搜索的子区中心点对应的curr坐标系下的3d点
        f_curr_prime.normalize();
        double beta_prime = acos(f_curr_prime.dot(-t) / t_norm); // 12.8
        double gamma = M_PI - alpha - beta_prime;
        double p_prime = t_norm * sin(beta_prime) / sin(gamma); // 12.9
        double d_cov = p_prime - depth_estimation;              // 12.10
        double d_cov2 = d_cov * d_cov;

        // 高斯融合 12.6
        double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))]; // depth和depth_cov2一开始会给初值,然后迭代求解
        double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

        double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
        double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

        depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse; // 对depth和depth_cov2进行更新
        depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

        return true;
    };

    void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate)
    {
        double ave_depth_error = 0;    // 平均误差
        double ave_depth_error_sq = 0; // 平方误差
        int cnt_depth_data = 0;
        for (int y = boarder; y < depth_truth.rows - boarder; y++)
            for (int x = boarder; x < depth_truth.cols - boarder; x++)
            {
                double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
                ave_depth_error += error;
                ave_depth_error_sq += error * error;
                cnt_depth_data++;
            }
        ave_depth_error /= cnt_depth_data;
        ave_depth_error_sq /= cnt_depth_data;

        cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
    };

}