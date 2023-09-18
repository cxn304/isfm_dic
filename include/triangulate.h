#ifndef TRIANGULATE_H
#define TRIANGULATE_H

// triangulation used in
#include "common.h"
namespace ISfM
{
    /**
     * linear triangulation with SVD, 由多个视角观测到的同一个点,重建出这个点
     * @param poses     poses,
     * @param points    points in normalized plane
     * @param pt_world  triangulated point in the world
     * @return true if success
     */
    inline bool triangulation(const std::vector<SE3> &poses,
                              const std::vector<Vec3> points, Vec3 &pt_world)
    {
        MatXX A(2 * poses.size(), 4);
        VecX b(2 * poses.size());
        b.setZero();
        for (size_t i = 0; i < poses.size(); ++i)
        {
            Mat34 m = poses[i].matrix3x4();
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }
        auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-1)
        {
            // 解质量不好,放弃
            return true;
        }
        return false;
    }

    inline Eigen::Vector3d TriangulatePoint(const Mat34 &proj_matrix1,
                                            const Mat34 &proj_matrix2,
                                            const Eigen::Vector2d &point1,
                                            const Eigen::Vector2d &point2)
    {
        Eigen::Matrix4d A;

        A.row(0) = point1(0) * proj_matrix1.row(2) - proj_matrix1.row(0);
        A.row(1) = point1(1) * proj_matrix1.row(2) - proj_matrix1.row(1);
        A.row(2) = point2(0) * proj_matrix2.row(2) - proj_matrix2.row(0);
        A.row(3) = point2(1) * proj_matrix2.row(2) - proj_matrix2.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

        return svd.matrixV().col(3).hnormalized();
    }

    inline Eigen::Vector3d TriangulateMultiViewPoint(
        const std::vector<Mat34> &proj_matrices,
        const std::vector<Eigen::Vector2d> &points)
    {
        Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

        for (size_t i = 0; i < points.size(); i++)
        {
            const Eigen::Vector3d point = points[i].homogeneous().normalized();
            const Mat34 term =
                proj_matrices[i] - point * point.transpose() * proj_matrices[i];
            A += term.transpose() * term;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

        return eigen_solver.eigenvectors().col(0).hnormalized();
    }

    // converters
    inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }
}
#endif
