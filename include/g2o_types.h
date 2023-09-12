
#ifndef G2O_TYPES_H
#define G2O_TYPES_H

#include "common.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace ISfM
{
    // 自定义相机内参顶点类(fx,fy,cx,cy,k1,k2)
    class VertexIntrinsics : public g2o::BaseVertex<6, Vec6>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override
        {
            _estimate.setZero();
        }

        virtual void oplusImpl(const double *update) override
        {
            // 在这里,v是一个常量向量类型,它通过将update指针映射为长度为VertexIntrinsics::Dimension的常量向量来创建
            Vec6::ConstMapType v(update, VertexIntrinsics::Dimension);
            _estimate += v;
        }

        virtual bool read(std::istream &) override { return false; }
        virtual bool write(std::ostream &) const override { return false; }
    };
    /// vertex and edges used in g2o ba
    /// 位姿顶点
    class VertexPose : public g2o::BaseVertex<6, SE3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override { _estimate = SE3(); }

        /// left multiplication on SE3
        virtual void oplusImpl(const double *update) override
        {
            Vec6 update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4],
                update[5];
            _estimate = SE3::exp(update_eigen) * _estimate;
        }

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }
    };

    /// 路标顶点
    class VertexXYZ : public g2o::BaseVertex<3, Vec3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        virtual void setToOriginImpl() override { _estimate = Vec3::Zero(); }

        virtual void oplusImpl(const double *update) override
        {
            _estimate[0] += update[0];
            _estimate[1] += update[1];
            _estimate[2] += update[2];
        }

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }
    };

    /// 仅估计位姿的一元边 <2, Vec2, VertexPose>,传入的是pos和k,
    /// 2是dimension of measurement,Vec2是type of measurement
    class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeProjectionPoseOnly(const Vec3 &pos, const Mat33 &K)
            : _pos3d(pos), _K(K) {}
        // 这里计算error,通过G2O定义的虚函数,下面两个函数
        virtual void computeError() override
        {
            const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
            SE3 T = v->estimate();
            Vec3 pos_pixel = _K * (T * _pos3d);
            pos_pixel /= pos_pixel[2];
            _error = _measurement - pos_pixel.head<2>();
        }
        // 这里计算雅可比矩阵
        virtual void linearizeOplus() override
        {
            const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
            SE3 T = v->estimate();
            Vec3 pos_cam = T * _pos3d;
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;
            _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                -fy * X * Zinv;
        } // 重投影误差对相机位姿（李代数）的一阶导数

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }

    private:
        Vec3 _pos3d;
        Mat33 _K;
    };

    /// 带有地图和位姿的二元边
    class EdgeProjection
        : public g2o::BaseBinaryEdge<2, Vec2, VertexPose, VertexXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /// 构造时传入相机内外参
        EdgeProjection(const Mat33 &K) : _K(K)
        {
        }

        virtual void computeError() override
        {
            const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
            const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);
            SE3 T = v0->estimate();
            Vec3 pos_pixel = _K * (T * v1->estimate());
            pos_pixel /= pos_pixel[2];
            _error = _measurement - pos_pixel.head<2>();
        }

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }

    private:
        Mat33 _K;
    };

    // 自定义重投影误差边类,包括内参外参和3维点
    class EdgeReprojectionIntrisic : public g2o::BaseFixedSizedEdge<2,
                                                                    Vec2,
                                                                    VertexPose,
                                                                    VertexXYZ,
                                                                    VertexIntrinsics>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeReprojectionIntrisic()
        {
        }

        virtual void computeError() override
        {
            const VertexPose *pose = static_cast<const VertexPose *>(vertex(0));
            const VertexXYZ *point = static_cast<const VertexXYZ *>(vertex(1));
            const VertexIntrinsics *intrinsics = static_cast<const VertexIntrinsics *>(vertex(2));

            SE3 T = pose->estimate(); // 这里的T是world to camera
            Vec3 pw = point->estimate();
            double fx = intrinsics->estimate()[0]; // fx
            double fy = intrinsics->estimate()[1]; // fy
            double cx = intrinsics->estimate()[2]; // cx
            double cy = intrinsics->estimate()[3]; // cy

            Vec3 pc = T * pw;
            Vec2 pos_camera_norm = pc.head<2>() / pc[2];

            // 畸变模型
            double k1 = intrinsics->estimate()[4];
            double k2 = intrinsics->estimate()[5];
            double r2 = pos_camera_norm.squaredNorm();
            double distortion = 1.0 + r2 * (k1 + k2 * r2);
            Vec2 pos_pixel_distorted = distortion * pos_camera_norm;
            Vec2 pos_pixel;
            pos_pixel[0] = fx * pos_pixel_distorted[0] + cx;
            pos_pixel[1] = fy * pos_pixel_distorted[1] + cy;
            _error = _measurement - pos_pixel;
        }

        virtual bool read(std::istream &in) override { return true; }
        virtual bool write(std::ostream &out) const override { return true; }

    private:
        Vec2 observed_;
    };

} // namespace myslam

#endif // MYSLAM_G2O_TYPES_H
