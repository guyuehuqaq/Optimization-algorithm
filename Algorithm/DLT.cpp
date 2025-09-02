#include <Eigen/Dense>
#include <vector>
namespace Eigen {
    typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
}

/*********
 DLT双视角标准版
 基于DLT最小二乘线性代数求解，双视角三维重建
 最小的是代数误差，不是几何误差(重投影误差)
 *********/
Eigen::Vector3d TriangulatePoint(
        const Eigen::Matrix3x4d& cam1_from_world,
        const Eigen::Matrix3x4d& cam2_from_world,
        const Eigen::Vector2d& point1,
        const Eigen::Vector2d& point2) {
    Eigen::Matrix4d A;

    A.row(0) = point1(0) * cam1_from_world.row(2) - cam1_from_world.row(0);
    A.row(1) = point1(1) * cam1_from_world.row(2) - cam1_from_world.row(1);
    A.row(2) = point2(0) * cam2_from_world.row(2) - cam2_from_world.row(0);
    A.row(3) = point2(1) * cam2_from_world.row(2) - cam2_from_world.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    return svd.matrixV().col(3).hnormalized();
}

std::vector<Eigen::Vector3d> TriangulatePoints(
        const Eigen::Matrix3x4d& cam1_from_world,
        const Eigen::Matrix3x4d& cam2_from_world,
        const std::vector<Eigen::Vector2d>& points1,
        const std::vector<Eigen::Vector2d>& points2) {
    std::vector<Eigen::Vector3d> points3D(points1.size());

    for (size_t i = 0; i < points3D.size(); ++i) {
        points3D[i] = TriangulatePoint(
                cam1_from_world, cam2_from_world, points1[i], points2[i]);
    }

    return points3D;
}


/*********
 DLT多视角的扩展版本
 基于代数误差最小化的多视图几何三角化扩展形式，线性多视图三角化
 最小的是代数误差，不是几何误差(重投影误差)
 *********/
Eigen::Vector3d TriangulateMultiViewPoint(
        const std::vector<Eigen::Matrix3x4d>& cams_from_world,
        const std::vector<Eigen::Vector2d>& points) {
    if (cams_from_world.size() != points.size()) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    for (size_t i = 0; i < points.size(); i++) {
        const Eigen::Vector3d point = points[i].homogeneous().normalized();
        const Eigen::Matrix3x4d term =
                cams_from_world[i] - point * point.transpose() * cams_from_world[i];
        A += term.transpose() * term;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);
    return eigen_solver.eigenvectors().col(0).hnormalized();
}


