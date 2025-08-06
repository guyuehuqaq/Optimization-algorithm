#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace Eigen{
    typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
}

struct Camera{
    Eigen::Matrix3d K;  // 内参
    Eigen::Matrix3d R;  // 外参，旋转矩阵
    Eigen::Vector3d T;  // 外参，唯一向量
};

struct Observation {
    Camera cam;
    Eigen::Vector2d uv;  // Observed 2D point
};

/*************
   计算投影矩阵
 *************/
Eigen::Matrix3x4d CalProjectMatrix(const Eigen::Matrix3d &K,
                                   const Eigen::Matrix3d &R,
                                   const Eigen::Vector3d &T) {
    Eigen::Matrix3x4d RT;
    RT.block<3, 3>(0, 0) = R;
    RT.col(3) = T;

    Eigen::Matrix3x4d P = K * RT;
    return P;
}

/*************
   计算三维点的投影点
 *************/
Eigen::Vector2d Project(const Camera& cam, const Eigen::Vector3d& point) {
    Eigen::Vector3d Pc = cam.R * point + cam.T;
    if (Pc.z() < 1e-6){
        std::cerr << "Warning: point behind camera or too close to plane, skipping observation.\n";
        return Eigen::Vector2d(0.0,0.0);
    }
    Eigen::Vector3d p_img = cam.K * Pc;
    return p_img.hnormalized();
}

/*************
   高斯牛顿优化
 *************/
void OptimizePointGN(Eigen::Vector3d& point3d,
                     const std::vector<Observation>& observations,
                     int max_iterations = 10){

    const double eps = 1e-6;
    for (int iter = 0; iter < max_iterations; ++iter){
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();  // 近似Hessian矩阵
        Eigen::Vector3d b = Eigen::Vector3d::Zero();  // 梯度项
        double total_error = 0.0;
        for (const auto& obs : observations){
            const auto& cam = obs.cam;
            // 计算投影点
            Eigen::Vector2d p_proj;
            p_proj = Project(cam, point3d);
            // 累计残差
            Eigen::Vector2d e = obs.uv - p_proj;
            total_error += e.squaredNorm();
            /********************************
             计算雅可比矩阵，残差对point3d的一阶雅克比矩阵。
             ∂π(Xc) / ∂X = ∂π(Xc) / ∂Xc ⋅ (∂Xc / ∂X)
             根据链式法则可以将∂π(Xc) / ∂X分为∂π(Xc) / ∂Xc 和 ∂Xc / ∂X
             X 是三维点在世界坐标系下的坐标；
             Xc 是该点在相机坐标系下的坐标；
             π(Xc) 是三维点在图像上的投影点（通常是归一化或像素坐标）；
             ∂π(Xc) / ∂Xc 是相机投影模型对相机系坐标的导数；
             ∂Xc / ∂X 是相机变换（如旋转 + 平移）对世界坐标的导数。
             结果:
                 ∂π/∂Xc =
                    [ fx / Zc      0           -fx * Xc / Zc²
                        0          fy / Zc     -fy * Yc / Zc² ];
                  ∂Xc / ∂X = R;
             *******************************/
            Eigen::Matrix<double, 2, 3> de_dPc;
            de_dPc << cam.K(0,0) / point3d.z(), 0, -cam.K(0,0) * point3d.x() / (point3d.z() * point3d.z()),
                      0, cam.K(1,1) / point3d.z(), -cam.K(1,1) * point3d.y() / (point3d.z() * point3d.z());
            // 链式法则：de/dX = de/dPc * dPc/dX = de/dPc * R
            Eigen::Matrix<double, 2, 3> J = de_dPc * cam.R;
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        // 求解线性方程 H * dx = b，得到更新量
        Eigen::Vector3d dx = H.ldlt().solve(b);

        // 判断是否收敛
        if (dx.norm() < eps) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }
        // 更新3D点
        point3d += dx;
        std::cout << "Iter " << iter << ": total error = " << total_error
                  << ", update norm = " << dx.norm() << "\n";
    }
}

int main(int argc, char **argv){

}