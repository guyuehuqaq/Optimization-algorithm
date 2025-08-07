#include <Eigen/Dense>
#include <vector>
#include <iostream>

// 相机参数结构体
struct Camera {
    Eigen::Matrix3d R;   // 旋转矩阵（世界坐标系到相机坐标系）
    Eigen::Vector3d t;   // 平移向量
    Eigen::Matrix3d K;   // 内参矩阵
};

// 单个观测：相机 + 观测到的二维点
struct Observation {
    Camera cam;
    Eigen::Vector2d uv;  // 观测二维点（像素坐标）
};

// 投影函数：三维点 -> 像素点（u, v）
Eigen::Vector2d project(const Camera& cam, const Eigen::Vector3d& point) {
    Eigen::Vector3d Pc = cam.R * point + cam.t;   // 世界坐标 -> 相机坐标
    Eigen::Vector3d p_img = cam.K * Pc;           // 相机坐标 -> 像素坐标（齐次）
    return p_img.hnormalized();  // 齐次归一化
}

/****************************
 梯度计算公式：
f(X) = sum_{i=1}^N || u_i_obs - pi(R_i * X + t_i) ||^2
∇_X f(X) = ∂f(X) / ∂X = Σ_{i=1}^N [ -2 * J_i^T * (u_i_obs - π(R_i * X + t_i)) ]
J_i = ∂π(R_i * X + t_i) / ∂X ∈ R^{2×3}
 ****************************/

// 计算残差对三维点的梯度（Jacobian）
Eigen::Vector3d ComputeGradient(const Camera& cam,
                                const Eigen::Vector3d& point3D,
                                const Eigen::Vector2d& observed_uv) {
    Eigen::Vector3d Pc = cam.R * point3D + cam.t;
    double x = Pc(0), y = Pc(1), z = Pc(2);
    if (z < 1e-8) return Eigen::Vector3d::Zero();  // 防止除0

    // 计算投影点
    double u = cam.K(0,0) * x / z + cam.K(0,2);
    double v = cam.K(1,1) * y / z + cam.K(1,2);

    // 计算残差
    Eigen::Vector2d e(observed_uv(0) - u,
                      observed_uv(1) - v);

    // 链式法则，残差对三维点的梯度 = de/dPc * dPc/dX
    // de/dPc(不包含相机畸变)
    Eigen::Matrix<double, 2, 3> de_dPc;
    de_dPc << -cam.K(0,0) / z, 0, cam.K(0,0) * x / (z*z),
            0, -cam.K(1,1) / z, cam.K(1,1) * y / (z*z);

    // dPc/dX = R
    Eigen::Vector3d grad = de_dPc.transpose() * e;  // 先计算de/dPc的转置乘e
    grad = cam.R.transpose() * grad;                // 乘以R转置

    return grad * 2.0; // 因为是平方残差，对应链式法则乘2
}

// 梯度下降优化三维点位置
void OptimizePointGD(Eigen::Vector3d& point3D,
                     const std::vector<Observation>& observations,
                     int max_iters = 100,
                     double learning_rate = 1e-4,
                     double epsilon = 1e-6) {
    for (int iter = 0; iter < max_iters; ++iter) {
        Eigen::Vector3d grad_sum = Eigen::Vector3d::Zero();
        double total_error = 0;

        for (const auto& obs : observations) {
            Eigen::Vector3d Pc = obs.cam.R * point3D + obs.cam.t;
            double x = Pc(0), y = Pc(1), z = Pc(2);
            if (z < 1e-8) continue;
            double u = obs.cam.K(0,0) * x / z + obs.cam.K(0,2);
            double v = obs.cam.K(1,1) * y / z + obs.cam.K(1,2);
            Eigen::Vector2d e(obs.uv(0) - u, obs.uv(1) - v);
            total_error += e.squaredNorm();
            grad_sum += ComputeGradient(obs.cam, point3D, obs.uv);
        }

        point3D += learning_rate * grad_sum; // 更新点位置（注意是加，因为梯度计算时带负号）

        std::cout << "Iter " << iter << ", total error = " << total_error
                  << ", grad norm = " << grad_sum.norm() << "\n";
        if (grad_sum.norm() < epsilon) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
}

