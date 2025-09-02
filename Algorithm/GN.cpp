#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;

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
        /***************************
        求解线性方程 H * dx = b，得到更新量
        Cholesky 分解求解 Δx
        llt()：对称正定矩阵的 Cholesky 分解（下三角）
        ldlt()：对称矩阵的 LDLᵗ 分解（可用于半正定）
         ***************************/
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

/*************
   生成曲线数据, 设曲线为 y = exp(a*x*x + b*x + c)
 *************/
void GenerateCurveData(std::vector<double>& paras,
                       double& w_sigma,
                       int& data_num,
                       std::vector<double>& x_data,
                       std::vector<double>& y_data){
  cv::RNG rng;

  std::cout << paras[0] << paras[1] << paras[2] << std::endl;

  for (int i = 0; i < data_num; i++){
    double x = i / 100.0;
    x_data.push_back(x);
    double y = exp(paras[0]*x*x + paras[1]*x + paras[2]) + rng.gaussian(w_sigma * w_sigma);
    y_data.push_back(y);
  }
}

/*************
   高斯牛顿优化拟合曲线
 *************/
void OptimizeCurveUsingGN(){
  std::vector<double> paras = {1.0, 2.0, 1.0};   // 曲线真实参数
  double ae = 2.0, be = -1.0, ce = 5.0;          // 曲线初始估计参数
  int data_num = 100;                            // 数据量
  double w_sigma = 1.0;                          // 高斯误差
  double inv_sigma = 1.0 / w_sigma;

  std::vector<double> x_data, y_data;
  GenerateCurveData(paras, w_sigma, data_num, x_data, y_data);

//  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
//  cv::RNG rng;                                 // OpenCV随机数产生器
//  vector<double> x_data, y_data;      // 数据
//  for (int i = 0; i < data_num; i++) {
//    double x = i / 100.0;
//    x_data.push_back(x);
//    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
//  }

  double cost = 0, lastCost = 0;

  // 高斯牛顿优化
  int max_iterations = 100;                // 迭代次数
  for (int iter = 0; iter < max_iterations; iter++){
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();  // 近似Hessian矩阵
    Eigen::Vector3d b = Eigen::Vector3d::Zero();  // 梯度项

    cost = 0;
    for (int i = 0; i < data_num; ++i){
      double xi = x_data[i], yi = y_data[i];
      double error = yi - exp(ae*xi*xi + be*xi + ce);
      Eigen::Vector3d  J;  // 雅可比矩阵
      J[0] = -xi*xi*exp(ae*xi*xi + be*xi + ce);   // d_error/d_a
      J[1] = -xi*exp(ae*xi*xi + be*xi + ce);      // d_error/d_b
      J[2] = -exp(ae * xi * xi + be * xi + ce);   // d_error/dc
      H +=  inv_sigma * inv_sigma * J * J.transpose();    // inv_sigma * inv_sigma * J.transpose() * J;
      b += -inv_sigma * inv_sigma * error * J;
      cost += error * error;
    }
    // 求解H * d_x = b, 更新x_k+1
    Eigen::Vector3d d_x  = H.ldlt().solve(b);
    if (isnan(d_x[0])) {
      std::cout << "result is nan!" << std::endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      std::cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << std::endl;
      break;
    }
    ae += d_x[0];
    be += d_x[1];
    ce += d_x[2];
    lastCost = cost;
    std::cout << "total cost: " << cost << ", \t\tupdate: " << d_x.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << std::endl;
  }

  std::cout << "estimated abc = " << ae << ", " << be << ", " << ce << std::endl;
}


int main(int argc, char **argv){

  OptimizeCurveUsingGN();

  return 0;
}