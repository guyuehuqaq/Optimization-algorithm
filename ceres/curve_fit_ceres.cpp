#include <iostream>
#include <opencv2/opencv.hpp>
#include "curve_fit_ceres.h"

// 生成数据
void GenerateData(const std::vector<double>& paras,
                  const int& data_num,
                  const double& w_sigma,
                  std::vector<Eigen::Vector2d>& data){
  cv::RNG rng;

  for (int i = 0; i < data_num; i++){
    Eigen::Vector2d point2d = Eigen::Vector2d::Zero();
    point2d.x() = i / double(data_num);
    point2d.y() = ExpQuadraticModel::Eval(paras.data(), point2d.x()) + rng.gaussian(w_sigma * w_sigma);
    data.emplace_back(point2d);
  }
}


int main(int argc, char **argv){

  std::vector<double> paras = {1.0, 2.0, 1.0};
  int data_num = 100;
  double w_sigma = 0.1;
  std::vector<Eigen::Vector2d> data;
  GenerateData(paras, data_num, w_sigma, data);

  ceres::Solver::Summary summary;
  ceres::Problem problem;
  ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

  problem.AddParameterBlock(paras.data(), 3);

  for (const auto& point2D : data){
    ceres::CostFunction* cost_function = nullptr;
    cost_function = CurveFitErrorCostFunction<ExpQuadraticModel>::Create(point2D);
    if (cost_function == nullptr) {
      return -1;
    }
    problem.AddResidualBlock(cost_function, loss_function, paras.data());
  }

  // -----------------------------
  // 设置求解器
  ceres::Solver::Options solver_options;
  solver_options.minimizer_progress_to_stdout = true;  // 打印优化过程
  solver_options.linear_solver_type = ceres::DENSE_QR; // 小型问题用 DENSE_QR
  solver_options.max_num_iterations = 100;

  // 求解
  ceres::Solve(solver_options, &problem, &summary);

  // 打印结果
  std::cout << "------ Optimization Summary ------\n";
  std::cout << summary.FullReport() << "\n";
  std::cout << "Optimized parameters: a=" << paras[0]
            << ", b=" << paras[1]
            << ", c=" << paras[2] << "\n";

  return 0;


}