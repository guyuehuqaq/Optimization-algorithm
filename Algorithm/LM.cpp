#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>



/*************
   生成曲线数据, 设曲线为 y = exp(a*x*x + b*x + c)
 *************/
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

void GenerateCurveData(const std::vector<double>& paras,
                       double w_sigma,
                       int data_num,
                       std::vector<double>& x_data,
                       std::vector<double>& y_data){
  cv::RNG rng;

  for (int i = 0; i < data_num; i++){
    double x = i / 100.0;
    x_data.push_back(x);
    double y = std::exp(paras[0]*x*x + paras[1]*x + paras[2]) + rng.gaussian(w_sigma * w_sigma);
    y_data.push_back(y);
  }
}

void OptimizeCurveUsingLM() {
  std::vector<double> paras = {1.0, 2.0, 1.0};   // 曲线真实参数
  double ae = 2.0, be = -1.0, ce = 5;          // 初始估计
  int data_num = 100;
  double w_sigma = 1.0;

  std::vector<double> x_data, y_data;
  GenerateCurveData(paras, w_sigma, data_num, x_data, y_data);

  double cost = 0;
  int max_iterations = 100;
  double lambda = 1.0;   // 阻尼因子
  double ni = 2.0;

  for (int iter = 0; iter < max_iterations; ++iter){
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    cost = 0;

    for (int i = 0; i < data_num; ++i){
      double xi = x_data[i], yi = y_data[i];
      if (yi <= 0) yi = 1e-6; // 避免 log(0)
      double error = std::log(yi) - (ae*xi*xi + be*xi + ce);
      Eigen::Vector3d J;
      J[0] = -xi*xi; // d_error/d_a
      J[1] = -xi;    // d_error/d_b
      J[2] = -1;     // d_error/d_c

      H += J * J.transpose();
      b += -error * J;
      cost += error * error;
    }

    Eigen::Matrix3d H_lm = H + lambda * Eigen::Matrix3d::Identity();
    Eigen::Vector3d d_x = H_lm.ldlt().solve(b);

    if (!d_x.allFinite()) {
      std::cout << "d_x contains NaN/Inf, increasing lambda..." << std::endl;
      lambda *= ni;
      ni *= 2.0;
      continue;
    }

    double new_ae = ae + d_x[0];
    double new_be = be + d_x[1];
    double new_ce = ce + d_x[2];

    // 新 cost
    double new_cost = 0.0;
    for (int i = 0; i < data_num; ++i){
      double xi = x_data[i], yi = y_data[i];
      if (yi <= 0) yi = 1e-6;
      double error = std::log(yi) - (new_ae*xi*xi + new_be*xi + new_ce);
      new_cost += error * error;
    }

    double F_actual = cost - new_cost;
    double F_predict = -(b.transpose()*d_x + 0.5* d_x.transpose()* H *d_x).value();
    double rho = F_actual / F_predict;

    if (rho > 0){
      ae = new_ae;
      be = new_be;
      ce = new_ce;
      cost = new_cost;
      lambda *= std::max(1.0/3.0, 1.0 - pow(2*rho - 1, 3));
      ni = 2.0;
    }else{
      lambda *= ni;
      ni *= 2.0;
    }

    std::cout << "Iter " << iter << ", cost: " << cost << ", a,b,c: "
              << ae << ", " << be << ", " << ce << std::endl;

    if (d_x.norm() < 1e-6) break; // 收敛判断
  }

  std::cout << "===========" << std::endl;
  std::cout << "Final cost: " << cost << std::endl;
  std::cout << "Estimated a,b,c: " << ae << ", " << be << ", " << ce << std::endl;
}

int main() {
  OptimizeCurveUsingLM();
  return 0;
}
