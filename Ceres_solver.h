#pragma once
#include <ceres/ceres.h>
#include <Eigen/Core>

/********************
   定义模型: y = exp(a*x* + b*x + c)
 ********************/
struct ExpQuadraticModel  {
  static const int  num_paras = 3;
  template <typename T>
  static T Eval(const T* params, const T& x) {
    return ceres::exp(params[0] * x * x + params[1] * x + params[2]);
  }
};

template<typename TemplateModel>
class CurveFitErrorCostFunction{
public:
  CurveFitErrorCostFunction(const Eigen::Vector2d &point2D) : x_(point2D(0)), y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D){
    return (new ceres::AutoDiffCostFunction<
        CurveFitErrorCostFunction<TemplateModel>,
            1,
        TemplateModel::num_paras>(new CurveFitErrorCostFunction(point2D)));
  }

  template<typename T>
  bool operator()(const T* const paras,
                  T* residual) const {
    residual[0] = y_ - TemplateModel::Eval(paras, T(x_));
    return true;
  }
private:
  const double x_, y_;
};

