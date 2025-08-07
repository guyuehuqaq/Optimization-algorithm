#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <functional>

using namespace Eigen;

// LM算法核心实现
// 参数说明：
//  - x_init: 初始参数向量
//  - residualFunc: 计算残差向量的函数，输入参数x，输出残差向量r
//  - jacobianFunc: 计算雅可比矩阵的函数，输入参数x，输出雅可比矩阵J
//  - maxIterations: 最大迭代次数
//  - tol: 收敛阈值
VectorXd LevenbergMarquardt(
        const VectorXd& x_init,
        std::function<VectorXd(const VectorXd&)> residualFunc,
        std::function<MatrixXd(const VectorXd&)> jacobianFunc,
        int maxIterations = 100,
        double tol = 1e-6)
{
    VectorXd x = x_init;
    double lambda = 1e-3;         // 初始阻尼参数
    double v = 2.0;               // 阻尼调整因子

    VectorXd r = residualFunc(x);
    double prevCost = r.squaredNorm();

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        MatrixXd J = jacobianFunc(x);
        MatrixXd A = J.transpose() * J;
        VectorXd g = J.transpose() * r;

        // 判断梯度范数是否足够小，提前收敛
        if (g.norm() < tol)
        {
            std::cout << "Gradient norm below tolerance, stop at iteration " << iter << std::endl;
            break;
        }

        bool foundBetter = false;
        VectorXd delta_x;

        while (!foundBetter)
        {
            // 构造带阻尼的线性方程组 (A + lambda * I) delta_x = -g
            MatrixXd A_lm = A + lambda * MatrixXd::Identity(x.size(), x.size());

            // 求解增量
            delta_x = A_lm.ldlt().solve(-g);

            if (delta_x.norm() < tol)
            {
                std::cout << "Step size below tolerance, stop at iteration " << iter << std::endl;
                return x;
            }

            VectorXd x_new = x + delta_x;
            VectorXd r_new = residualFunc(x_new);
            double newCost = r_new.squaredNorm();

            if (newCost < prevCost)
            {
                // 接受更新
                x = x_new;
                r = r_new;
                prevCost = newCost;

                // 减小阻尼参数
                lambda /= v;
                foundBetter = true;
            }
            else
            {
                // 拒绝更新，增大阻尼参数
                lambda *= v;

                // 阻尼过大则停止
                if (lambda > 1e12)
                {
                    std::cout << "Lambda too large, stop at iteration " << iter << std::endl;
                    return x;
                }
            }
        }

        std::cout << "Iteration " << iter << ", cost = " << prevCost << std::endl;

        if (prevCost < tol)
        {
            std::cout << "Cost below tolerance, optimization finished." << std::endl;
            break;
        }
    }

    return x;
}

// ---------- 示例使用 ----------
// 目标：拟合 y = x^2
// 残差函数 r(x) = x^2 - y_obs

VectorXd residualExample(const VectorXd& x)
{
    // 假设观测值 y_obs = 4，想找到 x 使 x^2 ≈ 4
    VectorXd r(1);
    r(0) = x(0) * x(0) - 4.0;
    return r;
}

MatrixXd jacobianExample(const VectorXd& x)
{
    MatrixXd J(1, 1);
    J(0, 0) = 2 * x(0);
    return J;
}

int main()
{
    VectorXd x_init(1);
    x_init(0) = 1.0; // 初始猜测

    VectorXd x_opt = LevenbergMarquardt(x_init, residualExample, jacobianExample);

    std::cout << "Optimized x: " << x_opt.transpose() << std::endl;

    return 0;
}
