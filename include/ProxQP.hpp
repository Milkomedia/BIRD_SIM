#pragma once
#include <Eigen/Core>
#include <proxsuite/proxqp/dense/dense.hpp>

class ProxQP {
public:
    ProxQP();
    bool solve(const Eigen::Matrix<double, 6, 6>& B, const Eigen::Matrix<double, 6, 1>& wrench_error, Eigen::Matrix<double, 6, 1>& u);

private:
    proxsuite::proxqp::dense::QP<double> qp_;
    Eigen::Matrix<double, 6, 1> wrench_weight_, input_weight_, u_min_, u_max_;
    bool initialized_ = false;
};