#pragma once
#include <Eigen/Core>
#include <proxsuite/proxqp/dense/dense.hpp>

class ProxQP {
public:
  ProxQP();
  bool solve(const Eigen::Matrix<double, 6, 6>& B, const Eigen::Matrix<double, 6, 1>& wrench_error, Eigen::Matrix<double, 6, 1>& u);
    
  void reset_telemetry() noexcept {
    solve_us_ = 0.0;
    solved_ = -1;
  }
  double solve_us() const noexcept {return solve_us_;}
  int solved() const noexcept {return solved_;}

private:
  proxsuite::proxqp::dense::QP<double> qp_;
  Eigen::Matrix<double, 6, 1> wrench_weight_, input_weight_, default_weight_, input_weight_plus_default_weight_, u_default_, u_min_, u_max_;
  bool initialized_ = false;
  double solve_us_ = 0.0;
  int solved_ = -1; // -1: idle, 0: any failure, 1: succeeded (since the last reset)
};