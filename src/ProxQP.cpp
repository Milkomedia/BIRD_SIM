#include "ProxQP.hpp"
#include "params.hpp"

#include <stdexcept>
#include <chrono>
#include <cmath>

ProxQP::ProxQP() : qp_(6, 0, 0, true, proxsuite::proxqp::DenseBackend::PrimalDualLDLT) {
  u_min_ << param::MIN_FREQ, param::MIN_FLAPPING_AMPLITUDE, param::MIN_FLAPPING_DIFFERENCE, param::MIN_PITCHING_AMPLITUDE, param::MIN_PITCHING_DIFFERENCE, param::MIN_SWEEP_BIAS;
  u_max_ << param::MAX_FREQ, param::MAX_FLAPPING_AMPLITUDE, param::MAX_FLAPPING_DIFFERENCE, param::MAX_PITCHING_AMPLITUDE, param::MAX_PITCHING_DIFFERENCE, param::MAX_SWEEP_BIAS;

  constexpr double wrench_scale[6] = {1.0, 1.0, 1.0, 0.1, 0.1, 0.1}; // [N, N, N, N.m, N.m, N.m]
  for (std::size_t i=0; i<6; ++i) {
    const Eigen::Index index = static_cast<Eigen::Index>(i);
    const double input_range = u_max_(index)-u_min_(index);
    if (!std::isfinite(param::QP_WRENCH_WEIGHT[i]) || param::QP_WRENCH_WEIGHT[i] < 0.0 || !std::isfinite(param::QP_DELTA_WEIGHT[i]) || param::QP_DELTA_WEIGHT[i] < 0.0 || !std::isfinite(param::QP_DEFAULT_WEIGHT[i]) || param::QP_DEFAULT_WEIGHT[i] < 0.0) {throw std::invalid_argument("Invalid QP weight.");}
    if (!std::isfinite(param::QP_DEFAULT_INPUT[i]) || param::QP_DEFAULT_INPUT[i] < u_min_(index) || param::QP_DEFAULT_INPUT[i] > u_max_(index)) {throw std::invalid_argument("Invalid QP default input.");}
    u_default_(index) = param::QP_DEFAULT_INPUT[i];
    default_weight_(index) = param::QP_DEFAULT_WEIGHT[i] / (input_range*input_range);
    wrench_weight_(index) = param::QP_WRENCH_WEIGHT[i] / (wrench_scale[i]*wrench_scale[i]);
    input_weight_(index) = param::QP_DELTA_WEIGHT[i] / (input_range*input_range);
  }
  input_weight_plus_default_weight_ = input_weight_ + default_weight_;

  qp_.settings.eps_abs = param::QP_EPS_ABS;
  qp_.settings.max_iter = param::QP_MAX_ITER;
  qp_.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
}

bool ProxQP::solve(const Eigen::Matrix<double, 6, 6>& B, const Eigen::Matrix<double, 6, 1>& wrench_error, Eigen::Matrix<double, 6, 1>& u) {
  const std::chrono::steady_clock::time_point solve_begin = std::chrono::steady_clock::now();

  const auto finish = [&](const bool success) {
    solve_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-solve_begin).count();
    if (!success) {solved_ = 0;}
    else if (solved_ < 0) {solved_ = 1;}
    return success;
  };
  
  if (!B.allFinite() || !wrench_error.allFinite() || !u.allFinite()) {return finish(false);}

  const Eigen::Matrix<double, 6, 6> weighted_B = wrench_weight_.asDiagonal() * B;
  Eigen::Matrix<double, 6, 6> H = B.transpose() * weighted_B;
  H.diagonal() += input_weight_plus_default_weight_;
  H.triangularView<Eigen::StrictlyUpper>() = H.transpose();

  const Eigen::Matrix<double, 6, 1> g = -B.transpose() * wrench_weight_.cwiseProduct(wrench_error) + default_weight_.cwiseProduct(u - u_default_);
  const Eigen::Matrix<double, 6, 1> lower = u_min_ - u;
  const Eigen::Matrix<double, 6, 1> upper = u_max_ - u;
  if (!H.allFinite() || !g.allFinite() || !lower.allFinite() || !upper.allFinite()) {return finish(false);}

  if (!initialized_) {
    qp_.init(H, g, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, lower, upper);
    initialized_ = true;
  }
  else {qp_.update(H, g, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, lower, upper, false);}

  qp_.solve();

  if (qp_.results.info.status != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED || !qp_.results.x.allFinite()) {
    qp_.cleanup();
    initialized_ = false;
    return finish(false);
  }
    
  u = (u + qp_.results.x).cwiseMax(u_min_).cwiseMin(u_max_);
  return finish(true);
}