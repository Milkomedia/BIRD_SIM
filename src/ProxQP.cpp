#include "ProxQP.hpp"
#include "params.hpp"
#include <stdexcept>

ProxQP::ProxQP() : qp_(6, 0, 0, true, proxsuite::proxqp::DenseBackend::PrimalDualLDLT) {
    u_min_ << param::MIN_FREQ, param::MIN_FLAPPING_AMPLITUDE, param::MIN_FLAPPING_DIFFERENCE, param::MIN_PITCHING_AMPLITUDE, param::MIN_PITCHING_DIFFERENCE, param::MIN_SWEEP_BIAS;
    u_max_ << param::MAX_FREQ, param::MAX_FLAPPING_AMPLITUDE, param::MAX_FLAPPING_DIFFERENCE, param::MAX_PITCHING_AMPLITUDE, param::MAX_PITCHING_DIFFERENCE, param::MAX_SWEEP_BIAS;

    const double force_weight = 1.0 / (param::QP_FORCE_SCALE * param::QP_FORCE_SCALE);
    const double moment_weight = 1.0 / (param::QP_MOMENT_SCALE * param::QP_MOMENT_SCALE);
    wrench_weight_ << force_weight, force_weight, force_weight, moment_weight, moment_weight, moment_weight;
    input_weight_.array() = param::QP_DELTA_WEIGHT / (u_max_ - u_min_).array().square();

    qp_.settings.eps_abs = param::QP_EPS_ABS;
    qp_.settings.max_iter = param::QP_MAX_ITER;
    qp_.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
}

bool ProxQP::solve(const Eigen::Matrix<double, 6, 6>& B, const Eigen::Matrix<double, 6, 1>& wrench_error, Eigen::Matrix<double, 6, 1>& u) {
    if (!B.allFinite() || !wrench_error.allFinite() || !u.allFinite()) {return false;}

    const Eigen::Matrix<double, 6, 6> weighted_B = wrench_weight_.asDiagonal() * B;
    Eigen::Matrix<double, 6, 6> H = B.transpose() * weighted_B;
    H.diagonal() += input_weight_;
    H.triangularView<Eigen::StrictlyUpper>() = H.transpose();

    const Eigen::Matrix<double, 6, 1> g = -B.transpose() * wrench_weight_.cwiseProduct(wrench_error);
    const Eigen::Matrix<double, 6, 1> lower = u_min_ - u;
    const Eigen::Matrix<double, 6, 1> upper = u_max_ - u;
    if (!H.allFinite() || !g.allFinite() || !lower.allFinite() || !upper.allFinite()) {return false;}

    if (!initialized_) {
        qp_.init(H, g, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, lower, upper);
        initialized_ = true;
    }
    else {qp_.update(H, g, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt, lower, upper, false);}

    qp_.solve();

    if (qp_.results.info.status != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED || !qp_.results.x.allFinite()) {
        qp_.cleanup();
        initialized_ = false;
        return false;
    }
    
    u = (u + qp_.results.x).cwiseMax(u_min_).cwiseMin(u_max_);
    return true;
}