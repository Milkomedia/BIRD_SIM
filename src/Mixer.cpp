#include "Mixer.hpp"

#include "coeff/coeff.hpp"
#include "utils.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double RAD_TO_DEG = 57.2957795130823209;
constexpr double HALF_RHO = 0.5 * param::AIR_DENSITY;
constexpr double INV_KINEMATIC_VISCOSITY = 1.0 / param::AIR_KINEMATIC_VISCOSITY;
constexpr double MIN_VECTOR_NORM2 = 1.0e-20;
constexpr double MIN_FLOW_SPEED2 = 1.0e-12;

constexpr std::array<double, 6> INPUT_LOWER_BOUNDS = {0.0, 0.0, -param::MAX_FLAPPING_DIFFERENCE, 0.0, -param::MAX_PITCHING_DIFFERENCE, -param::MAX_SWEEP_BIAS};
constexpr std::array<double, 6> INPUT_UPPER_BOUNDS = {param::MAX_FREQ, param::MAX_FLAPPING_AMPLITUDE, param::MAX_FLAPPING_DIFFERENCE, param::MAX_PITCHING_AMPLITUDE, param::MAX_PITCHING_DIFFERENCE, param::MAX_SWEEP_BIAS};

inline void waveform(double r, double f, double& cos_r1, double& sin_r1, double& cos_r1_dot, double& sin_r1_dot, double& one_minus_cos_r2, double& one_minus_cos_r2_dot) noexcept {
  if (r < param::R1) {
    const double angle = PI * r / param::R1;
    cos_r1 = -std::cos(angle);
    sin_r1 = std::sin(angle);
    cos_r1_dot = f * (PI/param::R1)*sin_r1;
    sin_r1_dot = -f * (PI / param::R1) * cos_r1;
  }
  else {
    const double angle = PI * (r - param::R1) / (1.0 - param::R1);
    cos_r1 = std::cos(angle);
    sin_r1 = -std::sin(angle);
    cos_r1_dot = f * (PI / (1.0 - param::R1)) * sin_r1;
    sin_r1_dot = -f * (PI / (1.0 - param::R1)) * cos_r1;
  }

  if (r > param::R2) {
    const double angle = 2.0 * PI * (r - param::R2) / (1.0 - param::R2);
    one_minus_cos_r2 = 1.0 - std::cos(angle);
    one_minus_cos_r2_dot = f * (2.0 * PI / (1.0 - param::R2)) * std::sin(angle);
  }
  else {
    one_minus_cos_r2 = 0.0;
    one_minus_cos_r2_dot = 0.0;
  }
}

// Build bRsi = [be1_si, be2_si, be3_si] and the chord-axis time derivative.
inline void update_strip_rotation(Eigen::Matrix3d& bRsi, Eigen::Vector3d& be1_dot_si, const Eigen::Vector3d& n, const Eigen::Vector3d& n_dot, const Eigen::Vector3d& be3_si, const Eigen::Vector3d& be3_dot_si, const Eigen::Vector3d& chord_ref, const Eigen::Vector3d& chord_ref_dot) noexcept {
  Eigen::Vector3d be1_si = n.cross(be3_si);
  Eigen::Vector3d be1_dot = n_dot.cross(be3_si) + n.cross(be3_dot_si);
  double be1_norm2 = be1_si.squaredNorm();

  if (be1_norm2 <= MIN_VECTOR_NORM2) {
    const double e3_chord = be3_si.dot(chord_ref);
    const double e3_chord_dot = be3_dot_si.dot(chord_ref) + be3_si.dot(chord_ref_dot);
    be1_si = chord_ref - e3_chord*be3_si;
    be1_dot = chord_ref_dot - e3_chord_dot*be3_si - e3_chord*be3_dot_si;
    be1_norm2 = be1_si.squaredNorm();
  }

  if (be1_norm2 <= MIN_VECTOR_NORM2) {
    bRsi.col(0) = be3_si.unitOrthogonal();
    be1_dot_si.setZero();
  }
  else {
    const double be1_norm = std::sqrt(be1_norm2);
    bRsi.col(0) = be1_si / be1_norm;
    be1_dot_si = (be1_dot - bRsi.col(0)*bRsi.col(0).dot(be1_dot)) / be1_norm;
  }

  if (bRsi.col(0).dot(chord_ref) < 0.0) {
    bRsi.col(0) = -bRsi.col(0);
    be1_dot_si = -be1_dot_si;
  }

  bRsi.col(1) = be3_si.cross(bRsi.col(0));
  bRsi.col(2) = be3_si;
}

inline void update_projected_phi(double& phi, double& phi_dot, const Eigen::Vector3d& xi, const Eigen::Vector3d& xi_dot, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& b_omega_r0) noexcept {
  const Eigen::Vector3d be1_dot_r0 = b_omega_r0.cross(bRr0.col(0));
  const Eigen::Vector3d be3_dot_r0 = b_omega_r0.cross(bRr0.col(2));
  const double x = bRr0.col(0).dot(xi);
  const double z = bRr0.col(2).dot(xi);
  const double x_dot = be1_dot_r0.dot(xi) + bRr0.col(0).dot(xi_dot);
  const double z_dot = be3_dot_r0.dot(xi) + bRr0.col(2).dot(xi_dot);
  const double rho2 = x*x + z*z;

  if (rho2 <= MIN_VECTOR_NORM2) {phi = 0.0; phi_dot = 0.0; return;}

  phi = std::atan2(-z, x);
  phi_dot = (-x*z_dot + z*x_dot) / rho2;
}

} // namespace

Eigen::Matrix<double, 6, 1> Mixer::forward(const State& state, const Eigen::Matrix<double, 6, 1>& prev_input) noexcept {
  rebuild_kinematics(prev_input);

  Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::Vector3d RtVrel = state.R.transpose()*(state.vel_f-state.vel);

  accumulate_wrench(wrench, 0, HUMERUS_SAMPLE_COUNT, RtVrel, state.w, state.bpc, param::coeff::NACA_CD, param::coeff::NACA_CL, param::coeff::NACA_CM);
  accumulate_wrench(wrench, RADIUS_SAMPLE_BEGIN, MANUS_SAMPLE_BEGIN, RtVrel, state.w, state.bpc, param::coeff::S20_CD, param::coeff::S20_CL, param::coeff::S20_CM);
  accumulate_wrench(wrench, MANUS_SAMPLE_BEGIN, TOTAL_SAMPLES, RtVrel, state.w, state.bpc, param::coeff::S40_CD, param::coeff::S40_CL, param::coeff::S40_CM);
  wrench *= 1.0 / static_cast<double>(N_PHASE);

  return wrench;
}

void Mixer::update_B(const State& state, const Eigen::Matrix<double, 6, 1>& prev_input, Eigen::Matrix<double, 6, 6>& B, Eigen::Matrix<double, 6, 1>& nominal_wrench) noexcept {

  nominal_wrench = forward(state, prev_input);
  B.setZero();

  for (Eigen::Index input_idx=0; input_idx<6; ++input_idx) {
    const double lower = INPUT_LOWER_BOUNDS[static_cast<std::size_t>(input_idx)];
    const double upper = INPUT_UPPER_BOUNDS[static_cast<std::size_t>(input_idx)];
    const double step = param::MIXER_B_FD_FRACTION*(upper-lower);
    const double input = prev_input(input_idx);
    Eigen::Matrix<double, 6, 1> derivative = Eigen::Matrix<double, 6, 1>::Zero();

    if (input-step >= lower && input+step <= upper) {
      Eigen::Matrix<double, 6, 1> plus_input = prev_input;
      Eigen::Matrix<double, 6, 1> minus_input = prev_input;
      plus_input(input_idx) += step;
      minus_input(input_idx) -= step;
      const Eigen::Matrix<double, 6, 1> plus_wrench = forward(state, plus_input);
      const Eigen::Matrix<double, 6, 1> minus_wrench = forward(state, minus_input);
      derivative = (plus_wrench-minus_wrench)/(2.0*step);
    }
    else if (input+2.0*step <= upper) {
      // Second-order forward difference at a lower input bound.
      Eigen::Matrix<double, 6, 1> first_input = prev_input;
      Eigen::Matrix<double, 6, 1> second_input = prev_input;
      first_input(input_idx) += step;
      second_input(input_idx) += 2.0*step;
      const Eigen::Matrix<double, 6, 1> first_wrench = forward(state, first_input);
      const Eigen::Matrix<double, 6, 1> second_wrench = forward(state, second_input);
      derivative = (-3.0*nominal_wrench+4.0*first_wrench-second_wrench)/(2.0*step);
    }
    else if (input-2.0*step >= lower) {
      // Second-order backward difference at an upper input bound.
      Eigen::Matrix<double, 6, 1> first_input = prev_input;
      Eigen::Matrix<double, 6, 1> second_input = prev_input;
      first_input(input_idx) -= step;
      second_input(input_idx) -= 2.0*step;
      const Eigen::Matrix<double, 6, 1> first_wrench = forward(state, first_input);
      const Eigen::Matrix<double, 6, 1> second_wrench = forward(state, second_input);
      derivative = (3.0*nominal_wrench-4.0*first_wrench+second_wrench)/(2.0*step);
    }
    else {
      // Defensive fallback for a future input range narrower than two steps.
      const double plus = std::min(input+step, upper);
      const double minus = std::max(input-step, lower);
      if (plus > minus) {
        Eigen::Matrix<double, 6, 1> plus_input = prev_input;
        Eigen::Matrix<double, 6, 1> minus_input = prev_input;
        plus_input(input_idx) = plus;
        minus_input(input_idx) = minus;
        const Eigen::Matrix<double, 6, 1> plus_wrench = forward(state, plus_input);
        const Eigen::Matrix<double, 6, 1> minus_wrench = forward(state, minus_input);
        derivative = (plus_wrench-minus_wrench)/(plus-minus);
      }
    }

    B.col(input_idx) = derivative;
  }
}

void Mixer::accumulate_wrench(Eigen::Matrix<double, 6, 1>& wrench, const std::size_t begin, const std::size_t end, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& b_omega, const Eigen::Vector3d& bpc, const double (&CD)[176][14], const double (&CL)[176][14], const double (&CM)[176][14]) noexcept {
  for (std::size_t i=begin; i<end; ++i) {
    const double area = area_[i];
    if (area <= 0.0) {continue;}

    const Eigen::Matrix3d& bRsi = bRsi_[i];
    // Add the rigid-body velocity at the aerodynamic center to the cached
    // joint-driven velocity before evaluating the local relative flow.
    const Eigen::Vector3d bVrel_ac = RtVrel - b_omega.cross(bp_ac_[i]) - bv_ac_[i];
    const double vx = bRsi.col(0).dot(bVrel_ac);
    const double vz = bRsi.col(2).dot(bVrel_ac);
    const double U2 = vx*vx + vz*vz;
    if (U2 <= MIN_FLOW_SPEED2) {continue;}

    const double U = std::sqrt(U2);
    const double alpha = std::atan2(vz, vx);
    std::size_t alpha_idx;
    double k_alpha;
    param::coeff::get_idx_alpha(alpha_idx, k_alpha, alpha*RAD_TO_DEG);

    const double c = c_[i];
    const double Re = U*c*INV_KINEMATIC_VISCOSITY;
    std::size_t Re_idx;
    double k_Re;
    param::coeff::get_idx_Re(Re_idx, k_Re, Re);

    const double Cd = param::coeff::bilinear_interpolate(CD, alpha_idx, Re_idx, k_alpha, k_Re);
    const double Cl = param::coeff::bilinear_interpolate(CL, alpha_idx, Re_idx, k_alpha, k_Re);
    const double Cm = param::coeff::bilinear_interpolate(CM, alpha_idx, Re_idx, k_alpha, k_Re);

    const double k_f = HALF_RHO*U*area;
    const double Fx = k_f*(Cd*vx - Cl*vz);
    const double Fz = k_f*(Cd*vz + Cl*vx);
    const double My = k_f*U*c*Cm;
    const Eigen::Vector3d bF = Fx*bRsi.col(0) + Fz*bRsi.col(2);
    const Eigen::Vector3d bM = (bp_ac_[i] - bpc).cross(bF) + My*bRsi.col(1);

    wrench.head<3>() += bF;
    wrench.tail<3>() += bM;
  }
}

void Mixer::rebuild_kinematics(const Eigen::Matrix<double, 6, 1>& prev_input) noexcept {
  constexpr double MANUS_DX = param::D_P / static_cast<double>(param::NM-param::DECLINE_IDX_K);

  const double f = prev_input(0);
  const double Af_bar = prev_input(1);
  const double Af_delta = prev_input(2);
  const double Ap_bar = prev_input(3);
  const double Ap_delta = prev_input(4);
  const double sweep_bias = prev_input(5);

  std::size_t h_idx = 0;
  std::size_t r_idx = RADIUS_SAMPLE_BEGIN;
  std::size_t m_idx = MANUS_SAMPLE_BEGIN;

  for (std::size_t phase_idx=0; phase_idx<N_PHASE; ++phase_idx) {
    const double r = (static_cast<double>(phase_idx)+0.5) / static_cast<double>(N_PHASE);

    double cos_r1;
    double sin_r1;
    double cos_r1_dot;
    double sin_r1_dot;
    double one_minus_cos_r2;
    double one_minus_cos_r2_dot;
    waveform(r, f, cos_r1, sin_r1, cos_r1_dot, sin_r1_dot, one_minus_cos_r2, one_minus_cos_r2_dot);

    for (std::size_t wing=0; wing<2; ++wing) {
      const double side_sign = wing == 0 ? 1.0 : -1.0;
      const double span_sign = param::STRIP_SPAN_SIGN[wing];
      const double Af = Af_bar + side_sign*0.5*Af_delta;
      const double Ap = Ap_bar + side_sign*0.5*Ap_delta;
      const std::size_t j0 = wing*param::NUM_WING_JOINTS_PER_WING;

      // Joint angles [rad] and actual angular rates [rad/s]
      std::array<double, param::NUM_WING_JOINTS_PER_WING> theta{};
      std::array<double, param::NUM_WING_JOINTS_PER_WING> theta_dot{};

      const double flapping = param::FLAPPING_DELTA_0 + Af*cos_r1;
      const double pitching = param::PITCHING_DELTA_0 + 0.5*Ap*one_minus_cos_r2;
      const double sweep = sweep_bias + param::SWEEP_AMPLITUDE*sin_r1;
      const double folding = param::FOLDING_DELTA_0 + 0.5*param::FOLDING_AMPLITUDE*one_minus_cos_r2;

      theta[0] = param::INITIAL_DES_THETA[j0] + flapping;
      theta[1] = param::INITIAL_DES_THETA[j0+1] + pitching;
      theta[2] = param::INITIAL_DES_THETA[j0+2] + sweep;
      theta[3] = param::INITIAL_DES_THETA[j0+3] + folding;
      theta[4] = j5_angle(theta[2]);
      theta[5] = param::INITIAL_DES_THETA[j0+5] - 2.0*folding;

      theta_dot[0] = Af*cos_r1_dot;
      theta_dot[1] = 0.5*Ap*one_minus_cos_r2_dot;
      theta_dot[2] = param::SWEEP_AMPLITUDE*sin_r1_dot;
      theta_dot[3] = 0.5*param::FOLDING_AMPLITUDE*one_minus_cos_r2_dot;
      theta_dot[4] = j5_slope(theta[2])*theta_dot[2];
      theta_dot[5] = -2.0*theta_dot[3];

      std::array<Eigen::Matrix3d, param::NUM_WING_JOINTS_PER_WING> bRj{};
      std::array<Eigen::Vector3d, param::NUM_WING_JOINTS_PER_WING> bpj{};
      std::array<Eigen::Vector3d, param::NUM_WING_JOINTS_PER_WING> bvj{};
      std::array<Eigen::Vector3d, param::NUM_WING_JOINTS_PER_WING> b_omega_j{};

      // Forward kinematics
      Eigen::Matrix3d bRj_cur = Eigen::Matrix3d::Identity();
      Eigen::Vector3d bpj_cur = Eigen::Vector3d::Zero();
      Eigen::Vector3d bvj_cur = Eigen::Vector3d::Zero();
      Eigen::Vector3d b_omega_b_theta = Eigen::Vector3d::Zero();
      for (std::size_t j=0; j<param::NUM_WING_JOINTS_PER_WING; ++j) {
        const Eigen::Matrix4d& fixed_transform = param::JOINT_FIXED_TRANSFORM[j0+j];
        const Eigen::Vector3d drho = bRj_cur * fixed_transform.block<3, 1>(0, 3);
        bpj_cur += drho;
        bvj_cur += b_omega_b_theta.cross(drho);
        bRj_cur = bRj_cur * fixed_transform.block<3, 3>(0, 0);
        const Eigen::Vector3d bRje1 = bRj_cur.col(0);
        rotate_x(bRj_cur, theta[j]);
        b_omega_b_theta += theta_dot[j]*bRje1;

        bRj[j] = bRj_cur;
        bpj[j] = bpj_cur;
        bvj[j] = bvj_cur;
        b_omega_j[j] = b_omega_b_theta;
      }

      Eigen::Matrix3d bRh0;
      Eigen::Matrix3d bRr0;
      Eigen::Matrix3d bRm0;
      Eigen::Vector3d bph0;
      Eigen::Vector3d bpr0;
      Eigen::Vector3d bpm0;
      Eigen::Vector3d bvh0;
      Eigen::Vector3d bvr0;
      Eigen::Vector3d bvm0;

      {
        const Eigen::Matrix4d& jTh0 = param::J_T_S0[3*wing];
        const Eigen::Vector3d drho0 = bRj[2] * jTh0.block<3, 1>(0, 3);
        bRh0.noalias() = bRj[2] * jTh0.block<3, 3>(0, 0);
        bph0 = bpj[2] + drho0;
        bvh0 = bvj[2] + b_omega_j[2].cross(drho0);
      }
      {
        const Eigen::Matrix4d& jTr0 = param::J_T_S0[3*wing+1];
        const Eigen::Vector3d drho0 = bRj[3] * jTr0.block<3, 1>(0, 3);
        bRr0.noalias() = bRj[3] * jTr0.block<3, 3>(0, 0);
        bpr0 = bpj[3] + drho0;
        bvr0 = bvj[3] + b_omega_j[3].cross(drho0);
      }
      {
        const Eigen::Matrix4d& jTm0 = param::J_T_S0[3*wing+2];
        const Eigen::Vector3d drho0 = bRj[5] * jTm0.block<3, 1>(0, 3);
        bRm0.noalias() = bRj[5] * jTm0.block<3, 3>(0, 0);
        bpm0 = bpj[5] + drho0;
        bvm0 = bvj[5] + b_omega_j[5].cross(drho0);
      }

      const Eigen::Vector3d bRsec_y = bRj[1].col(0);
      const Eigen::Vector3d bRsec_y_dot = b_omega_j[1].cross(bRsec_y);
      const Eigen::Vector3d be3_h0 = bRh0.col(2);
      const Eigen::Vector3d be3_m0 = bRm0.col(2);
      const Eigen::Vector3d be3_dot_h0 = b_omega_j[2].cross(be3_h0);
      const Eigen::Vector3d be3_dot_m0 = b_omega_j[5].cross(be3_m0);
      const Eigen::Vector3d be1_h0 = bRh0.col(0);
      const Eigen::Vector3d be1_m0 = bRm0.col(0);
      const Eigen::Vector3d be1_dot_h0 = b_omega_j[2].cross(be1_h0);
      const Eigen::Vector3d be1_dot_m0 = b_omega_j[5].cross(be1_m0);

      Eigen::Matrix3d bRhi;
      Eigen::Matrix3d bRmi;
      Eigen::Vector3d be1_dot_hi;
      Eigen::Vector3d be1_dot_mi;
      update_strip_rotation(bRhi, be1_dot_hi, bRsec_y, bRsec_y_dot, be3_h0, be3_dot_h0, be1_h0, be1_dot_h0);
      update_strip_rotation(bRmi, be1_dot_mi, bRsec_y, bRsec_y_dot, be3_m0, be3_dot_m0, be1_m0, be1_dot_m0);

      // Each reduced sample represents adjacent MST strips; the final group
      // absorbs the remainder and preserves their total discrete width.
      for (std::size_t strip=0; strip<NH; ++strip) {
        const std::size_t i0 = strip*NSTRIP_REDUCTION;
        const std::size_t i1 = std::min(i0+NSTRIP_REDUCTION, param::NH);
        const std::size_t strip_count = i1-i0;
        const double i_mid = 0.5*static_cast<double>(i0+i1-1);
        const double y = param::DY_H*i_mid;
        const double dy = param::DY_H*static_cast<double>(strip_count);
        const double c = param::C_H0 + param::DL_H*i_mid;
        const Eigen::Vector3d drho = (span_sign*y)*bRh0.col(1);
        const Eigen::Vector3d bp_le = bph0 + drho;
        const Eigen::Vector3d bv_le = bvh0 + b_omega_j[2].cross(drho);
        const Eigen::Vector3d bp_ac = bp_le + 0.25*c*bRhi.col(0);
        const Eigen::Vector3d bv_ac = bv_le + 0.25*c*be1_dot_hi;
        const double dy_exposed = dy*std::abs(bRhi.col(1).dot(bRh0.col(1)));

        bp_ac_[h_idx] = bp_ac;
        bv_ac_[h_idx] = bv_ac;
        bRsi_[h_idx] = bRhi;
        c_[h_idx] = c;
        area_[h_idx] = c*dy_exposed;
        ++h_idx;
      }

      double phi_h;
      double phi_m;
      double phi_dot_h;
      double phi_dot_m;
      update_projected_phi(phi_h, phi_dot_h, bRhi.col(0), be1_dot_hi, bRr0, b_omega_j[3]);
      update_projected_phi(phi_m, phi_dot_m, bRmi.col(0), be1_dot_mi, bRr0, b_omega_j[3]);

      double delta_phi = phi_m-phi_h;
      if (delta_phi > PI) {delta_phi -= 2.0*PI;}
      else if (delta_phi < -PI) {delta_phi += 2.0*PI;}

      const Eigen::Vector3d be1_dot_r0 = b_omega_j[3].cross(bRr0.col(0));
      const Eigen::Vector3d be3_dot_r0 = b_omega_j[3].cross(bRr0.col(2));

      for (std::size_t strip=0; strip<NR; ++strip) {
        const std::size_t i0 = strip*NSTRIP_REDUCTION;
        const std::size_t i1 = std::min(i0+NSTRIP_REDUCTION, param::NR);
        const std::size_t strip_count = i1-i0;
        const double i_mid = 0.5*static_cast<double>(i0+i1-1);
        const double y = param::DY_R*i_mid;
        const double dy = param::DY_R*static_cast<double>(strip_count);
        const double lambda = y/param::L_R;
        const double phi = phi_h + lambda*delta_phi;
        const double phi_dot = phi_dot_h + lambda*(phi_dot_m-phi_dot_h);
        const double sin_phi = std::sin(phi);
        const double cos_phi = std::cos(phi);
        const Eigen::Vector3d u = cos_phi*bRr0.col(0) - sin_phi*bRr0.col(2);
        const Eigen::Vector3d be3_ri = sin_phi*bRr0.col(0) + cos_phi*bRr0.col(2);
        const Eigen::Vector3d be3_dot_ri = phi_dot*u + sin_phi*be1_dot_r0 + cos_phi*be3_dot_r0;
        const Eigen::Vector3d chord_ref = (1.0-lambda)*bRhi.col(0) + lambda*bRmi.col(0);
        const Eigen::Vector3d chord_ref_dot = (1.0-lambda)*be1_dot_hi + lambda*be1_dot_mi;
        Eigen::Matrix3d bRri;
        Eigen::Vector3d be1_dot_ri;
        update_strip_rotation(bRri, be1_dot_ri, bRsec_y, bRsec_y_dot, be3_ri, be3_dot_ri, chord_ref, chord_ref_dot);

        const double c = param::C_R0 + param::DL_R*i_mid;
        const Eigen::Vector3d drho = (span_sign*y)*bRr0.col(1);
        const Eigen::Vector3d bp_le = bpr0 + drho;
        const Eigen::Vector3d bv_le = bvr0 + b_omega_j[3].cross(drho);
        const Eigen::Vector3d bp_ac = bp_le + 0.25*c*bRri.col(0);
        const Eigen::Vector3d bv_ac = bv_le + 0.25*c*be1_dot_ri;
        const double dy_exposed = dy*std::abs(bRri.col(1).dot(bRr0.col(1)));

        bp_ac_[r_idx] = bp_ac;
        bv_ac_[r_idx] = bv_ac;
        bRsi_[r_idx] = bRri;
        c_[r_idx] = c;
        area_[r_idx] = c*dy_exposed;
        ++r_idx;
      }

      for (std::size_t strip=0; strip<NM; ++strip) {
        const std::size_t i0 = strip*NSTRIP_REDUCTION;
        const std::size_t i1 = std::min(i0+NSTRIP_REDUCTION, param::NM);
        const std::size_t strip_count = i1-i0;
        const double i_mid = 0.5*static_cast<double>(i0+i1-1);
        const double y = param::DY_M*i_mid;
        const double dy = param::DY_M*static_cast<double>(strip_count);
        const bool primary = i_mid >= static_cast<double>(param::DECLINE_IDX_K);
        const double c = primary ? param::C_MK + param::DL_M2*(i_mid-static_cast<double>(param::DECLINE_IDX_K)) : param::C_M0 + param::DL_M1*i_mid;
        const double primary_steps = primary ? i_mid-static_cast<double>(param::DECLINE_IDX_K)+1.0 : 0.0;
        const Eigen::Vector3d drho = (span_sign*y)*bRm0.col(1) + (MANUS_DX*primary_steps)*bRm0.col(0);
        const Eigen::Vector3d bp_le = bpm0 + drho;
        const Eigen::Vector3d bv_le = bvm0 + b_omega_j[5].cross(drho);
        const Eigen::Vector3d bp_ac = bp_le + 0.25*c*bRmi.col(0);
        const Eigen::Vector3d bv_ac = bv_le + 0.25*c*be1_dot_mi;
        const double dy_exposed = dy*std::abs(bRmi.col(1).dot(bRm0.col(1)));

        bp_ac_[m_idx] = bp_ac;
        bv_ac_[m_idx] = bv_ac;
        bRsi_[m_idx] = bRmi;
        c_[m_idx] = c;
        area_[m_idx] = c*dy_exposed;
        ++m_idx;
      }
    }
  }
}
