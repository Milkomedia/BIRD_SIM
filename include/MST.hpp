#pragma once

#include "params.hpp"

#include <Eigen/Core>
#include <array>
#include <cstddef>

struct State;

class MST {
public:
  static constexpr std::size_t NUM_STRIPS = 2*(param::NH + param::NR + param::NM);

  template <std::size_t N>
  struct StripRotation {
    std::array<Eigen::Matrix3d, N> bRri{};
    std::array<double, N> sin_psi{};
    std::array<double, N> cos_psi{};
    std::array<double, N> sin_phi{};
    std::array<double, N> cos_phi{};
    std::array<double, N> psi_dot{};
    std::array<double, N> psi_ddot{};
    std::array<double, N> phi_dot{};
    std::array<double, N> phi_ddot{};
    bool initialized = false;

    StripRotation() {reset();}

   void reset() {
      for (Eigen::Matrix3d& R : bRri) {R.setIdentity();}
      sin_psi.fill(0.0);
      cos_psi.fill(1.0);
      sin_phi.fill(0.0);
      cos_phi.fill(1.0);
      psi_dot.fill(0.0);
      psi_ddot.fill(0.0);
      phi_dot.fill(0.0);
      phi_ddot.fill(0.0);
      initialized = false;
    }
  };

  struct StripState {
    std::array<Eigen::Vector3d, 2*param::NH> p_h{};
    std::array<Eigen::Vector3d, 2*param::NR> p_r{};
    std::array<Eigen::Vector3d, 2*param::NM> p_m{};
    std::array<Eigen::Vector3d, 2*param::NH> v_h{};
    std::array<Eigen::Vector3d, 2*param::NR> v_r{};
    std::array<Eigen::Vector3d, 2*param::NM> v_m{};
    std::array<Eigen::Vector3d, 2*param::NH> a_h{};
    std::array<Eigen::Vector3d, 2*param::NR> a_r{};
    std::array<Eigen::Vector3d, 2*param::NM> a_m{};
    std::array<Eigen::Vector3d, 2> w_h{};
    std::array<Eigen::Vector3d, 2*param::NR> w_r{};
    std::array<Eigen::Vector3d, 2> w_m{};
    std::array<Eigen::Vector3d, 2> wdot_h{};
    std::array<Eigen::Vector3d, 2*param::NR> wdot_r{};
    std::array<Eigen::Vector3d, 2> wdot_m{};
    std::array<StripRotation<1>, 2> humerus_rotation{};
    std::array<StripRotation<param::NR>, 2> radius_rotation{};
    std::array<StripRotation<1>, 2> manus_rotation{};

    StripState() {reset();}

    void reset() {
      const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
      p_h.fill(zero); p_r.fill(zero); p_m.fill(zero);
      v_h.fill(zero); v_r.fill(zero); v_m.fill(zero);
      a_h.fill(zero); a_r.fill(zero); a_m.fill(zero);
      w_h.fill(zero); w_r.fill(zero); w_m.fill(zero);
      wdot_h.fill(zero); wdot_r.fill(zero); wdot_m.fill(zero);
      for (auto& rotation : humerus_rotation) {rotation.reset();}
      for (auto& rotation : radius_rotation) {rotation.reset();}
      for (auto& rotation : manus_rotation) {rotation.reset();}
    }
  };

  struct AeroTelemetry {
    std::array<double, NUM_STRIPS> alpha{};
    std::array<double, NUM_STRIPS> alpha_dot{};
    std::array<double, NUM_STRIPS> speed{};
    std::array<double, NUM_STRIPS> Re{};
    std::array<double, NUM_STRIPS> Cd{};
    std::array<double, NUM_STRIPS> Cl_lut{};
    std::array<double, NUM_STRIPS> Cl_dynamic{};
    std::array<double, NUM_STRIPS> Cm{};
    std::array<double, NUM_STRIPS> X_eq{};
    std::array<double, NUM_STRIPS> X{};
    std::array<double, NUM_STRIPS> X_target{};
    std::array<double, NUM_STRIPS> tau1{};
    std::array<double, NUM_STRIPS> tau2{};
    std::array<double, NUM_STRIPS> stall_active{};
    std::array<Eigen::Vector3d, NUM_STRIPS> lut_force{};
    std::array<Eigen::Vector3d, NUM_STRIPS> dynamic_force{};
    std::array<Eigen::Vector3d, NUM_STRIPS> added_bias_force{};
    std::array<Eigen::Vector3d, NUM_STRIPS> added_full_force{};
    std::array<Eigen::Vector3d, NUM_STRIPS> lut_moment{};
    std::array<Eigen::Vector3d, NUM_STRIPS> added_bias_moment{};
    std::array<Eigen::Vector3d, NUM_STRIPS> added_full_moment{};

    void reset() {
      const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
      alpha.fill(0.0); alpha_dot.fill(0.0); speed.fill(0.0); Re.fill(0.0);
      Cd.fill(0.0); Cl_lut.fill(0.0); Cl_dynamic.fill(0.0); Cm.fill(0.0);
      X_eq.fill(1.0); X.fill(1.0); X_target.fill(1.0);
      tau1.fill(0.0); tau2.fill(0.0); stall_active.fill(0.0);
      lut_force.fill(zero); dynamic_force.fill(zero);
      added_bias_force.fill(zero); added_full_force.fill(zero);
      lut_moment.fill(zero);
      added_bias_moment.fill(zero); added_full_moment.fill(zero);
    }
  };

  MST();
  void reset();
  void update_dynamics(const State& s, const bool update_telemetry = false) {update(s, true, true, update_telemetry); update_body_elipsoid(s);}
  void update_visualization(const State& s) {update(s, false, false, false); update_full_added_mass_telemetry();}

  const std::array<Eigen::Vector3d, 7>& positions() const noexcept {return aero_pos_;}
  const std::array<Eigen::Vector3d, 7>& forces() const noexcept {return aero_force_;}
  const std::array<Eigen::Vector3d, 7>& torques() const noexcept {return aero_torque_;}
  const std::array<Eigen::Vector3d, 6>& added_mass_positions() const noexcept {return added_mass_pos_;}
  const std::array<Eigen::Matrix<double, 6, 6>, 6>& added_mass_matrices() const noexcept {return added_mass_matrix_;}
  const StripState& copy_strip_state() const noexcept {return strip_state_;}
  const AeroTelemetry& aero_telemetry() const noexcept {return aero_telemetry_;}

private:
  struct DynamicStallState {
    std::array<double, 2> X{1.0, 1.0};
    std::array<double, 2> X_eq{1.0, 1.0};
    std::array<double, 2> X_target{1.0, 1.0};
    std::array<double, 2> q_ss{};
    std::array<double, 2> D2{};
    double alpha = 0.0;
    bool state_initialized = false;
    bool alpha_initialized = false;
    std::array<bool, 2> active{};
  };

  StripState strip_state_{};
  AeroTelemetry aero_telemetry_{};
  std::array<DynamicStallState, 2*(param::NH+param::NR+param::NM)> dynamic_stall_state_{};
  // [0:5] wing segments (RH, RR, RM, LH, LR, LM), [6] body ellipsoid
  std::array<Eigen::Vector3d, 7> aero_pos_{};    // [m], body FRD
  std::array<Eigen::Vector3d, 7> aero_force_{};  // [N], body FRD
  std::array<Eigen::Vector3d, 7> aero_torque_{}; // [N.m], body FRD
  std::array<Eigen::Vector3d, 6> added_mass_pos_{};                    // Reference points, body FRD [m]
  std::array<Eigen::Matrix<double, 6, 6>, 6> added_mass_matrix_{};     // At reference points, body FRD

  void update(const State& s, bool acceleration_bias_only, bool update_loads, bool update_telemetry);
  void update_body_elipsoid(const State& s);
  void update_full_added_mass_telemetry();

  void update_atan2_dot_ddot(double& angle_dot, double& angle_ddot, const double y, const double x, const double y_dot, const double x_dot, const double y_ddot, const double x_ddot);
  void update_relative_vector_dot_ddot(Eigen::Vector3d& x_dot_0, Eigen::Vector3d& x_ddot_0, const Eigen::Vector3d& x_0, const Eigen::Matrix3d& bR0, const Eigen::Vector3d& b_omega_0, const Eigen::Vector3d& b_omega_dot_0, const Eigen::Vector3d& b_omega_x, const Eigen::Vector3d& b_omega_dot_x);
  void update_Rz_psi(Eigen::Matrix3d& bRsi, double& sin_psi, double& cos_psi, double& psi_dot, double& psi_ddot, const Eigen::Vector3d& n, const Eigen::Matrix3d& bRs0, const Eigen::Vector3d& b_omega_n, const Eigen::Vector3d& b_omega_dot_n, const Eigen::Vector3d& b_omega_s0, const Eigen::Vector3d& b_omega_dot_s0);
  void update_Ryphi_Rzpsi(Eigen::Matrix3d& bRri, double& sin_psi, double& cos_psi, double& sin_phi, double& cos_phi, double& psi_dot, double& psi_ddot, double& phi_dot, double& phi_ddot, const Eigen::Vector3d& xi, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& b_omega_xi, const Eigen::Vector3d& b_omega_dot_xi, const Eigen::Vector3d& b_omega_r0, const Eigen::Vector3d& b_omega_dot_r0, const bool initialized);
  void update_radius_strip_rotation(StripRotation<param::NR>& rotation, const Eigen::Vector3d& n, const Eigen::Matrix3d& bRhi, const Eigen::Matrix3d& bRmi, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& b_omega_n, const Eigen::Vector3d& b_omega_dot_n, const Eigen::Vector3d& b_omega_hi, const Eigen::Vector3d& b_omega_dot_hi, const Eigen::Vector3d& b_omega_mi, const Eigen::Vector3d& b_omega_dot_mi, const Eigen::Vector3d& b_omega_r0, const Eigen::Vector3d& b_omega_dot_r0);

  void update_humerus_stream_p_v_a(const std::size_t idx0, const Eigen::Matrix3d& bRh0, const Eigen::Vector3d& bph0, const Eigen::Matrix3d& bRhi, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvh0, const Eigen::Vector3d& bah0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy);
  void update_radius_stream_p_v_a(const std::size_t idx0, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& bpr0, const std::array<Eigen::Matrix3d, param::NR>& bRri, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvr0, const Eigen::Vector3d& bar0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy);
  void update_manus_stream_p_v_a(const std::size_t idx0, const Eigen::Matrix3d& bRm0, const Eigen::Vector3d& bpm0, const Eigen::Matrix3d& bRmi, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvm0, const Eigen::Vector3d& bam0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy);
  void update_strip_w_wdot(Eigen::Vector3d& omega_i, Eigen::Vector3d& omega_dot_i, const Eigen::Matrix3d& bRsi, const Eigen::Vector3d& b_omega_b_theta, const Eigen::Vector3d& b_omega_dot_b_theta, const Eigen::Vector3d& omega_phi_psi, const Eigen::Vector3d& omega_dot_phi_psi);

  template <const double (&CD)[176][14], const double (&CL)[176][14], const double (&CM)[176][14], const double (&X0)[176][14], const double (&ALPHA_STALL_POS)[14], const double (&ALPHA_STALL_NEG)[14], std::size_t N, typename RotationAt, typename OmegaAt, typename OmegaDotYAt, typename ChordAt, typename WidthAt>
  void update_segment_aerodynamics(const std::array<Eigen::Vector3d, 2*N>& p, const std::array<Eigen::Vector3d, 2*N>& v, const std::array<Eigen::Vector3d, 2*N>& a, std::size_t idx0, std::size_t state_idx0, std::size_t load_idx, RotationAt&& rotation_at, OmegaAt&& omega_at, OmegaDotYAt&& omega_dot_y_at, ChordAt&& chord_at, WidthAt&& width_at, bool update_telemetry);

  template <std::size_t N, typename RotationAt, typename OmegaAt, typename OmegaDotYAt, typename ChordAt, typename WidthAt>
  void update_full_added_mass_segment(const std::array<Eigen::Vector3d, 2*N>& v, const std::array<Eigen::Vector3d, 2*N>& a, std::size_t idx0, std::size_t state_idx0, RotationAt&& rotation_at, OmegaAt&& omega_at, OmegaDotYAt&& omega_dot_y_at, ChordAt&& chord_at, WidthAt&& width_at);
};
